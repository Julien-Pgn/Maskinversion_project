"""Run MaskInversion on a test image and save the mask / explainability-map overlays.

Adapted from the upstream example_maskinversion.py, with two fixes:
- the test assets are downloaded from the MaskInversion repo (the original URLs return 404);
- masks are converted to single-channel ('L'), otherwise RGBA masks break the dice loss.

Usage:
    python run_example.py
    python run_example.py --image my.jpg --masks a.png b.png --prompts "a photo of x" "a photo of y"
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # save figures to files instead of opening windows
import matplotlib.pyplot as plt
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer

from maskinversion import (MaskInversion, MaskInversionImagePreprocess, MaskInversionMaskPreprocess,
                           overlay_image_mask, overlay_image_expl_map)

ASSETS_URL = 'https://raw.githubusercontent.com/WalBouss/MaskInversion/main/assets'
DATA_DIR = Path('data')
OUT_DIR = Path('outputs')


def download_example_assets():
    DATA_DIR.mkdir(exist_ok=True)
    for name in ['cats_and_dogs.jpg', 'cat_mask.png', 'dog_mask.png']:
        path = DATA_DIR / name
        if not path.exists():
            r = requests.get(f'{ASSETS_URL}/{name}', timeout=30)
            r.raise_for_status()
            path.write_bytes(r.content)
    return (DATA_DIR / 'cats_and_dogs.jpg',
            [DATA_DIR / 'cat_mask.png', DATA_DIR / 'dog_mask.png'],
            ['a photo of a cat', 'a photo of a dog'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, help='input image (default: cats-and-dogs example)')
    parser.add_argument('--masks', type=Path, nargs='+', help='one binary mask per region')
    parser.add_argument('--prompts', nargs='+', help='text prompts to match the regions against')
    parser.add_argument('--model', default='ViT-B-16')
    parser.add_argument('--pretrained', default='openai')
    parser.add_argument('--iterations', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.5)
    args = parser.parse_args()

    if args.image is None:
        image_path, mask_paths, prompts = download_example_assets()
    else:
        image_path, mask_paths, prompts = args.image, args.masks, args.prompts

    # ------ init model ------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}' + (f' ({torch.cuda.get_device_name(0)})' if device == 'cuda' else ''))
    model, _, preprocess = create_model_and_transforms(model_name=args.model, pretrained=args.pretrained, device=device)
    tokenizer = get_tokenizer(model_name=args.model)

    # ------ use MaskInversion wrapper ------
    model = MaskInversion(model, lr=args.lr, iterations=args.iterations)
    preprocess = MaskInversionImagePreprocess(preprocess, image_size=448)
    mask_preprocess = MaskInversionMaskPreprocess()

    # ------ inputs ------
    img_pil = Image.open(image_path).convert('RGB')
    image = preprocess(img_pil).unsqueeze(0).to(device)
    masks = torch.stack([mask_preprocess(Image.open(p).convert('L')) for p in mask_paths]).to(device)  # [num_masks, H, W]

    text_embeddings = F.normalize(model.encode_text(tokenizer(prompts).to(device)), dim=-1)  # [num_prompts, dim]

    # ------ compute localized embedding for each mask ------
    localized_embeddings, expl_map = model.compute_maskinversion(image=image, masks_target=masks, verbose=True,
                                                                 return_expl_map=True)
    localized_embeddings = F.normalize(localized_embeddings, dim=-1)  # [num_masks, dim]

    # ------ region-text matching ------
    # scale by CLIP's logit scale (~100) so the softmax is not flattened around 1/num_prompts
    logits = model.logit_scale.exp() * localized_embeddings @ text_embeddings.T  # [num_masks, num_prompts]
    probs = logits.softmax(dim=-1)

    OUT_DIR.mkdir(exist_ok=True)
    for i, mask in enumerate(masks.cpu().numpy()):
        matched = prompts[probs[i].argmax()]
        print(f'mask {i} ({mask_paths[i].name}) -> {matched} | '
              + ', '.join(f'{p}: {v:.3f}' for p, v in zip(prompts, probs[i].tolist())))

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(overlay_image_mask(image=img_pil, mask=mask, show=False))
        ax[0].set_title('target mask')
        ax[1].imshow(overlay_image_expl_map(image=img_pil, expl_map=expl_map[0, i], show=False))
        ax[1].set_title(f'explainability map -> {matched}')
        for a in ax:
            a.axis('off')
        fig.tight_layout()
        out_path = OUT_DIR / f'maskinversion_{i}_{mask_paths[i].stem}.png'
        fig.savefig(out_path)
        plt.close(fig)
        print(f'saved {out_path}')


if __name__ == '__main__':
    main()
