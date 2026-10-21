# MaskInversion – project journal

Running log of what was done, what broke, and why. Newest entries at the bottom.

- **Paper**: https://arxiv.org/html/2407.20034v2
- **Upstream repo**: https://github.com/WalBouss/MaskInversion
- **Package**: `maskinversion_torch` (PyPI) – wrapper around OpenCLIP + LeGrad

---

## 2026-10-21 – Local setup and first run on a test image

### Goal
Get MaskInversion running locally on GPU on the upstream cats-and-dogs test image.

### Machine
| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Ti, 16 GB (Blackwell, compute capability `sm_120`) |
| NVIDIA driver | 595.99.02 – supports up to CUDA 13.2 |
| System CUDA toolkit (`nvcc`) | 12.5 in `/usr/local/cuda-12.5` – **not used** by PyTorch |
| System Python | 3.14 (`/usr/bin/python3`, Ubuntu-managed) |

**CUDA takeaway:** PyTorch pip wheels ship their own CUDA runtime, so the system toolkit version does not matter; only the driver does. Blackwell (`sm_120`) needs **torch ≥ 2.7** built with CUDA ≥ 12.8. The default PyPI wheel (torch 2.14.1+cu130) works out of the box:
```python
torch.cuda.is_available()      # True
torch.cuda.get_arch_list()     # [..., 'sm_120']
```

### Environment
Created with `uv`, Python 3.10:
```bash
uv venv maskinversion_env --python 3.10
uv pip install --python maskinversion_env/bin/python -r requirements.txt
```

`requirements.txt` changes:
- `torch>=1.11` → `torch>=2.7` (minimum for the RTX 5070 Ti)
- added `maskinversion_torch`, `jupyterlab`, `ipykernel`
- fixed a missing trailing newline that merged two lines into `maskinversion_torchjupyterlab`

Key installed versions: torch 2.14.1+cu130, torchvision 0.29.1, open_clip_torch 2.23.0, legrad_torch 1.1, maskinversion_torch 1.1, timm 1.0.30, numpy 2.2.6, jupyterlab 4.6.4, ipykernel 7.3.0.

#### Lesson: don't use `pip` in a uv venv
`uv venv` creates the Python interpreter but **does not install pip** (unlike `python -m venv`). After activating the env, `pip` falls through `PATH` to `/usr/bin/pip` (system Python 3.14), which either refuses (`externally-managed-environment`, PEP 668) or installs into the wrong Python. `python -m pip` fails with `No module named pip`.

Use uv instead:
```bash
source maskinversion_env/bin/activate
uv pip install <package>
# or without activating
uv pip install --python maskinversion_env/bin/python <package>
```
(`uv venv --seed` would add pip, but it isn't needed.)

### Problems in the upstream example (`example_maskinversion.py`)
1. **Dead asset URLs** – the image/masks point to `github.com/WalBouss/Zero-shot-RIS/blob/...`, which returns 404 (and `/blob/` URLs return HTML, not images, anyway). The same files exist in the MaskInversion repo:
   `https://raw.githubusercontent.com/WalBouss/MaskInversion/main/assets/{cats_and_dogs.jpg, cat_mask.png, dog_mask.png}`
2. **RGBA masks crash the dice loss** – the mask PNGs are RGBA (480×639×4). `MaskInversionMaskPreprocess` just does `np.array(mask) / 255`, so the mask tensor is `[N, 480, 639, 4]`, and `expl_map` is interpolated to the last two dims `(639, 4)`:
   ```
   RuntimeError: The size of tensor a (2556) must match the size of tensor b (1226880)
   ```
   Fix: load masks as single channel with `Image.open(path).convert('L')` (the RGB channels are binary 0/255, alpha is constant).
3. **Region–text probabilities look ~50/50** – see the next section.

#### Lesson: CLIP logit scale
The example does `softmax(cosine_similarity)`. CLIP cosine similarities live in a narrow band (~0.15–0.35), so the softmax is nearly uniform (0.54 / 0.46 here). CLIP was trained on `logit_scale.exp() * cosine` (≈ 100 for OpenAI weights) before the softmax; with the scale the same scores give 1.000 / 0.000.

- **Not a bug in the method**: scaling by a positive constant doesn't change the ranking, so the argmax prediction, the optimized embeddings and the explainability maps are identical.
- **Matters when the probabilities are used**: thresholds, comparing confidence across regions, averaging over prompt templates.
- Use `model.logit_scale.exp()` rather than a hard-coded 100, since other checkpoints may learn a different scale.

### Script: `run_example.py`
Upstream example with the three fixes above, plus:
- downloads test assets to `data/` on first run;
- saves side-by-side figures (target mask | explainability map) to `outputs/` instead of opening windows (`matplotlib` Agg backend);
- CLI options for custom inputs and hyperparameters (`--image`, `--masks`, `--prompts`, `--model`, `--pretrained`, `--iterations`, `--lr`).

```bash
source maskinversion_env/bin/activate
python run_example.py
python run_example.py --image my.jpg --masks a.png b.png --prompts "a photo of x" "a photo of y"
```

`data/` and `outputs/` are git-ignored.

### Result
ViT-B-16 / `openai` weights, image size 448, 10 iterations, lr 0.5, AdamW:
```
device: cuda (NVIDIA GeForce RTX 5070 Ti)
loss 0.670 -> 0.246 over 10 iterations, 0.23 s total
mask 0 (cat_mask.png) -> a photo of a cat | cat: 1.000, dog: 0.000
mask 1 (dog_mask.png) -> a photo of a dog | cat: 0.000, dog: 1.000
```
Both regions are matched to the correct prompt, and the explainability maps closely follow the cat and dog masks (`outputs/maskinversion_0_cat_mask.png`, `outputs/maskinversion_1_dog_mask.png`).

### Next steps
- [ ] Run on my own test image (masks from SAM).
- [ ] Try other backbones (`maskinversion.list_pretrained()`) and the effect of `iterations` / `lr` / `alpha`.
- [ ] Explore the downstream uses from the paper: region classification, localized captioning, localized diffusion.
