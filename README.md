# MasInversion reimplementation

**paper** : https://arxiv.org/html/2407.20034v2
**repo** : https://github.com/WalBouss/MaskInversion/blob/main/README.md

### For the 22/10:
- Read the paper: MasInversion modify the embedding of images to a subregion of interest know by a mask (SAM). MOdifies the explainability map of CLIP models to resemble the mask (with dice loss and AdamW optimizer). What to do do with that vector? 
    - the vector can be used to classify a specific region of an image, be transformed into a localized captioning to direct the attention of an LLM to specific parts of an image, or in a localized diffusion where the embedding is used in conjunction with a diffusion model to generate variations of specific regions of images.

- Run the project locally

### To run maskinversion:
```bash
source maskinversion_env/bin/activate
python run_example.py
python run_example.py --image my.jpg --masks a.png b.png --prompts "a photo of x" "a photo of y"
```