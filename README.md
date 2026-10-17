# SALAD: Skeleton-Aware Latent Diffusion for Text-driven Motion Generation and Editing

Text-to-motion generation on KIT-ML, based on **SALAD** (CVPR 2025).
See the [paper](https://arxiv.org/abs/2503.13836) and [project page](https://seokhyeonhong.github.io/projects/salad/) for the method, figures and results.

Original authors: Seokhyeon Hong, Chaelin Kim, Serin Yoon, Junghyun Nam, Sihun Cha, Junyong Noh.

## Overview

Generation is a two-stage pipeline:

```
text ─▶ Denoiser (50 DDIM steps from noise) ─▶ latent ─▶ VAE decoder ─▶ motion
```

| Model | Role | Tensors (B = batch, T = frames) |
|---|---|---|
| **VAE** (`models/vae`) | Compresses a motion clip into a small skeleto-temporal latent | `[B, T, 251]` → `[B, T/4, 7, 32]` → `[B, T, 251]` |
| **Denoiser** (`models/denoiser`) | Diffusion model that denoises the latent, conditioned on text and timestep | `[B, T/4, 7, 32]` → `[B, T/4, 7, 32]` |

## Architecture

### VAE

```
motion [B,T,251] ─ MotionEncoder ─▶ [B,T,21,32] ─ STConvEncoder ─▶ [B,T/4,7,32] ─ dist ─▶ mu, logvar ─▶ z
z [B,T/4,7,32]   ─ STConvDecoder ─▶ [B,T,21,32] ─ MotionDecoder ─▶ motion [B,T,251]
```

- **MotionEncoder / MotionDecoder**: the 251 KIT-ML features are regrouped by joint and each of the 21 joints gets its own small MLP, producing one 32-d token per joint.
- **STConv**: the core block, a sum of two branches: a graph convolution over the skeleton (each joint mixes with its neighbours) and a temporal `Conv1d` per joint. Wrapped in residual blocks (`ResSTConv`).
- **STPool / STUnpool**: parameter-free pooling that coarsens the skeleton (21 → 12 → 7 body parts: root, two legs, spine, two arms, head) while halving the frame rate at each stage. The decoder mirrors this.
- **Latent distribution**: one linear layer per body part outputs `mu` and `logvar`; `z` is sampled with the reparameterisation trick. The latent is about 8x smaller than the input.

### Denoiser

A transformer over a grid of tokens, `T/4` frames × 7 body parts, that predicts the denoising target for every token.

```
noisy latent ─ InputProcess (MLP) ─ + positional emb ──┐
timestep t   ─ sinusoidal + MLP ─▶ [B,256] ── FiLM in every layer
text         ─ frozen CLIP ─ Linear ─▶ [B,77,256] ── cross-attention
                                                        ▼
                                  SkipTransformer (5 STTransformerLayers)
                                                        ▼
                                           OutputProcess (MLP) ─▶ prediction
```

- **SkipTransformer**: U-Net-style, with 2 input blocks, a middle block and 2 output blocks. Activations from the input blocks are concatenated back in before the matching output blocks.
- **STTransformerLayer**: four residual sub-blocks in order: temporal attention (over frames, per body part), skeletal attention (over body parts, per frame), cross-attention to the text, and a feed-forward network. Each is modulated by FiLM from the timestep embedding.
- **Diffusion**: trained with `v_prediction` over 1000 timesteps. Sampling uses 50 DDIM steps with classifier-free guidance (scale 7.5), calling the denoiser with an empty prompt and with the real prompt at every step.

`visualization/visualize_architecture.ipynb` draws both models with `torchview` and walks through each step with shapes and parameter counts.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                     # core dependencies
uv sync --group dev         # + notebooks, moviepy, gdown (for the download scripts)
uv sync --group app         # + gradio (web demo)
```

## Data and pre-trained weights

Download [KIT-ML](https://github.com/EricGuo5513/HumanML3D) and place or link it as:

```
salad
└─ dataset
    └─ KIT-ML
```

Download the pre-trained weights, evaluation models and GloVe embeddings:

```bash
bash prepare/download_kit.sh
bash prepare/download_glove.sh      # needed for evaluation
```

## Usage

### Generate motion

```python
from utils.inference import MotionGenerator
```

`MotionGenerator` wraps the pretrained VAE and denoiser for text → motion. `visualization/export_examples.py` renders a few example prompts to `visualization/renders/`.

### Visualisation notebooks (`visualization/`)

- `visualize_architecture.ipynb`: model diagrams and step-by-step walkthrough
- `visualize_attn.ipynb`: attention maps produced during generation
- `visualize_kit.ipynb`: KIT-ML data

### Training

```bash
uv run python train_vae.py --name vae_example
uv run python train_denoiser.py --name denoiser_example --vae_name vae_example
```

### Evaluation

```bash
uv run python test_vae.py --name vae_example
uv run python test_denoiser.py --name denoiser_example --num_inference_timesteps 50
```

Results are written to `checkpoints/{dataset}/{vae_name | denoiser_name}/eval/eval.log`.

## Citation

```bibtex
@inproceedings{hong2025salad,
  title     = {SALAD: Skeleton-aware Latent Diffusion for Text-driven Motion Generation and Editing},
  author    = {Hong, Seokhyeon and Kim, Chaelin and Yoon, Serin and Nam, Junghyun and Cha, Sihun and Noh, Junyong},
  booktitle = {CVPR},
  year      = {2025}
}
```
