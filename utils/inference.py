"""
Minimal text -> motion inference wrapper around the pretrained SALAD KIT-ML
VAE + Denoiser checkpoints. Used by app.py (the Gradio demo).

Weights are pulled from the Hugging Face Hub model repo (see HF_MODEL_REPO
below) rather than read from a local ./checkpoints/ directory, so the app can
be deployed as a HF Space without bundling ~44MB of checkpoint files in the
Space's git repo.
"""
import os
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from huggingface_hub import snapshot_download

from models.vae.model import VAE
from models.denoiser.model import Denoiser
from utils.get_opt import get_opt
from utils.motion_process import recover_from_ric
from utils.plot_script import plot_3d_motion
from utils import paramUtil

# Repo layout (flat, uploaded via `hf upload`):
#   vae/{opt.txt, net_best_fid.tar}
#   denoiser/{opt.txt, net_best_fid.tar}
#   eval_stats/{mean.npy, std.npy}
HF_MODEL_REPO = os.environ.get("SALAD_MODEL_REPO", "akashvenkateshwaran/txt2motion")


def _lengths_to_mask(lengths):
    max_frames = torch.max(lengths)
    return torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)


class MotionGenerator:
    def __init__(self, device=None, repo_id=HF_MODEL_REPO):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        local_dir = snapshot_download(repo_id=repo_id)

        self.vae_opt = get_opt(pjoin(local_dir, "vae", "opt.txt"), self.device)
        self.vae = VAE(self.vae_opt)
        vae_ckpt = torch.load(pjoin(local_dir, "vae", "net_best_fid.tar"), map_location="cpu")
        self.vae.load_state_dict(vae_ckpt["vae"])
        self.vae.freeze()
        self.vae = self.vae.to(self.device)

        self.opt = get_opt(pjoin(local_dir, "denoiser", "opt.txt"), self.device)
        self.denoiser = Denoiser(self.opt, vae_dim=self.vae_opt.latent_dim)
        den_ckpt = torch.load(pjoin(local_dir, "denoiser", "net_best_fid.tar"), map_location="cpu")
        missing, unexpected = self.denoiser.load_state_dict(den_ckpt["denoiser"], strict=False)
        assert len(unexpected) == 0
        assert all(k.startswith("clip_model.") for k in missing)
        self.denoiser = self.denoiser.to(self.device).eval()

        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.opt.num_train_timesteps,
            beta_start=self.opt.beta_start,
            beta_end=self.opt.beta_end,
            beta_schedule=self.opt.beta_schedule,
            prediction_type=self.opt.prediction_type,
            clip_sample=False,
        )

        # The VAE/Denoiser pair was trained on motion normalized with the Comp_v6_KLD005
        # evaluator's stats, not the raw dataset/KIT-ML/Mean.npy & Std.npy.
        self.mean = np.load(pjoin(local_dir, "eval_stats", "mean.npy"))
        self.std = np.load(pjoin(local_dir, "eval_stats", "std.npy"))

    @torch.no_grad()
    def generate(self, text: str, duration_s: float, cond_scale: float = 7.5,
                 num_inference_timesteps: int = 50, seed: int | None = None):
        if seed is not None:
            torch.manual_seed(seed)

        opt = self.opt
        opt.cond_scale = cond_scale
        opt.num_inference_timesteps = num_inference_timesteps
        opt.classifier_free_guidance = cond_scale > 1.0

        n_frames = int(round(duration_s * opt.fps / opt.unit_length) * opt.unit_length)
        n_frames = max(opt.unit_length, min(n_frames, opt.max_motion_length))

        motion_dummy = torch.zeros(1, opt.max_motion_length, self.vae_opt.pose_dim, device=self.device)
        m_lens = torch.tensor([n_frames], device=self.device)
        len_mask = _lengths_to_mask(m_lens // opt.unit_length)

        input_text = [""]
        if opt.classifier_free_guidance:
            input_text.append(text)
        else:
            input_text = [text]

        z, _ = self.vae.encode(motion_dummy)
        latents = torch.randn_like(z) * self.scheduler.init_noise_sigma
        len_mask = F.pad(len_mask, (0, latents.shape[1] - len_mask.shape[1]), value=False)
        latents = latents * len_mask[..., None, None].float()

        self.scheduler.set_timesteps(opt.num_inference_timesteps)
        for t in self.scheduler.timesteps.to(self.device):
            if opt.classifier_free_guidance:
                in_latents = torch.cat([latents] * 2, dim=0)
                in_mask = torch.cat([len_mask] * 2, dim=0)
            else:
                in_latents, in_mask = latents, len_mask

            pred, _ = self.denoiser.forward(in_latents, t, input_text, len_mask=in_mask,
                                             need_attn=False, use_cached_clip=True)
            if opt.classifier_free_guidance:
                pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
                pred = pred_uncond + opt.cond_scale * (pred_cond - pred_uncond)

            latents = self.scheduler.step(pred, t, latents).prev_sample
            latents = latents * len_mask[..., None, None].float()

        self.denoiser.remove_clip_cache()

        pred_motion = self.vae.decode(latents)[0].detach().cpu().numpy()
        feat = pred_motion[:n_frames] * self.std + self.mean
        joints = recover_from_ric(torch.from_numpy(feat).float(), self.vae_opt.joints_num).numpy()
        return joints, feat

    def render(self, joints, text, save_path, fps=None):
        plot_3d_motion(save_path, paramUtil.kit_kinematic_chain, joints, title=text,
                        fps=fps or self.opt.fps, radius=self.opt.radius)
        return save_path
