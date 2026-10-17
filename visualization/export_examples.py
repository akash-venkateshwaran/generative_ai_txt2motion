"""Render 5 fixed-duration (4 s) text-to-motion examples to visualization/renders/ as mp4."""
import os
import sys
from os.path import join as pjoin, dirname, abspath

sys.path.insert(0, dirname(dirname(abspath(__file__))))

from utils.inference import MotionGenerator

PROMPTS = [
    "a person walks forward and waves with their right hand",
    "a person jumps up and down on the spot",
    "a person kicks with their left leg",
    "a person walks in a circle",
    "a person raises both arms and then lowers them",
]
DURATION_S = 4
SEED = 42
OUT_DIR = pjoin(dirname(abspath(__file__)), "renders")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    gen = MotionGenerator()
    fps = gen.opt.fps
    n_frames = int(round(DURATION_S * fps))  # generated length is a multiple of 4; trim to exactly 4 s
    for i, text in enumerate(PROMPTS):
        joints, _ = gen.generate(text, DURATION_S, seed=SEED)
        gen.render(joints[:n_frames], text, pjoin(OUT_DIR, f"example_{i + 1}.mp4"), fps=fps)
        print(f"[{i + 1}/{len(PROMPTS)}] {text}")


if __name__ == "__main__":
    main()
