# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s

"""
Sample script to compute VJEPA surprise (World Model Reward) for videos.

Usage:
    python compute_wmreward.py --video_path /path/to/video.mp4
"""

import copy
import os
import torch
import argparse
from pathlib import Path
from torchvision.transforms.functional import resize
from utils import (
    compute_vjepa_loss_sliding_window,
    get_video,
    load_vjepa_model_source,
)

VJEPA_BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
VJEPA_CHECKPOINT_FILES = {
    "vith": "vith.pt",
    "vitg": "vitg.pt",
    "vitg384": "vitg-384.pt",
    "vitgac": "vjepa2-ac-vitg.pt",
}


def _local_vjepa_checkpoint(model_name: str) -> Path:
    checkpoint_dir = Path(os.environ.get("VJEPA_CHECKPOINT_DIR", "./checkpoints"))
    return checkpoint_dir / VJEPA_CHECKPOINT_FILES[model_name]


def _patch_torchhub_vjepa_url() -> None:
    """Fix cached VJEPA2 hub code if it points to the localhost test server."""
    hub_dir = Path(torch.hub.get_dir())
    for backbones_py in hub_dir.glob("facebookresearch_vjepa2_*/src/hub/backbones.py"):
        text = backbones_py.read_text()
        patched = text.replace("http://localhost:8300", VJEPA_BASE_URL)
        if patched != text:
            backbones_py.write_text(patched)
            print(f"Patched VJEPA torch.hub checkpoint URL in {backbones_py}")


def load_vjepa_models(model_name="vitg"):
    """Load VJEPA models from torchhub."""
    img_size = 384 if "384" in model_name else 256

    checkpoint_path = _local_vjepa_checkpoint(model_name)
    if checkpoint_path.exists():
        print(f"Loading VJEPA checkpoint from local path: {checkpoint_path}")
        return load_vjepa_model_source(model_name)

    _patch_torchhub_vjepa_url()

    if model_name == "vith":
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_huge")
    elif model_name == "vitg":
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant")
    elif model_name == "vitg384":
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_vit_giant_384")
    elif model_name == "vitgac":
        encoder, predictor = torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    target_encoder = copy.deepcopy(encoder)
    return encoder, target_encoder, predictor, img_size


def load_video_as_tensor(video_path, max_frames=49, img_size=256):
    """Load video and convert to tensor [1, C, T, H, W] in range [-1, 1]."""
    video_np = get_video(video_path, max_frames=max_frames)
    video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2).float()
    video_tensor = resize(video_tensor.permute(1, 0, 2, 3), [img_size, img_size])
    video_tensor = video_tensor.permute(1, 0, 2, 3)
    video_tensor = (video_tensor / 127.5) - 1.0
    return video_tensor.unsqueeze(0)


def compute_vjepa_surprise(
    video_path: str,
    model_name: str = "vitg",
    window_size: int = 16,
    context_frames: int = 8,
    stride: int = 2,
    seed: int = 42,
):
    """Compute VJEPA surprise score for a video."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading VJEPA model: {model_name}...")
    encoder, target_encoder, predictor, img_size = load_vjepa_models(model_name)
    encoder = encoder.to(device).eval()
    target_encoder = target_encoder.to(device).eval()
    predictor = predictor.to(device).eval()

    print(f"Loading video: {video_path}")
    video_tensor = load_video_as_tensor(video_path, max_frames=49, img_size=img_size)
    video_tensor = video_tensor.to(device)
    print(f"Video tensor shape: {video_tensor.shape}")

    print("Computing VJEPA surprise...")
    with torch.no_grad():
        loss = compute_vjepa_loss_sliding_window(
            video_tensor=video_tensor,
            encoder=encoder,
            target_encoder=target_encoder,
            predictor=predictor,
            img_size=img_size,
            window_size=window_size,
            loss_exp=2,
            masking_mode="causal",
            context_frames=context_frames,
            is_vae_output=True,
            seed=seed,
            stride=stride,
            mode="mean",
        )

    surprise_score = loss.item()
    print(f"\n{'='*50}")
    print(f"VJEPA Surprise Score: {surprise_score:.6f}")
    print(f"VJEPA Similarity Score: {1.0 - surprise_score:.6f}")
    print(f"{'='*50}")

    return surprise_score


def main():
    parser = argparse.ArgumentParser(description="Compute VJEPA surprise for videos")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video")
    parser.add_argument("--model", type=str, default="vitg",
                        choices=["vith", "vitg", "vitg384", "vitgac"],
                        help="VJEPA model variant")
    parser.add_argument("--window_size", type=int, default=16, help="Sliding window size")
    parser.add_argument("--context_frames", type=int, default=8, help="Context frames per window")
    parser.add_argument("--stride", type=int, default=8, help="Sliding window stride")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    surprise_score = compute_vjepa_surprise(
        video_path=args.video_path,
        model_name=args.model,
        window_size=args.window_size,
        context_frames=args.context_frames,
        stride=args.stride,
        seed=args.seed,
    )



if __name__ == "__main__":
    main()
