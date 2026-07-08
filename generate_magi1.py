# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s


"""
MAGI-1 single-prompt I2V generation script with VJEPA guidance support.
This is the main quick-start script for video generation using the MAGI-1 submodule.
"""

import os
import sys
import argparse
import shutil
import subprocess
from pathlib import Path

from prompt_expansion import PromptExpansionConfig, expand_prompt

# Add MAGI-1 submodule to path
MAGI1_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MAGI-1")
MAGI1_MODEL_VARIANTS = ("4.5B_base", "4.5B_distill", "4.5B_distill_quant", "24B_base")
DEFAULT_MAGI1_MODEL_VARIANT = os.environ.get("MAGI1_MODEL_VARIANT", "4.5B_base")


def default_magi1_config(model_variant: str) -> str:
    if model_variant.startswith("4.5B_"):
        return os.path.join(MAGI1_PATH, "example", "4.5B", f"{model_variant}_config.json")
    if model_variant == "24B_base":
        return os.path.join(MAGI1_PATH, "example", "24B", "24B_base_config.json")
    raise ValueError(f"Unsupported MAGI-1 model variant: {model_variant}")


def ensure_magi1_submodule() -> None:
    if not os.path.exists(MAGI1_PATH) or len(os.listdir(MAGI1_PATH)) == 0:
        raise RuntimeError(
            "MAGI-1 submodule not found. Please initialize it with:\n"
            "  git submodule update --init --recursive"
        )


def ensure_video_output_path(output_path: str) -> None:
    output_parent = Path(output_path).expanduser().resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg executable was not found. Install it with:\n"
            "  apt-get install -y ffmpeg\n"
            "or:\n"
            "  conda install -c conda-forge ffmpeg -y"
        )

    test_path = output_parent / ".ffmpeg_write_test.mp4"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=16x16:d=0.04:r=24",
        "-frames:v",
        "1",
        str(test_path),
    ]
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    finally:
        test_path.unlink(missing_ok=True)

    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip() or "no ffmpeg stderr captured"
        raise RuntimeError(f"ffmpeg cannot write mp4 files to {output_parent}:\n{stderr}")

# Set SPECIAL_TOKEN_PATH for MAGI-1 if not already set
os.environ.setdefault("SPECIAL_TOKEN_PATH", os.path.join(MAGI1_PATH, "example/assets/special_tokens.npz"))
os.environ.setdefault("PAD_HQ", "1")
os.environ.setdefault("PAD_DURATION", "1")
os.environ.setdefault("OFFLOAD_T5_CACHE", "true")
os.environ.setdefault("OFFLOAD_VAE_CACHE", "true")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "6009")
os.environ.setdefault("GPUS_PER_NODE", "1")
os.environ.setdefault("NNODES", "1")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")

if MAGI1_PATH not in sys.path:
    sys.path.insert(0, MAGI1_PATH)


def maybe_expand_prompt(prompt: str, args) -> str:
    if not getattr(args, "enable_prompt_expansion", False):
        return prompt
    expanded = expand_prompt(
        prompt,
        PromptExpansionConfig(
            mode=getattr(args, "prompt_expansion_mode", "coect"),
            max_events=getattr(args, "prompt_expansion_max_events", 4),
        ),
    )
    if expanded != prompt:
        print(f"Original prompt: {prompt}")
        print(f"Expanded prompt: {expanded}")
    return expanded


def main():
    parser = argparse.ArgumentParser(description="MAGI-1 I2V generation with VJEPA guidance")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt describing the video")
    parser.add_argument(
        "--magi_model_variant",
        type=str,
        default=DEFAULT_MAGI1_MODEL_VARIANT,
        choices=MAGI1_MODEL_VARIANTS,
        help="MAGI-1 checkpoint/config variant used when --config_file is not set.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Path to MAGI-1 configuration JSON file. Defaults to MAGI-1 4.5B base.",
    )
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--enable_prompt_expansion", action="store_true", help="Enable deterministic event-centric prompt expansion before generation.")
    parser.add_argument("--prompt_expansion_mode", type=str, default="coect", help="Prompt expansion template mode: coect, soft_stage, or action_focus. All modes are deterministic/template-based.")
    parser.add_argument("--prompt_expansion_max_events", type=int, default=4, help="Maximum number of event clauses added by prompt expansion.")
    parser.add_argument("--guidance_scale", type=float, default=0.001, help="VJEPA guidance scale.")
    parser.add_argument("--guidance_frequency", type=int, default=5, help="VJEPA guidance frequency.")
    parser.add_argument("--vjepa_guidance_max_calls", type=int, default=int(os.environ.get("VJEPA_GUIDANCE_MAX_CALLS", "0")), help="Maximum number of V-JEPA guidance applications per generated video. 0 means unlimited.")
    parser.add_argument("--enable_adaptive_guidance", action="store_true", help="Apply the capped V-JEPA guidance call in a middle denoising window instead of the first eligible step.")
    parser.add_argument("--adaptive_guidance_t_min", type=float, default=float(os.environ.get("VJEPA_ADAPTIVE_T_MIN", "0.25")), help="Lower t bound for adaptive one-shot guidance.")
    parser.add_argument("--adaptive_guidance_t_max", type=float, default=float(os.environ.get("VJEPA_ADAPTIVE_T_MAX", "0.75")), help="Upper t bound for adaptive one-shot guidance.")
    parser.add_argument("--adaptive_guidance_target_t", type=float, default=float(os.environ.get("VJEPA_ADAPTIVE_TARGET_T", "0.50")), help="Preferred t value for adaptive one-shot guidance.")
    parser.add_argument("--vjepa_guidance_target_fps", type=int, default=int(os.environ.get("VJEPA_GUIDANCE_TARGET_FPS", "16")), help="Temporal FPS used inside V-JEPA guidance. Lower values reduce guidance memory.")
    parser.add_argument("--vjepa_type", type=str, default="vitg", help="VJEPA model variant.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["t2v", "i2v", "v2v"],
        default="i2v",
        help="Generation mode: t2v (text-to-video), i2v (image-to-video), v2v (video-to-video)",
    )
    parser.add_argument("--init_image", type=str, default=None, help="Path to initial image for I2V mode")
    parser.add_argument("--init_video", type=str, default=None, help="Path to prefix video for V2V mode")
    args = parser.parse_args()
    args.config_file = args.config_file or default_magi1_config(args.magi_model_variant)
    args.prompt = maybe_expand_prompt(args.prompt, args)
    ensure_magi1_submodule()
    ensure_video_output_path(args.output_path)

    # Import MAGI-1 pipeline (after sys.path modification). The guidance
    # pipeline eagerly loads V-JEPA, so use the vanilla pipeline for smoke tests.
    if args.guidance_scale > 0:
        from inference.pipeline.pipeline_w_guidance import MagiPipeline
    else:
        from inference.pipeline.pipeline import MagiPipeline

    # Initialize MAGI-1 pipeline
    pipeline = MagiPipeline(args.config_file)
    if args.guidance_scale > 0:
        os.environ["VJEPA_GUIDANCE_MAX_CALLS"] = str(int(args.vjepa_guidance_max_calls or 0))
        os.environ["VJEPA_GUIDANCE_ADAPTIVE"] = "1" if args.enable_adaptive_guidance else "0"
        os.environ["VJEPA_ADAPTIVE_T_MIN"] = str(args.adaptive_guidance_t_min)
        os.environ["VJEPA_ADAPTIVE_T_MAX"] = str(args.adaptive_guidance_t_max)
        os.environ["VJEPA_ADAPTIVE_TARGET_T"] = str(args.adaptive_guidance_target_t)
        os.environ["VJEPA_GUIDANCE_TARGET_FPS"] = str(int(args.vjepa_guidance_target_fps or 16))
        pipeline.guidance_scale = args.guidance_scale
        pipeline.guidance_frequency = args.guidance_frequency
        pipeline.vjepa_type = args.vjepa_type

    # Run the appropriate mode
    if args.mode == "t2v":
        pipeline.run_text_to_video(prompt=args.prompt, output_path=args.output_path)
    elif args.mode == "i2v":
        if not args.init_image:
            print("Error: --init_image is required for i2v mode.")
            sys.exit(1)
        pipeline.run_image_to_video(prompt=args.prompt, image_path=args.init_image, output_path=args.output_path)
    elif args.mode == "v2v":
        if not args.init_video:
            print("Error: --init_video is required for v2v mode.")
            sys.exit(1)
        pipeline.run_video_to_video(prompt=args.prompt, prefix_video_path=args.init_video, output_path=args.output_path)

    print(f"Saved: {args.output_path}")


if __name__ == "__main__":
    main()
