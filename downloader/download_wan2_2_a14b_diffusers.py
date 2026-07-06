"""
Download Wan2.2 A14B Diffusers checkpoints to ../../models.

Default target:
  ../../models/Wan2.2-I2V-A14B-Diffusers

Run from the WMReward repo root:
  nohup python -u downloader/download_wan2_2_a14b_diffusers.py \
      > downloader/download_wan2_2_a14b_diffusers.log 2>&1 &

Download T2V instead:
  python -u downloader/download_wan2_2_a14b_diffusers.py --variant t2v

Download both I2V and T2V:
  python -u downloader/download_wan2_2_a14b_diffusers.py --variant both

View progress:
  tail -f downloader/download_wan2_2_a14b_diffusers.log
  watch -n 5 'du -sh ../../models/Wan2.2-*A14B-Diffusers 2>/dev/null'
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPOS = {
    "i2v": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "t2v": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
}
BASE_URL = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
OUT_ROOT = Path("../../models")


def _hf_api_and_token():
    try:
        from huggingface_hub import HfApi, get_token
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: huggingface_hub. Install it with "
            "`pip install huggingface_hub` or `pip install 'huggingface_hub[cli]'`."
        ) from exc

    token = os.environ.get("HF_TOKEN") or get_token()
    return HfApi(endpoint=BASE_URL, token=token), token


def get_files(repo_id: str) -> list[tuple[str, int | None]]:
    api, _ = _hf_api_and_token()
    info = api.repo_info(repo_id, files_metadata=True)
    return [(item.rfilename, getattr(item, "size", None)) for item in info.siblings]


def download_file(repo_id: str, out_dir: Path, file_path: str, expected_size: int | None) -> None:
    final_path = out_dir / file_path
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and expected_size is not None:
        current_size = final_path.stat().st_size
        if current_size >= expected_size:
            print(f"[SKIP] {repo_id}/{file_path} ({current_size}/{expected_size})", flush=True)
            return
        print(f"[RESUME] {repo_id}/{file_path} ({current_size}/{expected_size})", flush=True)
    elif final_path.exists() and final_path.stat().st_size > 0:
        print(f"[SKIP] {repo_id}/{file_path}", flush=True)
        return
    else:
        print(f"[DOWN] {repo_id}/{file_path}", flush=True)

    _, token = _hf_api_and_token()
    url = f"{BASE_URL}/{repo_id}/resolve/main/{file_path}"
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "999",
        "--retry-delay",
        "10",
        "--retry-all-errors",
        "--connect-timeout",
        "60",
        "--max-time",
        "0",
        "-C",
        "-",
        "-o",
        str(final_path),
    ]
    if token:
        cmd.extend(["-H", f"Authorization: Bearer {token}"])
    cmd.append(url)

    subprocess.run(cmd, check=True)
    print(f"[OK] {repo_id}/{file_path}", flush=True)


def download_repo(variant: str) -> None:
    repo_id = REPOS[variant]
    out_dir = OUT_ROOT / repo_id.split("/", 1)[1]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"REPO_ID={repo_id}", flush=True)
    print(f"BASE_URL={BASE_URL}", flush=True)
    print(f"OUT_DIR={out_dir}", flush=True)

    files = get_files(repo_id)
    print(f"共 {len(files)} 个文件", flush=True)
    for file_path, expected_size in files:
        download_file(repo_id, out_dir, file_path, expected_size)

    print(f"全部下载完成：{out_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Wan2.2 A14B Diffusers checkpoints.")
    parser.add_argument(
        "--variant",
        choices=["i2v", "t2v", "both"],
        default="i2v",
        help="i2v is the default because WMReward's PhysicsIQ path is image-conditioned.",
    )
    args = parser.parse_args()

    variants = ["i2v", "t2v"] if args.variant == "both" else [args.variant]
    for variant in variants:
        download_repo(variant)


if __name__ == "__main__":
    main()
