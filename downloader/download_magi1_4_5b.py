"""
Download MAGI-1 4.5B checkpoints into the layout expected by MAGI-1 configs.

Run from the WMReward repo root:
  nohup python -u downloader/download_magi1_4_5b.py \
      > downloader/download_magi1_4_5b.log 2>&1 &

Download a lower-memory checkpoint:
  python -u downloader/download_magi1_4_5b.py --variant 4.5B_distill
  python -u downloader/download_magi1_4_5b.py --variant 4.5B_distill_quant

Download all 4.5B variants:
  python -u downloader/download_magi1_4_5b.py --variant all

View progress:
  tail -f downloader/download_magi1_4_5b.log
  watch -n 5 'du -sh downloads/4.5B_* downloads/vae downloads/t5_pretrained 2>/dev/null'
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ID = "sand-ai/MAGI-1"
BASE_URL = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
OUT_ROOT = Path("downloads")
VARIANTS = ("4.5B_base", "4.5B_distill", "4.5B_distill_quant")
SHARED_PREFIXES = ("ckpt/vae/", "ckpt/t5/")


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


def get_repo_files() -> list[tuple[str, int | None]]:
    api, _ = _hf_api_and_token()
    info = api.repo_info(REPO_ID, files_metadata=True)
    return [(item.rfilename, getattr(item, "size", None)) for item in info.siblings]


def wanted_prefixes(variants: list[str]) -> tuple[str, ...]:
    return tuple(f"ckpt/magi/{variant}/" for variant in variants) + SHARED_PREFIXES


def local_path(file_path: str) -> Path:
    if file_path.startswith("ckpt/magi/"):
        _, _, variant, remainder = file_path.split("/", 3)
        return OUT_ROOT / variant / remainder
    if file_path.startswith("ckpt/vae/"):
        return OUT_ROOT / "vae" / file_path.removeprefix("ckpt/vae/")
    if file_path.startswith("ckpt/t5/"):
        return OUT_ROOT / "t5_pretrained" / file_path.removeprefix("ckpt/t5/")
    raise ValueError(f"Unexpected file path: {file_path}")


def download_file(file_path: str, expected_size: int | None) -> None:
    final_path = local_path(file_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)

    if final_path.exists() and expected_size is not None:
        current_size = final_path.stat().st_size
        if current_size >= expected_size:
            print(f"[SKIP] {file_path} -> {final_path} ({current_size}/{expected_size})", flush=True)
            return
        print(f"[RESUME] {file_path} -> {final_path} ({current_size}/{expected_size})", flush=True)
    elif final_path.exists() and final_path.stat().st_size > 0:
        print(f"[SKIP] {file_path} -> {final_path}", flush=True)
        return
    else:
        print(f"[DOWN] {file_path} -> {final_path}", flush=True)

    _, token = _hf_api_and_token()
    url = f"{BASE_URL}/{REPO_ID}/resolve/main/{file_path}"
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
    print(f"[OK] {file_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MAGI-1 4.5B checkpoints.")
    parser.add_argument(
        "--variant",
        choices=[*VARIANTS, "all"],
        default="4.5B_base",
        help="Checkpoint variant to download. Shared VAE/T5 files are always included.",
    )
    args = parser.parse_args()

    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    prefixes = wanted_prefixes(variants)

    print(f"REPO_ID={REPO_ID}", flush=True)
    print(f"BASE_URL={BASE_URL}", flush=True)
    print(f"OUT_ROOT={OUT_ROOT}", flush=True)
    print(f"VARIANTS={','.join(variants)}", flush=True)

    files = [
        (file_path, expected_size)
        for file_path, expected_size in get_repo_files()
        if file_path.startswith(prefixes)
    ]
    print(f"共 {len(files)} 个文件", flush=True)

    for file_path, expected_size in files:
        download_file(file_path, expected_size)

    print(f"全部下载完成：{OUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
