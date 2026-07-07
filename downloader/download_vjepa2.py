#!/usr/bin/env python3
"""Download VJEPA-2 checkpoints used by WMReward.

Run from the WMReward repo root:
  python -u downloader/download_vjepa2.py --model vitg
"""

import argparse
import subprocess
from pathlib import Path


BASE_URL = "https://dl.fbaipublicfiles.com/vjepa2"
CHECKPOINTS = {
    "vith": "vith.pt",
    "vitg": "vitg.pt",
    "vitg384": "vitg-384.pt",
    "vitgac": "vjepa2-ac-vitg.pt",
}


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Skip existing file: {output_path}")
        return

    print(f"Downloading {url}")
    print(f"       -> {output_path}")
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
        str(output_path),
        url,
    ]
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download VJEPA-2 checkpoints")
    parser.add_argument(
        "--model",
        choices=[*CHECKPOINTS.keys(), "all"],
        default="vitg",
        help="Checkpoint to download. Default: vitg.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for downloaded checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = CHECKPOINTS.keys() if args.model == "all" else [args.model]
    for model in models:
        filename = CHECKPOINTS[model]
        download_file(f"{BASE_URL}/{filename}", args.output_dir / filename)


if __name__ == "__main__":
    main()
