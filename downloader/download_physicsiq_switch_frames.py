#!/usr/bin/env python3
"""Download Physics-IQ switch-frame images needed for WMReward I2V runs.

This avoids installing gcloud. It downloads the files referenced by
prompts/physics_iq.json from the public GCS HTTPS endpoint.

Run from the WMReward repo root:
  python -u downloader/download_physicsiq_switch_frames.py --max_entries 8
"""

import argparse
import json
import subprocess
from pathlib import Path


BASE_URL = "https://storage.googleapis.com/physics-iq-benchmark"


def load_json_with_header_comments(path: Path) -> list[dict]:
    text = "".join(
        line for line in path.read_text().splitlines(keepends=True)
        if not line.lstrip().startswith("#")
    )
    return json.loads(text)


def remote_path(input_image: str) -> str:
    prefix = "physics-IQ-benchmark/"
    if input_image.startswith(prefix):
        return input_image.removeprefix(prefix)
    return input_image


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
    parser = argparse.ArgumentParser(description="Download Physics-IQ switch-frame images")
    parser.add_argument("--prompt_json", type=Path, default=Path("prompts/physics_iq.json"))
    parser.add_argument("--output_root", type=Path, default=Path("PhysicsIQ/code"))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_entries", type=int, default=0, help="0 downloads all referenced images.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = load_json_with_header_comments(args.prompt_json)
    entries = entries[args.start_idx:]
    if args.max_entries > 0:
        entries = entries[:args.max_entries]

    input_images = []
    seen = set()
    for entry in entries:
        input_image = entry.get("input_image")
        if input_image and input_image not in seen:
            seen.add(input_image)
            input_images.append(input_image)

    print(f"Downloading {len(input_images)} switch-frame images")
    for input_image in input_images:
        rel_remote = remote_path(input_image)
        url = f"{BASE_URL}/{rel_remote}"
        output_path = args.output_root / input_image
        download_file(url, output_path)

    print("Done.")


if __name__ == "__main__":
    main()
