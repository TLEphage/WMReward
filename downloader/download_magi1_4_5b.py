"""
下载 MAGI-1 4.5B_base、VAE、T5 权重到 ./downloads

运行：
nohup python -u downloader/download_magi1_4_5b.py > downloader/download_magi1_4_5b.log 2>&1 &

查看日志：
tail -f downloader/download_magi1_4_5b.log

查看目录大小：
watch -n 5 'du -sh downloads/4.5B_base downloads/vae downloads/t5_pretrained 2>/dev/null'
"""

import subprocess, os
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "sand-ai/MAGI-1"
# BASE_URL = "https://huggingface.co"
BASE_URL = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = BASE_URL
OUT_DIR = Path("downloads")
INCLUDE_PREFIXES = (
    "ckpt/magi/4.5B_base/",
    "ckpt/vae/",
    "ckpt/t5/",
)


def local_path(file_path: str) -> Path:
    if file_path.startswith("ckpt/magi/4.5B_base/"):
        return OUT_DIR / "4.5B_base" / file_path.removeprefix("ckpt/magi/4.5B_base/")
    if file_path.startswith("ckpt/vae/"):
        return OUT_DIR / "vae" / file_path.removeprefix("ckpt/vae/")
    if file_path.startswith("ckpt/t5/"):
        return OUT_DIR / "t5_pretrained" / file_path.removeprefix("ckpt/t5/")
    raise ValueError(f"Unexpected file path: {file_path}")


def download_file(file_path: str) -> None:
    final_path = local_path(file_path)
    part_path = final_path.with_suffix(final_path.suffix + ".part")
    final_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/{REPO_ID}/resolve/main/{file_path}"

    if final_path.exists() and final_path.stat().st_size > 0:
        print(f"[SKIP] {file_path}", flush=True)
        return

    print(f"[DOWN] {file_path}", flush=True)

    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--retry", "999",
            "--retry-delay", "10",
            "--retry-all-errors",
            "--connect-timeout", "60",
            "--max-time", "0",
            "-C", "-",
            "-o", str(part_path),
            url,
        ],
        check=True,
    )

    part_path.rename(final_path)
    print(f"[OK] {file_path}", flush=True)

def list_files_with_retry() -> list[str]:
    api = HfApi(endpoint=BASE_URL)

    for attempt in range(1, 1000):
        try:
            files = [
                file_path
                for file_path in api.list_repo_files(REPO_ID)
                if file_path.startswith(INCLUDE_PREFIXES)
            ]
            return files
        except Exception as e:
            print(f"[LIST-RETRY {attempt}] {repr(e)}", flush=True)
            time.sleep(10)

    raise RuntimeError("list_repo_files failed too many times")

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list_files_with_retry()
    print(f"共 {len(files)} 个文件", flush=True)

    for file_path in files:
        download_file(file_path)

    print(f"下载完成：{OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
