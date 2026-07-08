import argparse
import csv
import os
import subprocess
from statistics import mean

import pandas as pd
from physiq.binary_mask_generator import generate_binary_masks
from physiq.calculate_and_write_metrics_to_csv import ViewPaths, compute_view_metrics, load_view

TASKS = {
    "0001": {
        "scenario": "trimmed-ball-and-block-fall.mp4",
        "view": "perspective-left",
        "take1_id": "0001",
        "take2_id": "0199",
    },
    "0005": {
        "scenario": "trimmed-ball-behind-rotating-paper.mp4",
        "view": "perspective-center",
        "take1_id": "0005",
        "take2_id": "0203",
    },
}


def run(cmd):
    subprocess.run(cmd, check=True)


def md5(path):
    out = subprocess.check_output(["md5sum", path], text=True)
    return out.split()[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/root/physics-consistency-eval/WMReward/logs/physicsiq_singleview_0001_0005_3s")
    p.add_argument("--input-video", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--task-id", choices=sorted(TASKS), required=True)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seconds", type=int, default=3)
    args = p.parse_args()

    task = TASKS[args.task_id]
    frames = args.fps * args.seconds
    scenario = task["scenario"]
    view = task["view"]

    eval_dir = os.path.join(args.out, "generated_videos_3s", args.method)
    mask_dir = os.path.join(args.out, "video-masks", "generated", args.method, f"{args.fps}FPS")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    gen_name = f"{args.task_id}_{view}_{scenario}"
    gen_path = os.path.join(eval_dir, gen_name)
    run(["ffmpeg", "-y", "-v", "error", "-i", args.input_video, "-t", str(args.seconds), "-r", str(args.fps), "-an", "-c:v", "mpeg4", gen_path])
    generate_binary_masks(eval_dir, mask_dir, False)

    real_dir = os.path.join(args.out, "real_videos_3s", f"{args.fps}FPS")
    real_mask_dir = os.path.join(args.out, "video-masks", "real", f"{args.fps}FPS")
    real_v1 = os.path.join(real_dir, f"{task['take1_id']}_testing-videos_{args.fps}FPS_{view}_take-1_{scenario}")
    real_v2 = os.path.join(real_dir, f"{task['take2_id']}_testing-videos_{args.fps}FPS_{view}_take-2_{scenario}")
    paths = ViewPaths(
        real_v1=real_v1,
        real_v2=real_v2,
        generated=gen_path,
        mask_v1=os.path.join(real_mask_dir, f"{task['take1_id']}_video-masks_{args.fps}FPS_{view}_take-1_{scenario}"),
        mask_v2=os.path.join(real_mask_dir, f"{task['take2_id']}_video-masks_{args.fps}FPS_{view}_take-2_{scenario}"),
        mask_generated=os.path.join(mask_dir, f"{args.task_id}_video-masks_{args.fps}FPS_{view}_take-1_{scenario}"),
    )
    loaded = load_view(paths, 0, frames, frames)
    metrics = compute_view_metrics(loaded)
    row = {
        "method": args.method,
        "task_id": args.task_id,
        "scenario": scenario,
        "view": view,
        "frames": frames,
        "real_take1_md5": md5(real_v1),
        "real_take2_md5": md5(real_v2),
        "spatiotemporal_iou_mean": mean(metrics["spatiotemporal_iou_v1"] + metrics["spatiotemporal_iou_v2"]),
        "spatial_iou_mean": mean([metrics["spatial_iou_v1"], metrics["spatial_iou_v2"]]),
        "weighted_spatial_iou_mean": mean([metrics["weighted_spatial_iou_v1"], metrics["weighted_spatial_iou_v2"]]),
        "mse_mean": mean(metrics["v1_mse"] + metrics["v2_mse"]),
        "real_take_variance_mse_mean": mean(metrics["variance_mse"]),
        "real_take_variance_spatiotemporal_iou_mean": mean(metrics["variance_spatiotemporal_iou"]),
    }

    local_csv = os.path.join(args.out, f"physicsiq_singleview_{args.method}_{args.task_id}_3s.csv")
    with open(local_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    detail_csv = os.path.join(args.out, "physicsiq_singleview_detail_0001_0005_3s.csv")
    if os.path.exists(detail_csv):
        detail = pd.read_csv(detail_csv)
        compare = pd.concat([detail[detail["task_id"].astype(str).str.zfill(4) == args.task_id], pd.DataFrame([row])], ignore_index=True)
        compare_csv = os.path.join(args.out, f"physicsiq_singleview_compare_{args.task_id}_with_{args.method}_3s.csv")
        compare.to_csv(compare_csv, index=False)
        print(compare_csv)
        print(compare[["method", "spatiotemporal_iou_mean", "spatial_iou_mean", "weighted_spatial_iou_mean", "mse_mean"]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(local_csv)
    if row["real_take1_md5"] == row["real_take2_md5"]:
        print(f"WARNING: real take-1 and take-2 are byte-identical for task {args.task_id}; variance metrics are not informative.")


if __name__ == "__main__":
    main()
