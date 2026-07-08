import csv
import os
from statistics import mean

from physiq.calculate_and_write_metrics_to_csv import ViewPaths, load_view, compute_view_metrics

OUT = "/root/physics-consistency-eval/WMReward/logs/physicsiq_singleview_0001_0005_3s"
FPS = 24
FRAMES = 72
TASKS = [
    {
        "task_id": "0001",
        "scenario": "trimmed-ball-and-block-fall.mp4",
        "view": "perspective-left",
        "take1_id": "0001",
        "take2_id": "0199",
        "gen_name": "0001_perspective-left_trimmed-ball-and-block-fall.mp4",
    },
    {
        "task_id": "0005",
        "scenario": "trimmed-ball-behind-rotating-paper.mp4",
        "view": "perspective-center",
        "take1_id": "0005",
        "take2_id": "0203",
        "gen_name": "0005_perspective-center_trimmed-ball-behind-rotating-paper.mp4",
    },
]
METHODS = ["gf5base", "promptx", "action_focus"]

real_dir = os.path.join(OUT, "real_videos_3s", "24FPS")
real_mask_dir = os.path.join(OUT, "video-masks", "real", "24FPS")
rows = []

for method in METHODS:
    gen_dir = os.path.join(OUT, "generated_videos_3s", method)
    gen_mask_dir = os.path.join(OUT, "video-masks", "generated", method, "24FPS")
    for task in TASKS:
        scenario = task["scenario"]
        view = task["view"]
        paths = ViewPaths(
            real_v1=os.path.join(real_dir, f"{task['take1_id']}_testing-videos_{FPS}FPS_{view}_take-1_{scenario}"),
            real_v2=os.path.join(real_dir, f"{task['take2_id']}_testing-videos_{FPS}FPS_{view}_take-2_{scenario}"),
            generated=os.path.join(gen_dir, task["gen_name"]),
            mask_v1=os.path.join(real_mask_dir, f"{task['take1_id']}_video-masks_{FPS}FPS_{view}_take-1_{scenario}"),
            mask_v2=os.path.join(real_mask_dir, f"{task['take2_id']}_video-masks_{FPS}FPS_{view}_take-2_{scenario}"),
            mask_generated=os.path.join(gen_mask_dir, f"{task['task_id']}_video-masks_{FPS}FPS_{view}_take-1_{scenario}"),
        )
        frames = load_view(paths, 0, FRAMES, FRAMES)
        metrics = compute_view_metrics(frames)
        row = {
            "method": method,
            "task_id": task["task_id"],
            "scenario": scenario,
            "view": view,
            "frames": FRAMES,
            "spatiotemporal_iou_v1_mean": mean(metrics["spatiotemporal_iou_v1"]),
            "spatiotemporal_iou_v2_mean": mean(metrics["spatiotemporal_iou_v2"]),
            "spatiotemporal_iou_mean": mean(metrics["spatiotemporal_iou_v1"] + metrics["spatiotemporal_iou_v2"]),
            "spatial_iou_v1": metrics["spatial_iou_v1"],
            "spatial_iou_v2": metrics["spatial_iou_v2"],
            "spatial_iou_mean": mean([metrics["spatial_iou_v1"], metrics["spatial_iou_v2"]]),
            "weighted_spatial_iou_v1": metrics["weighted_spatial_iou_v1"],
            "weighted_spatial_iou_v2": metrics["weighted_spatial_iou_v2"],
            "weighted_spatial_iou_mean": mean([metrics["weighted_spatial_iou_v1"], metrics["weighted_spatial_iou_v2"]]),
            "mse_v1_mean": mean(metrics["v1_mse"]),
            "mse_v2_mean": mean(metrics["v2_mse"]),
            "mse_mean": mean(metrics["v1_mse"] + metrics["v2_mse"]),
            "real_take_variance_mse_mean": mean(metrics["variance_mse"]),
            "real_take_variance_spatiotemporal_iou_mean": mean(metrics["variance_spatiotemporal_iou"]),
        }
        rows.append(row)

detail_csv = os.path.join(OUT, "physicsiq_singleview_detail_0001_0005_3s.csv")
with open(detail_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

summary_rows = []
for method in METHODS:
    method_rows = [r for r in rows if r["method"] == method]
    summary = {"method": method, "n_tasks": len(method_rows)}
    for key in [
        "spatiotemporal_iou_mean",
        "spatial_iou_mean",
        "weighted_spatial_iou_mean",
        "mse_mean",
        "real_take_variance_mse_mean",
        "real_take_variance_spatiotemporal_iou_mean",
    ]:
        summary[key] = mean(r[key] for r in method_rows)
    summary_rows.append(summary)

summary_csv = os.path.join(OUT, "physicsiq_singleview_summary_0001_0005_3s.csv")
with open(summary_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    writer.writeheader()
    writer.writerows(summary_rows)

print(detail_csv)
print(summary_csv)
for r in summary_rows:
    print(r)
