# PhysicsIQ single-view subset results: 0001/0005, 3s

This is not the official multi-view Physics-IQ leaderboard score. The available generated videos for gf5base/promptx/action_focus contain only one view per task and are 24 FPS / 3 seconds, so this run compares the available single view against the corresponding original PhysicsIQ real take-1 and take-2 clips truncated to 24 FPS / 3 seconds.

Higher is better for IoU metrics. Lower is better for MSE.

## Summary over 0001 and 0005

| method | spatiotemporal_iou_mean | spatial_iou_mean | weighted_spatial_iou_mean | mse_mean |
|---|---:|---:|---:|---:|
| gf5base | 0.029947 | 0.360121 | 0.116158 | 0.013405 |
| promptx | 0.025951 | 0.209840 | 0.116231 | 0.017992 |
| action_focus | 0.072331 | 0.295847 | 0.193573 | 0.014748 |

## Interpretation

- action_focus is best on motion-mask overlap over time and weighted spatial overlap.
- gf5base is best on RGB MSE and plain spatial IoU.
- promptx is weakest on this two-task single-view subset.

Artifacts:

- physicsiq_singleview_detail_0001_0005_3s.csv
- physicsiq_singleview_summary_0001_0005_3s.csv
- run_singleview_metrics.py
- run_singleview_metrics.log

## Local 0005 checked run

A local video was evaluated as task `0005` / `perspective-center` and saved as method `local_new_checked`.

| method | spatiotemporal_iou_mean | spatial_iou_mean | weighted_spatial_iou_mean | mse_mean |
|---|---:|---:|---:|---:|
| gf5base | 0.039560 | 0.461514 | 0.179555 | 0.016599 |
| promptx | 0.031716 | 0.283934 | 0.170686 | 0.018194 |
| action_focus | 0.020813 | 0.400549 | 0.256370 | 0.018293 |
| local_new_checked | 0.023478 | 0.393078 | 0.201570 | 0.010143 |

Caveat: for task `0005`, the original PhysicsIQ `0005` take-1 and `0203` take-2 center-view real videos are byte-identical in the currently downloaded dataset. Therefore real-take variance metrics for this task are not informative.

Reusable script:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate wmreward1
PYTHONPATH=/root/autodl-tmp/eval_repos/physics-IQ-benchmark-main \
python /root/physics-consistency-eval/WMReward/logs/physicsiq_singleview_0001_0005_3s/evaluate_singleview_video.py \
  --input-video /path/to/video.mp4 \
  --method local_name \
  --task-id 0005
```
