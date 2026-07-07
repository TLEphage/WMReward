# WMReward MAGI-1 4.5B 轻量复现实验记录

## 1. 实验设置

| 项目 | 内容 |
| --- | --- |
| 实验目标 | 使用 MAGI-1 4.5B 替代原项目 MAGI-1 24B，完成单卡轻量 I2V 生成，并用 VJEPA-2 计算 WMReward |
| 实验日期 | 2026-07-07 |
| 代码目录 | `/root/physics-consistency-eval/WMReward` |
| Conda 环境 | `wmreward1` |
| GPU | A100 40G |
| 生成模型 | MAGI-1 `4.5B_base` |
| 生成模式 | I2V |
| Guidance | 关闭，`--guidance_scale 0` |
| WMReward 模型 | VJEPA-2 `vitg` |

本实验采用 MAGI-1 4.5B vanilla pipeline 生成视频，不在 denoising 过程中接入 WMReward guidance。视频生成完成后，再单独使用 `compute_wmreward.py` 对输出视频计算 VJEPA surprise score。

当前 MAGI-1 4.5B 默认生成配置：

```text
video_size_h = 720
video_size_w = 720
num_frames = 96
num_steps = 64
fps = 24
```

## 2. 环境配置

### 2.1 创建 Python 环境

运行命令：

```bash
cd /root/physics-consistency-eval/WMReward

conda env create -f environment.yml
conda activate wmreward
```

若已创建实验环境：

```bash
conda activate wmreward1
```

记录：

| 指标 | 数值 |
| --- | --- |
| Python 版本 | `[待填]` |
| Conda 环境名 | `wmreward1` |

### 2.2 安装推理依赖

运行命令：

```bash
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install flash-attn==2.4.2 --no-build-isolation

pip install flashinfer-python==0.2.0.post2 \
  --extra-index-url https://flashinfer.ai/whl/cu124/torch2.4/
```

记录：

| 指标 | 数值 |
| --- | --- |
| PyTorch 版本 | `[待填]` |
| PyTorch CUDA 版本 | `[待填]` |
| NVIDIA Driver | `[待填]` |
| `nvidia-smi` CUDA Version | `[待填]` |

检查命令：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

nvidia-smi
```

### 2.3 安装系统视频工具

运行命令：

```bash
apt-get update
apt-get install -y ffmpeg
```

或：

```bash
conda install -c conda-forge ffmpeg -y
```

检查命令：

```bash
which ffmpeg
ffmpeg -version | head -n 1
```

记录：

| 指标 | 数值 |
| --- | --- |
| ffmpeg 路径 | `[待填]` |
| ffmpeg 版本 | `[待填]` |

## 3. 代码和模型准备

### 3.1 初始化子模块

运行命令：

```bash
cd /root/physics-consistency-eval/WMReward
git submodule update --init --recursive
git submodule sync --recursive
```

记录：

| 指标 | 数值 |
| --- | --- |
| 代码 commit | `[待填：git rev-parse HEAD]` |
| MAGI-1 子模块 commit | `[待填]` |
| vjepa2 子模块 commit | `[待填]` |

查看命令：

```bash
git rev-parse HEAD
git submodule status
```

### 3.2 下载 MAGI-1 4.5B 权重

运行命令：

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1

pip install "huggingface_hub[cli]"
python -u downloader/download_magi1_4_5b.py
```

该命令下载并整理以下目录：

```text
downloads/
├── 4.5B_base/
├── vae/
└── t5_pretrained/
```

记录：

| 指标 | 数值 |
| --- | --- |
| 下载总耗时 | `[待填]` |
| `downloads/4.5B_base` 大小 | `[待填]` |
| `downloads/vae` 大小 | `[待填]` |
| `downloads/t5_pretrained` 大小 | `[待填]` |

查看命令：

```bash
du -sh downloads/4.5B_base downloads/vae downloads/t5_pretrained
```

### 3.3 下载 VJEPA-2 vitg 权重

推荐使用 `aria2c` 多连接下载：

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1

mkdir -p checkpoints

apt-get update
apt-get install -y aria2

aria2c \
  -c \
  -x 16 \
  -s 16 \
  -k 4M \
  --max-tries=0 \
  --retry-wait=10 \
  --timeout=60 \
  --connect-timeout=30 \
  -d checkpoints \
  -o vitg.pt \
  https://dl.fbaipublicfiles.com/vjepa2/vitg.pt
```

如果不安装 `aria2c`，也可以使用仓库脚本下载：

```bash
python -u downloader/download_vjepa2.py --model vitg
```

如使用旧版 `compute_wmreward.py`，可将下载好的 checkpoint 软链接到 torch hub 缓存目录：

```bash
mkdir -p /root/.cache/torch/hub/checkpoints
ln -sf /root/physics-consistency-eval/WMReward/checkpoints/vitg.pt \
  /root/.cache/torch/hub/checkpoints/vitg.pt
```

记录：

| 指标 | 数值 |
| --- | --- |
| 下载命令 | `aria2c -c -x 16 -s 16 -k 4M ... https://dl.fbaipublicfiles.com/vjepa2/vitg.pt` |
| VJEPA checkpoint | `checkpoints/vitg.pt` |
| checkpoint 大小 | `约 15.3 GB` |
| 下载总耗时 | `[待填]` |

查看命令：

```bash
ls -lh checkpoints/vitg.pt
```

## 4. 单样本 MAGI-1 4.5B I2V 生成

### 4.1 运行命令

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1

mkdir -p results logs

export MASTER_ADDR=localhost
export MASTER_PORT=6009
export GPUS_PER_NODE=1
export NNODES=1
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0
export PAD_HQ=1
export PAD_DURATION=1
export OFFLOAD_T5_CACHE=true
export OFFLOAD_VAE_CACHE=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python generate_magi1.py \
  --config_file ./MAGI-1/example/4.5B/4.5B_base_config.json \
  --prompt "A ball falls from the table onto the floor" \
  --init_image ./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg \
  --output_path ./results/magi1_output.mp4 \
  --mode i2v \
  --guidance_scale 0 \
  2>&1 | tee logs/generate_magi1_4_5b_i2v.log
```

该命令执行内容：

1. 加载 MAGI-1 4.5B、T5、VAE。
2. 读取输入首帧图像和文本 prompt。
3. 生成 `720x720`、`96` 帧、`24 fps` 视频。
4. 将结果保存为 `./results/magi1_output.mp4`。

### 4.2 运行记录

| 指标 | 数值 |
| --- | --- |
| 输入图像 | `./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg` |
| Prompt | `A ball falls from the table onto the floor` |
| 输出视频 | `./results/magi1_output.mp4` |
| 推理进度日志 | `InferBatch 0: 100% \| 5/5 [38:33<00:00, 462.75s/it]` |
| 单命令总运行时间 | `[待填：记录完整 wall time；当前已观测推理阶段 38 分 33 秒]` |
| 峰值显存 | `[待填：nvidia-smi 记录，例如 xx GB]` |
| 平均 GPU 利用率 | `[待填：例如 60%-90%]` |
| 输出视频大小 | `[待填：ls -lh results/magi1_output.mp4]` |

查看输出：

```bash
ls -lh ./results/magi1_output.mp4
ffprobe ./results/magi1_output.mp4
```

监控命令：

```bash
watch -n 1 nvidia-smi
tail -f logs/generate_magi1_4_5b_i2v.log
```

## 5. 单视频 WMReward 计算

### 5.1 检查 VJEPA-2 vitg 权重

运行命令：

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1

ls -lh checkpoints/vitg.pt
```

VJEPA-2 `vitg` 权重路径：

```text
checkpoints/vitg.pt
```

记录：

| 指标 | 数值 |
| --- | --- |
| VJEPA checkpoint | `checkpoints/vitg.pt` |
| checkpoint 大小 | `约 15.3 GB` |

### 5.2 运行命令

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1

python compute_wmreward.py \
  --video_path ./results/magi1_output.mp4 \
  --model vitg \
  --window_size 16 \
  --context_frames 8 \
  --stride 8 \
  2>&1 | tee logs/compute_wmreward_magi1_output_vitg.log
```

该命令执行内容：

1. 读取 `./results/magi1_output.mp4`。
2. 加载本地 VJEPA-2 `vitg` checkpoint：`checkpoints/vitg.pt`。
3. 以 16 帧窗口滑动计算 VJEPA surprise/loss。
4. 输出 `VJEPA Surprise Score` 和 `VJEPA Similarity Score`。

### 5.3 运行记录

| 指标 | 数值 |
| --- | --- |
| 输入视频 | `./results/magi1_output.mp4` |
| VJEPA 模型 | `vitg` |
| window size | `16` |
| context frames | `8` |
| stride | `8` |
| 单命令总运行时间 | `[待填]` |
| 峰值显存 | `[待填]` |
| VJEPA Surprise Score | `[待填]` |
| VJEPA Similarity Score | `[待填]` |

## 6. 批量 vanilla 复现

### 6.1 运行命令

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1

bash generation/generate_i2v_magi1_multinode.sh \
  2>&1 | tee logs/generate_i2v_magi1_4_5b_vanilla_batch.log
```

该命令执行内容：

1. 读取 `prompts/physics_iq.json`。
2. 对每条 PhysicsIQ 样本生成 1 个 MAGI-1 4.5B vanilla 视频。
3. 输出到 `generated_videos/<group>/MAGI-1-4.5B_base/`。

### 6.2 运行记录

| 指标 | 数值 |
| --- | --- |
| 样本数量 | `[待填]` |
| 每条候选视频数 | `1` |
| 总运行时间 | `[待填]` |
| 平均单样本时间 | `[待填]` |
| 峰值显存 | `[待填]` |
| 输出目录大小 | `[待填]` |

## 7. Best-of-N / rejection 复现

### 7.1 运行命令

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1

SAMPLE_METHODS_OVERRIDE="rejection" REJECTION_SAMPLES=2 \
  bash generation/generate_i2v_magi1_multinode.sh \
  2>&1 | tee logs/generate_i2v_magi1_4_5b_rejection_n2.log
```

该命令执行内容：

1. 每条样本生成 `REJECTION_SAMPLES` 个候选视频。
2. 对每个候选视频计算 VJEPA loss。
3. 保存 VJEPA loss 最低的候选作为最终结果。

如需对齐论文中的 Best-of-16，可使用：

```bash
SAMPLE_METHODS_OVERRIDE="rejection" REJECTION_SAMPLES=16 \
  bash generation/generate_i2v_magi1_multinode.sh \
  2>&1 | tee logs/generate_i2v_magi1_4_5b_rejection_n16.log
```

### 7.2 运行记录

| 指标 | N=2 | N=16 |
| --- | --- | --- |
| 样本数量 | `[待填]` | `[待填]` |
| 每条候选视频数 | `2` | `16` |
| 总运行时间 | `[待填]` | `[待填]` |
| 平均单样本时间 | `[待填]` | `[待填]` |
| 峰值显存 | `[待填]` | `[待填]` |
| 平均最佳 VJEPA loss | `[待填]` | `[待填]` |

按当前单样本推理速度估算，`REJECTION_SAMPLES=16` 的单样本生成时间约为 `38.5 分钟 x 16 = 10.3 小时`，另需叠加 VJEPA 打分时间。

## 8. 轻量化配置对比

若默认配置耗时过长，可降低帧数、分辨率和采样步数进行 smoke test。建议修改：

```text
MAGI-1/example/4.5B/4.5B_base_config.json
```

轻量验证配置：

```json
{
  "num_frames": 48,
  "video_size_h": 480,
  "video_size_w": 480,
  "num_steps": 24
}
```

配置对比记录：

| 配置 | 分辨率 | 帧数 | 步数 | 单样本总时间 | 峰值显存 | WMReward Score |
| --- | --- | --- | --- | --- | --- | --- |
| 默认配置 | `720x720` | `96` | `64` | `[待填，当前推理阶段 38 分 33 秒]` | `[待填]` | `[待填]` |
| 轻量配置 | `480x480` | `48` | `24` | `[待填]` | `[待填]` | `[待填]` |

## 9. 实验产物

| 产物 | 路径 | 关键记录 |
| --- | --- | --- |
| 单样本生成日志 | `logs/generate_magi1_4_5b_i2v.log` | 总时间 `[待填]` |
| 单样本输出视频 | `results/magi1_output.mp4` | 大小 `[待填]` |
| WMReward 日志 | `logs/compute_wmreward_magi1_output_vitg.log` | Surprise `[待填]` |
| 批量 vanilla 日志 | `logs/generate_i2v_magi1_4_5b_vanilla_batch.log` | 总时间 `[待填]` |
| rejection N=2 日志 | `logs/generate_i2v_magi1_4_5b_rejection_n2.log` | 总时间 `[待填]` |
| rejection N=16 日志 | `logs/generate_i2v_magi1_4_5b_rejection_n16.log` | 总时间 `[待填]` |

## 10. 关键结论

1. 当前单样本 MAGI-1 4.5B vanilla I2V 推理阶段已观测耗时为 `38 分 33 秒`。
2. 当前单样本命令只生成 `1` 个候选视频；只有 `REJECTION_SAMPLES=16` 时才会为每条样本生成 16 个候选视频。
3. 单卡 A100 40G 更适合先跑 vanilla 和小 N rejection；`N=16` 的单样本耗时约为 10 小时量级。
4. `--guidance_scale 0` 表示关闭 gradient guidance，生成阶段不会使用 VJEPA；VJEPA 只在后续 WMReward 打分或 rejection 选择候选时使用。
