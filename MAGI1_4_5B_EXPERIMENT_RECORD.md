# WMReward MAGI-1 4.5B 轻量复现实验过程记录

## 1. 实验基本信息

| 项目 | 内容 |
| --- | --- |
| 实验名称 | WMReward + MAGI-1 4.5B 轻量复现 |
| 实验目标 | 使用 MAGI-1 4.5B 替代原项目 MAGI-1 24B，在单卡 A100 40G 环境下完成 I2V 视频生成，并用 VJEPA-2 计算 WMReward |
| 实验日期 | 2026-07-07 |
| 代码目录 | `/root/physics-consistency-eval/WMReward` |
| Conda 环境 | `wmreward1` |
| GPU | A100 40G |
| CUDA Driver | `[待填，例如 550.xx / CUDA Version 12.8]` |
| PyTorch | `[待填，例如 torch==2.4.0+cu124]` |
| 生成模型 | MAGI-1 `4.5B_base` |
| 生成模式 | I2V |
| WMReward 模型 | VJEPA-2 `vitg`，必要时用 `vitl` 做轻量验证 |
| 是否启用 WMReward guidance | 否，`--guidance_scale 0` |

## 2. 实验结论摘要

本次轻量复现采用 MAGI-1 4.5B 原生 vanilla pipeline 生成视频，再使用 VJEPA-2 计算 WMReward。与论文原始 MAGI-1 24B + guidance 设置相比，本实验牺牲了生成模型规模和在线 guidance，但显著降低了显存需求，适合在单张 A100 40G 上验证主流程。

当前单条 I2V 生成配置为 MAGI-1 4.5B 默认配置：

```text
video_size_h = 720
video_size_w = 720
num_frames = 96
num_steps = 64
```

根据实测进度，生成过程在 A100 40G 上约 `38 分 33 秒完成 InferBatch 5/5`。该速度比最初按 20% 进度估算的 `75-90 分钟` 更快，说明后续 chunk 明显快于第一段。主要耗时仍来自高分辨率、长帧数、64 步采样以及单卡 offload。

最终结果：

| 输出项 | 结果 |
| --- | --- |
| MAGI-1 4.5B 是否成功加载 | `[待填：是/否]` |
| I2V 视频是否成功生成 | 推理完成，但首次运行在 ffmpeg 保存 mp4 阶段失败 |
| 输出视频路径 | `./results/magi1_output.mp4` |
| 输出视频大小 | `[待填：例如 xx MB]` |
| 实际生成耗时 | `38 分 33 秒`，不含失败后的重跑 |
| 峰值显存 | `[待填：例如 xx GB]` |
| 平均 GPU 利用率 | `[待填：例如 60%-90%]` |
| WMReward 是否成功计算 | `[待填：是/否]` |
| VJEPA surprise score | `[待填：例如 0.xxxxxx]` |
| VJEPA similarity score | `[待填：例如 0.xxxxxx]` |

## 3. 环境检查与记录

### 3.1 查看 GPU、驱动和 CUDA

运行命令：

```bash
nvidia-smi
nvcc --version || true
```

这一步做了什么：

检查当前机器 GPU 型号、显存、NVIDIA driver 版本、系统 CUDA toolkit 版本。这里需要注意：`nvidia-smi` 里显示的 `CUDA Version` 表示驱动支持的最高 CUDA runtime 版本，不等价于 PyTorch 编译时使用的 CUDA 版本。

实际结果：

```text
[待填：粘贴 nvidia-smi 关键输出，例如 GPU 型号、driver version、CUDA Version、显存占用]
[待填：粘贴 nvcc --version 输出，如果没有 nvcc 则记录未安装]
```

### 3.2 查看 Python、PyTorch 和 CUDA 可用性

运行命令：

```bash
conda activate wmreward1

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

这一步做了什么：

确认当前 Python 环境中的 PyTorch 是否能正常调用 CUDA。之前实验中曾出现 `torch==2.12.1+cu130` 与当前驱动不兼容的问题，表现为：

```text
torch: 2.12.1+cu130
torch cuda: 13.0
cuda available: False
UserWarning: The NVIDIA driver on your system is too old
```

修复方式是安装与当前驱动兼容的 PyTorch CUDA 版本，例如 CUDA 12.4 版 PyTorch。

实际结果：

```text
[待填：粘贴修复后的 torch 版本、torch cuda、cuda available 输出]
```

### 3.3 检查 ffmpeg

运行命令：

```bash
which ffmpeg
ffmpeg -version | head -n 1
```

这一步做了什么：

MAGI-1 在处理 I2V 首帧和导出视频时会调用系统 `ffmpeg`。之前实验中曾出现：

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

如果缺失，需要安装：

```bash
apt-get update
apt-get install -y ffmpeg
```

或在 conda 环境中安装：

```bash
conda install -c conda-forge ffmpeg -y
```

实际结果：

```text
[待填：ffmpeg 路径和版本]
```

## 4. 代码和子模块准备

### 4.1 初始化子模块

运行命令：

```bash
cd /root/physics-consistency-eval/WMReward

git submodule update --init --recursive
git submodule sync --recursive
```

这一步做了什么：

拉取本项目依赖的 `MAGI-1` 和 `vjepa2` 子模块。`generate_magi1.py` 需要 `MAGI-1` 推理代码，`compute_wmreward.py` 和 `utils.py` 需要 VJEPA-2 相关代码。

实际结果：

```text
[待填：子模块是否成功初始化，例如 git submodule status 输出]
```

检查命令：

```bash
ls -ld MAGI-1 vjepa2
```

预期结果：

```text
MAGI-1 和 vjepa2 目录均存在且非空
```

### 4.2 记录当前代码版本

运行命令：

```bash
git rev-parse HEAD
git status --short
```

这一步做了什么：

记录实验时使用的代码版本，便于之后复现实验或定位差异。

实际结果：

```text
[待填：commit hash]
[待填：git status --short 输出，如有本地修改需记录]
```

## 5. Python 环境配置

### 5.1 创建并激活环境

运行命令：

```bash
conda env create -f environment.yml
conda activate wmreward
```

如果已经存在实验环境 `wmreward1`，则直接激活：

```bash
conda activate wmreward1
```

这一步做了什么：

安装 WMReward 基础依赖，包括 `diffusers`、`transformers`、`decord`、`opencv`、`imageio`、`ffmpeg-python`、`timm` 等。

实际结果：

```text
[待填：环境创建是否成功，当前使用的环境名]
```

### 5.2 安装兼容的 PyTorch、flash-attn 和 flashinfer

运行命令：

```bash
pip uninstall -y torch torchvision torchaudio flash-attn flashinfer-python

pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install flash-attn==2.4.2 --no-build-isolation

pip install flashinfer-python==0.2.0.post2 \
  --extra-index-url https://flashinfer.ai/whl/cu124/torch2.4/
```

这一步做了什么：

修复 PyTorch CUDA 版本与驱动不兼容的问题，并安装 MAGI-1 推理依赖的 `flash-attn` 和 `flashinfer`。之前不兼容环境中出现过：

```text
ImportError: flash_attn_2_cuda... undefined symbol: _ZN3c104cuda...
```

这通常表示 `flash-attn` 编译版本和当前 PyTorch ABI 不匹配，需要与 PyTorch 一起重装。

实际结果：

```text
[待填：pip install 是否成功]
[待填：torch/flash-attn/flashinfer 版本]
```

验证命令：

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
import flash_attn
print("flash_attn import: ok")
PY
```

预期结果：

```text
cuda available: True
flash_attn import: ok
```

## 6. 模型权重准备

### 6.1 下载 MAGI-1 4.5B 权重

运行命令：

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1

pip install "huggingface_hub[cli]"

python -u downloader/download_magi1_4_5b.py
```

这一步做了什么：

下载 MAGI-1 `4.5B_base`、共享 VAE、共享 T5 文本编码器，并整理成 MAGI-1 配置文件期望的目录结构：

```text
downloads/
├── 4.5B_base/
├── vae/
└── t5_pretrained/
```

实际结果：

```text
[待填：下载开始时间]
[待填：下载结束时间]
[待填：是否全部下载成功]
[待填：如有失败，记录失败文件和错误信息]
```

检查命令：

```bash
du -sh downloads/4.5B_base downloads/vae downloads/t5_pretrained
find downloads/4.5B_base -maxdepth 2 -type f | head
```

实际结果：

```text
[待填：三个目录大小]
[待填：权重文件列表摘要]
```

### 6.2 VJEPA-2 权重准备

运行命令：

```bash
ls -lh checkpoints || true
```

这一步做了什么：

检查是否已经存在本地 VJEPA-2 权重。若不存在，`compute_wmreward.py` 或 guidance 相关代码会通过 `torch.hub` 自动下载。当前单样本生成使用 `--guidance_scale 0`，生成阶段不会加载 VJEPA；只有计算 WMReward 时才会加载 VJEPA。

实际结果：

```text
[待填：是否已有 VJEPA 权重]
[待填：如果自动下载，记录下载耗时和缓存位置]
```

## 7. 单样本 I2V 视频生成

### 7.1 设置运行环境变量

运行命令：

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1

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
```

这一步做了什么：

MAGI-1 pipeline 内部会初始化 `torch.distributed`，即使单卡运行也需要 `MASTER_ADDR`、`MASTER_PORT`、`WORLD_SIZE`、`RANK` 等环境变量。之前未设置时出现过：

```text
ValueError: environment variable MASTER_ADDR expected, but not set
```

当前代码中的 `generate_magi1.py` 已经为这些变量设置默认值，但实验记录中仍显式 export，便于排查环境差异。

注意：如果出现如下 warning，一般可以忽略：

```text
Warning: expandable_segments not supported on this platform
```

### 7.2 确认 `--guidance_scale 0` 使用 vanilla pipeline

运行命令：

```bash
grep -n "pipeline_w_guidance\|guidance_scale > 0" generate_magi1.py
```

这一步做了什么：

确认只有在 `--guidance_scale > 0` 时才加载 `pipeline_w_guidance`。本实验使用 `--guidance_scale 0`，因此生成阶段不应下载或加载 VJEPA。

预期结果：

```text
代码中应存在 guidance_scale > 0 的条件判断
```

实际结果：

```text
[待填：grep 输出]
```

### 7.3 运行单条 I2V 生成

运行命令：

```bash
mkdir -p results logs

python generate_magi1.py \
  --config_file ./MAGI-1/example/4.5B/4.5B_base_config.json \
  --prompt "A ball falls from the table onto the floor" \
  --init_image ./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg \
  --output_path ./results/magi1_output.mp4 \
  --mode i2v \
  --guidance_scale 0 \
  2>&1 | tee logs/generate_magi1_4_5b_i2v.log
```

这一步做了什么：

使用 MAGI-1 4.5B 根据首帧图像和文本 prompt 生成一个物理场景视频。`--guidance_scale 0` 表示关闭 WMReward gradient guidance，只跑 MAGI-1 原始采样流程。

关键配置：

```text
模型配置：./MAGI-1/example/4.5B/4.5B_base_config.json
输出路径：./results/magi1_output.mp4
生成模式：i2v
guidance：关闭
```

运行过程中的关键日志：

```text
[待填：模型初始化成功日志]
[待填：InferBatch 进度日志，例如 InferBatch 0: 20%|...| 1/5]
[待填：保存视频成功日志]
```

当前已观察到的进度：

```text
InferBatch 0: 0%|...| 0/5
[2026-07-07 11:28:20,213 - INFO] transport_inputs len: 1
约 17 分钟完成 20%
InferBatch 0: 100%|...| 5/5 [38:33<00:00, 462.75s/it]
```

耗时记录：

| 阶段 | 耗时 |
| --- | --- |
| 启动到开始加载模型 | `[待填]` |
| T5/VAE/DiT 加载 | `[待填]` |
| 进度 0% -> 20% | `约 17 分钟` |
| 进度 20% -> 40% | `[待填]` |
| 进度 40% -> 60% | `[待填]` |
| 进度 60% -> 80% | `[待填]` |
| 进度 80% -> 100% | `[待填]` |
| 总生成耗时 | `38 分 33 秒完成 InferBatch 5/5` |

预期耗时：

```text
单条 720x720、96 帧、64 步 I2V：实测约 38 分 33 秒完成推理；首次保存 mp4 失败，需要修复 ffmpeg/输出目录后重跑
```

实际结果：

```text
首次运行未成功生成 ./results/magi1_output.mp4，失败在 save_video_to_disk 的 ffmpeg 编码/写文件阶段。
总推理耗时：38 分 33 秒。
报错摘要：ffmpeg._run.Error: ffmpeg error (see stderr output for detail)
```

### 7.4 监控 GPU 和进度

运行命令：

```bash
watch -n 1 nvidia-smi
```

这一步做了什么：

监控 GPU 显存和 GPU 利用率，判断程序是正常计算还是卡住。

判断标准：

| 现象 | 判断 |
| --- | --- |
| 显存占用高，GPU-Util 持续 60%-100% | 正常计算中 |
| 显存占用高，GPU-Util 长时间 0%-20% | 可能卡在 offload、IO 或同步 |
| 显存几乎不占用 | 模型可能没有成功加载到 GPU |

实际记录：

```text
[待填：峰值显存]
[待填：GPU-Util 范围]
[待填：CPU 内存占用，如有记录]
```

也可以查看输出文件：

```bash
ls -lh ./results/magi1_output.mp4
tail -n 80 logs/generate_magi1_4_5b_i2v.log
```

实际结果：

```text
[待填：输出文件大小]
[待填：日志末尾关键信息]
```

## 8. 单视频 WMReward 计算

### 8.1 使用 VJEPA-2 vitg 计算 reward

运行命令：

```bash
python compute_wmreward.py \
  --video_path ./results/magi1_output.mp4 \
  --model vitg \
  --window_size 16 \
  --context_frames 8 \
  --stride 8 \
  2>&1 | tee logs/compute_wmreward_magi1_output_vitg.log
```

这一步做了什么：

读取生成的视频，使用 VJEPA-2 `vitg` 计算滑动窗口 surprise/loss。loss 越低，表示 VJEPA 越容易预测视频未来 latent，通常代表物理动态更连贯。

关键参数：

```text
model = vitg
window_size = 16
context_frames = 8
stride = 8
```

预期耗时：

```text
首次运行：约 5-15 分钟，可能包含 VJEPA 权重下载
后续运行：约 1-5 分钟
```

实际结果：

```text
[待填：是否成功加载 VJEPA]
[待填：是否成功完成打分]
[待填：VJEPA Surprise Score]
[待填：VJEPA Similarity Score]
[待填：总耗时]
[待填：峰值显存]
```

### 8.2 如 vitg 显存或下载压力较大，使用 vitl 做验证

运行命令：

```bash
python compute_wmreward.py \
  --video_path ./results/magi1_output.mp4 \
  --model vitl \
  --window_size 16 \
  --context_frames 8 \
  --stride 8 \
  2>&1 | tee logs/compute_wmreward_magi1_output_vitl.log
```

这一步做了什么：

使用更小的 VJEPA-2 变体验证 WMReward 打分流程。该结果不完全等同于论文中的 `vitg` 设置，但适合作为单卡轻量 smoke test。

实际结果：

```text
[待填：vitl Surprise Score]
[待填：vitl Similarity Score]
[待填：总耗时]
```

## 9. 批量轻量复现

### 9.1 批量 vanilla 生成

运行命令：

```bash
bash generation/generate_i2v_magi1_multinode.sh \
  2>&1 | tee logs/generate_i2v_magi1_4_5b_vanilla_batch.log
```

这一步做了什么：

读取 `prompts/physics_iq.json` 中的 PhysicsIQ 样本，使用 MAGI-1 4.5B 逐条生成 I2V 视频。当前脚本默认单卡、`4.5B_base`、`vanilla` 采样，每条样本生成 1 个视频。

运行前检查：

```bash
ls -lh prompts/physics_iq.json
ls -ld PhysicsIQ/code/physics-IQ-benchmark || true
```

实际结果：

```text
[待填：PhysicsIQ 数据集是否已准备]
[待填：批量生成样本数]
[待填：成功样本数]
[待填：失败样本数]
[待填：输出目录]
[待填：总耗时]
```

按当前单条速度估算：

```text
1 条：约 75-90 分钟
10 条：约 12-15 小时
100 条：约 5-6 天
```

### 9.2 批量 rejection / Best-of-N 生成

运行命令：

```bash
SAMPLE_METHODS_OVERRIDE="rejection" REJECTION_SAMPLES=2 \
  bash generation/generate_i2v_magi1_multinode.sh \
  2>&1 | tee logs/generate_i2v_magi1_4_5b_rejection_n2.log
```

这一步做了什么：

每条样本生成多个候选视频，然后用 VJEPA loss 打分，保存 loss 最低的视频。该方法是轻量复现中比 gradient guidance 更稳妥的 WMReward 使用方式。

预期耗时：

```text
REJECTION_SAMPLES=2：约为 vanilla 的 2 倍，再加 VJEPA 打分时间
按当前高配置估算，每条样本约 2.5-3 小时
```

实际结果：

```text
[待填：rejection 样本数]
[待填：每条样本候选数]
[待填：平均每条耗时]
[待填：最佳候选 loss]
[待填：输出目录]
```

### 9.3 暂不推荐运行 guidance

命令如下，但 A100 40G 上不建议作为第一轮实验运行：

```bash
SAMPLE_METHODS_OVERRIDE="guidance" \
  bash generation/generate_i2v_magi1_multinode.sh
```

原因：

`guidance` 会额外加载 VJEPA，并在 denoising 过程中保留梯度和中间视频张量，显存和时间开销明显高于 vanilla/rejection。A100 40G 上存在 OOM 或极慢风险。

如确实运行，实际结果记录：

```text
[待填：是否 OOM]
[待填：峰值显存]
[待填：每条样本耗时]
[待填：guidance_scale]
[待填：guidance_frequency]
```

## 10. 轻量加速配置实验

如果默认 720x720、96 帧、64 步过慢，可以修改：

```text
MAGI-1/example/4.5B/4.5B_base_config.json
```

建议 smoke test 配置：

```json
{
  "num_frames": 48,
  "video_size_h": 480,
  "video_size_w": 480,
  "num_steps": 24
}
```

这一步做了什么：

降低帧数、分辨率和扩散步数，以显著减少显存和耗时。该配置适合验证流程，不适合直接与论文结果比较。

实验记录：

| 配置 | num_frames | height | width | num_steps | 单条耗时 | 峰值显存 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 默认配置 | 96 | 720 | 720 | 64 | `[待填]` | `[待填]` | 当前主实验 |
| smoke test | 48 | 480 | 480 | 24 | `[待填]` | `[待填]` | 流程验证 |
| 自定义 1 | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` | `[待填]` |

## 11. 已遇到问题与解决记录

### 11.1 PyTorch CUDA 版本与驱动不兼容

现象：

```text
torch: 2.12.1+cu130
torch cuda: 13.0
cuda available: False
The NVIDIA driver on your system is too old
```

原因：

当前驱动不支持 PyTorch 编译所需的 CUDA 13.0 runtime。

解决：

安装 CUDA 12.4 版 PyTorch：

```bash
pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124
```

结果：

```text
[待填：修复后 cuda available 是否为 True]
```

### 11.2 flash-attn undefined symbol

现象：

```text
ImportError: flash_attn_2_cuda... undefined symbol: _ZN3c104cuda...
```

原因：

`flash-attn` 与当前 PyTorch ABI 不匹配。

解决：

在重装 PyTorch 后重新安装：

```bash
pip install flash-attn==2.4.2 --no-build-isolation
```

结果：

```text
[待填：flash_attn 是否可以 import]
```

### 11.3 torch.distributed 缺少 MASTER_ADDR

现象：

```text
ValueError: environment variable MASTER_ADDR expected, but not set
```

原因：

MAGI-1 单卡运行时仍会初始化分布式进程组。

解决：

显式设置单卡分布式环境变量，或使用已修改的 `generate_magi1.py` 默认值。

结果：

```text
[待填：是否解决]
```

### 11.4 缺少 ffmpeg

现象：

```text
FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'
```

原因：

系统没有安装 `ffmpeg` 可执行文件。`ffmpeg-python` 只是 Python wrapper，不包含系统二进制。

解决：

```bash
apt-get install -y ffmpeg
```

或：

```bash
conda install -c conda-forge ffmpeg -y
```

结果：

```text
[待填：ffmpeg 是否可用]
```

### 11.5 `--guidance_scale 0` 仍加载 VJEPA

现象：

```text
Downloading ... vjepa2
Downloading "http://localhost:8300/vitg.pt"
pipeline_w_guidance.py
```

原因：

旧版 `generate_magi1.py` 即使 `--guidance_scale 0` 也导入 `pipeline_w_guidance`，导致生成阶段仍尝试加载 VJEPA。

解决：

修改 `generate_magi1.py`，只有 `args.guidance_scale > 0` 时才导入：

```python
from inference.pipeline.pipeline_w_guidance import MagiPipeline
```

否则导入 vanilla pipeline：

```python
from inference.pipeline.pipeline import MagiPipeline
```

结果：

```text
[待填：运行时是否不再下载 VJEPA]
```

### 11.6 推理完成后 ffmpeg 保存 mp4 失败

现象：

```text
InferBatch 0: 100%|...| 5/5 [38:33<00:00, 462.75s/it]
ffmpeg._run.Error: ffmpeg error (see stderr output for detail)
```

原因：

MAGI-1 推理已经完成，但 `save_video_to_disk` 调用 ffmpeg 将 raw RGB 帧编码为 mp4 时失败。常见原因包括：输出目录 `./results` 不存在、ffmpeg 无法写入该目录、ffmpeg 缺少 mp4/h264 编码支持，或输出路径权限异常。

解决：

运行前显式创建输出和日志目录，并测试 ffmpeg：

```bash
mkdir -p results logs

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i color=c=black:s=16x16:d=0.04:r=24 \
  -frames:v 1 results/ffmpeg_smoke.mp4

ls -lh results/ffmpeg_smoke.mp4
rm -f results/ffmpeg_smoke.mp4
```

同时已在 `generate_magi1.py` 增加运行前预检：自动创建输出目录，并先写入一个 1 帧测试 mp4。如果 ffmpeg 仍不可用，会在模型加载和 38 分钟推理之前直接报出具体 stderr。

结果：

```text
[待填：ffmpeg smoke test 是否成功]
[待填：修复后是否成功保存 magi1_output.mp4]
```

## 12. 最终产物

| 文件/目录 | 内容 | 状态 |
| --- | --- | --- |
| `downloads/4.5B_base/` | MAGI-1 4.5B DiT 权重 | `[待填]` |
| `downloads/vae/` | MAGI-1 VAE 权重 | `[待填]` |
| `downloads/t5_pretrained/` | MAGI-1 T5 权重 | `[待填]` |
| `results/magi1_output.mp4` | 单样本 I2V 输出视频 | `[待填]` |
| `logs/generate_magi1_4_5b_i2v.log` | 单样本生成日志 | `[待填]` |
| `logs/compute_wmreward_magi1_output_vitg.log` | WMReward 打分日志 | `[待填]` |
| `generated_videos/` | 批量生成结果 | `[待填，如未运行则填未运行]` |

## 13. 后续计划

1. 完成当前单条 `magi1_output.mp4` 生成，并记录总耗时、峰值显存和 GPU 利用率。
2. 使用 `compute_wmreward.py --model vitg` 计算 VJEPA surprise score。
3. 若默认配置耗时过长，使用 `480x480 + 48 frames + 24 steps` 做一组 smoke test，对比耗时和显存。
4. 小规模运行 batch vanilla，例如先跑 1-3 条 PhysicsIQ 样本，确认数据路径和输出组织。
5. 在 vanilla 稳定后，再尝试 `REJECTION_SAMPLES=2` 的 Best-of-N 轻量复现。

## 14. 可直接复用的命令清单

### 环境检查

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
PY
which ffmpeg
```

### 单样本生成

```bash
cd /root/physics-consistency-eval/WMReward
conda activate wmreward1
mkdir -p results logs

python generate_magi1.py \
  --config_file ./MAGI-1/example/4.5B/4.5B_base_config.json \
  --prompt "A ball falls from the table onto the floor" \
  --init_image ./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg \
  --output_path ./results/magi1_output.mp4 \
  --mode i2v \
  --guidance_scale 0 \
  2>&1 | tee logs/generate_magi1_4_5b_i2v.log
```

### 查看进度

```bash
watch -n 1 nvidia-smi
tail -f logs/generate_magi1_4_5b_i2v.log
ls -lh results/
```

### 计算 WMReward

```bash
python compute_wmreward.py \
  --video_path ./results/magi1_output.mp4 \
  --model vitg \
  --window_size 16 \
  --context_frames 8 \
  --stride 8 \
  2>&1 | tee logs/compute_wmreward_magi1_output_vitg.log
```

### 小规模 batch vanilla

```bash
bash generation/generate_i2v_magi1_multinode.sh \
  2>&1 | tee logs/generate_i2v_magi1_4_5b_vanilla_batch.log
```

### Best-of-N / rejection

```bash
SAMPLE_METHODS_OVERRIDE="rejection" REJECTION_SAMPLES=2 \
  bash generation/generate_i2v_magi1_multinode.sh \
  2>&1 | tee logs/generate_i2v_magi1_4_5b_rejection_n2.log
```
