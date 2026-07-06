# WMReward 轻量复现与 Wan2.2 替换流程

## 1. 整体流程概述

WMReward 的目标不是重新训练视频生成模型，而是在推理阶段用一个预训练 latent world model 作为物理合理性奖励，搜索或引导视频生成模型产出更符合物理规律的视频。

本仓库中这个 world model 是 VJEPA-2。它的核心判断方式是：给定生成视频的若干上下文帧，让 VJEPA-2 预测未来帧的 latent 表示，再把预测 latent 和真实生成出来的未来帧 latent 做相似度比较。如果未来帧很好预测，说明运动和物理演化更连贯；如果 surprise/loss 高，说明生成视频更可能存在物理不合理之处。

完整流程可以拆成四层：

1. 准备环境和模型权重。
2. 用视频生成模型生成候选视频。论文原始设置使用 MAGI-1 24B；当前轻量化复现默认改为 MAGI-1 4.5B。
3. 用 VJEPA-2 计算 WMReward，也就是滑动窗口上的 surprise/loss。
4. 用 WMReward 改善生成结果。轻量复现优先使用 Best-of-N/rejection：生成多个候选，选择 VJEPA loss 最低的一个；完整复现还可以把 VJEPA 梯度接入扩散采样过程做 guidance。

替换为 Wan2.2 后，推荐先做黑盒式 Best-of-N 复现：Wan2.2 负责生成 N 个候选视频，现有 VJEPA reward 代码负责评分和选择。这个方案不需要修改 Wan2.2 的 denoising loop，改动量最小，也和论文中对 Wan2.2 的实验路径一致。

## 2. MAGI-1 4.5B 轻量流程

### 步骤 1：初始化子模块

```bash
git submodule update --init --recursive
git submodule sync --recursive
```

这一步做了什么：

拉取仓库依赖的两个子模块：`vjepa2` 和 `MAGI-1`。`compute_wmreward.py` 和 `utils.py` 中的 VJEPA 结构依赖 `vjepa2`；`generate_magi1.py` 和 `generator_i2v_multinode.py` 依赖 `MAGI-1` 的推理 pipeline。

### 步骤 2：创建 Python 环境

```bash
conda env create -f environment.yml
conda activate wmreward

pip install torch==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu124

pip install flash-attn==2.4.2 --no-build-isolation
pip install flashinfer-python==0.2.0.post2 \
  --extra-index-url https://flashinfer.ai/whl/cu124/torch2.4/
```

这一步做了什么：

安装 WMReward、VJEPA-2、MAGI-1 推理所需的基础依赖。`torch/torchvision` 用于模型推理和张量处理；`decord/opencv/Pillow` 用于视频和图像读取；`diffusers/transformers` 用于扩散模型相关组件；`flash-attn` 和 `flashinfer` 用于加速大模型推理。

### 步骤 3：下载 MAGI-1 4.5B 权重

```bash
pip install "huggingface_hub[cli]"

python -u downloader/download_magi1_4_5b.py
```

这一步做了什么：

下载 MAGI-1 4.5B 的视频生成 DiT 权重、共享 VAE、共享 T5 文本编码器，并整理成官方 4.5B 配置文件期望的目录结构：

```text
WMReward/
└── downloads/
    ├── 4.5B_base/
    ├── vae/
    └── t5_pretrained/
```

当前下载脚本只在 `../PhyT2V/downloader/download_cogvideox5b.py` 的基础上做必要调整，默认只下载 `4.5B_base`。如果后续想下载 `4.5B_distill` 或 `4.5B_distill_quant`，按同一模板复制脚本并把 `INCLUDE_PREFIXES` 中的 `ckpt/magi/4.5B_base/` 改成对应目录即可。

MAGI-1 只在视频生成阶段需要；如果只计算已有视频的 WMReward，则不需要下载 MAGI-1。

### 步骤 4：单条 prompt 生成 I2V 视频

```bash
python generate_magi1.py \
  --config_file ./MAGI-1/example/4.5B/4.5B_base_config.json \
  --prompt "A ball falls from the table onto the floor" \
  --init_image ./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg \
  --output_path ./results/magi1_output.mp4 \
  --mode i2v
```

这一步做了什么：

调用 MAGI-1 4.5B 的 image-to-video pipeline，用文本 prompt 和首帧图像生成视频。`generate_magi1.py` 会把 `MAGI-1` 子模块加入 `sys.path`，然后创建 `MagiPipeline`，最后根据 `--mode` 运行 T2V、I2V 或 V2V。

当前代码已经把 `--config_file` 改成可选参数；如果不传，它默认使用：

```text
./MAGI-1/example/4.5B/4.5B_base_config.json
```

如果已按同样目录结构准备好其他权重，也可以通过 `--magi_model_variant 4.5B_distill` 或环境变量 `MAGI1_MODEL_VARIANT=4.5B_distill` 切换到其他 4.5B 变体。

### 步骤 5：计算单个视频的 WMReward/VJEPA surprise

```bash
python compute_wmreward.py \
  --video_path ./results/magi1_output.mp4 \
  --model vitg \
  --window_size 16 \
  --context_frames 8 \
  --stride 8
```

这一步做了什么：

加载 VJEPA-2 模型，把输入视频转成 `[1, C, T, H, W]` 张量，然后用 `compute_vjepa_loss_sliding_window` 计算 surprise/loss。默认逻辑是用 16 帧窗口滑动，前 8 帧作为上下文，后续帧作为预测目标。输出的 surprise/loss 越低，表示 VJEPA 越容易预测这个视频的未来 latent，通常代表物理动态更连贯。

### 步骤 6：批量生成 PhysicsIQ 视频

```bash
bash generation/generate_i2v_magi1_multinode.sh
```

这一步做了什么：

读取 `prompts/physics_iq.json` 中的 PhysicsIQ 条目，将任务按 GPU 分片，调用 `generator_i2v_multinode.py` 批量生成视频。这个脚本支持三种采样方式：

- `vanilla`：每个样本只生成一个视频。
- `rejection`：每个样本生成多个候选视频，用 VJEPA loss 打分，选 loss 最低的候选。
- `guidance`：调用 MAGI-1 子模块中带 VJEPA guidance 的 pipeline，在 denoising 过程中直接使用 VJEPA 梯度。

当前 shell 脚本默认使用 `guidance`，默认单卡运行 `4.5B_base`，并把输出放到 `generated_videos/<group>/MAGI-1-4.5B_base/` 下。论文原始 MAGI-1 结果使用 24B，因此 4.5B 版本属于轻量化复现，显存更友好但结果不能直接等同论文 24B 数字。

## 3. 替换为 Wan2.2 后的推荐轻量流程

### 步骤 1：下载 Wan2.2 A14B Diffusers 权重

```bash
pip install "huggingface_hub[cli]"

python -u downloader/download_wan2_2_a14b_diffusers.py \
  --variant i2v
```

这一步做了什么：

下载 Wan2.2 A14B 的 Diffusers 版权重到：

```text
../../models/Wan2.2-I2V-A14B-Diffusers
```

PhysicsIQ 是图像或视频条件生成任务，因此优先使用 `Wan2.2-I2V-A14B-Diffusers`。如果只做 text-to-video，可以改成：

```bash
python -u downloader/download_wan2_2_a14b_diffusers.py \
  --variant t2v
```

如果两个都要下载：

```bash
python -u downloader/download_wan2_2_a14b_diffusers.py \
  --variant both
```

### 步骤 2：升级 Wan2.2 推理依赖

```bash
conda activate wmreward

pip install -U diffusers transformers accelerate safetensors imageio-ffmpeg
```

如果当前发布版 `diffusers` 里没有 Wan2.2 pipeline，可以使用源码版：

```bash
pip install -U git+https://github.com/huggingface/diffusers
```

这一步做了什么：

原始 `environment.yml` 中的 `diffusers==0.29.2` 面向 MAGI-1/原始仓库依赖，通常不足以直接加载 Wan2.2 Diffusers 权重。Wan2.2 需要新版 Diffusers，其中包含 Wan 相关 pipeline、transformer、VAE 和 scheduler 定义。

### 步骤 3：新增 Wan2.2 单视频生成入口

```bash
python generate_wan2_2.py \
  --model_path ../../models/Wan2.2-I2V-A14B-Diffusers \
  --prompt "A ball falls from the table onto the floor" \
  --init_image ./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg \
  --output_path ./results/wan2_2_output.mp4 \
  --height 480 \
  --width 720 \
  --num_frames 49 \
  --fps 8 \
  --num_inference_steps 50 \
  --guidance_scale 5.0
```

这一步做了什么：

这是替换 MAGI-1 后需要新增的脚本入口。它应当加载 `../../models/Wan2.2-I2V-A14B-Diffusers`，接收 prompt 和首帧图像，调用 Diffusers 的 Wan image-to-video pipeline，最后导出 mp4。实现时可以通过 `DiffusionPipeline.from_pretrained(..., torch_dtype=torch.bfloat16)` 或 Wan 专用 pipeline 加载模型，然后把 `image` 和 `prompt` 传给 pipeline。

轻量复现中，这个脚本只需要负责生成视频，不需要实现 WMReward guidance。

### 步骤 4：用现有脚本计算 Wan2.2 输出的 WMReward

```bash
python compute_wmreward.py \
  --video_path ./results/wan2_2_output.mp4 \
  --model vitg \
  --window_size 16 \
  --context_frames 8 \
  --stride 8
```

这一步做了什么：

这一部分完全不依赖 MAGI-1，因此无需修改。Wan2.2 生成的视频同样可以被读成视频帧，再输入 VJEPA-2 计算 surprise/loss。也就是说，WMReward 的评估部分可以直接复用。

### 步骤 5：用 Wan2.2 做 Best-of-N/rejection 轻量复现

```bash
python generator_i2v_wan2_2_multinode.py \
  --output_folder ./generated_videos/physics_iq/Wan2.2 \
  --batch_json ./prompts/physics_iq.json \
  --base_dir ./physicsiq_benchmark/code \
  --num_gpus 1 \
  --gpu_idx 0 \
  --sampling_method rejection \
  --rejection_samples 16 \
  --wan_model_path ../../models/Wan2.2-I2V-A14B-Diffusers \
  --num_inference_steps 50 \
  --num_frames 49 \
  --height 480 \
  --width 720 \
  --cfg_scale 5.0 \
  --vjepa_variant vit_giant \
  --vjepa_img_size 256 \
  --vjepa_masking_mode causal \
  --vjepa_context_frames 8 \
  --slice_window_size 16 \
  --slice_stride 8 \
  --loss_mode mean \
  --seed 42
```

这一步做了什么：

这是推荐的 Wan2.2 轻量复现主路径。每条 PhysicsIQ 样本生成 16 个候选视频，每个候选视频都用 VJEPA-2 打分，最后保存 loss 最低的候选。这里的 `generator_i2v_wan2_2_multinode.py` 可以从当前 `generator_i2v_multinode.py` 改造而来，保留批处理、分片、VJEPA scoring 和输出组织逻辑，只替换视频生成 backend。

## 4. 替换为 Wan2.2 后流程有什么不同

### 4.1 模型下载路径不同

论文中 MAGI-1 原始 24B 流程使用：

```text
WMReward/downloads/24B_base
WMReward/downloads/vae
WMReward/downloads/t5_pretrained
```

当前轻量 MAGI-1 4.5B 流程使用：

```text
WMReward/downloads/4.5B_base
WMReward/downloads/vae
WMReward/downloads/t5_pretrained
```

Wan2.2 轻量流程使用：

```text
../../models/Wan2.2-I2V-A14B-Diffusers
../../models/Wan2.2-T2V-A14B-Diffusers
```

因此当前仓库有两套轻量下载器：

```text
downloader/download_magi1_4_5b.py
downloader/download_wan2_2_a14b_diffusers.py
```

### 4.2 视频生成入口不同

MAGI-1 4.5B 入口：

```bash
python generate_magi1.py \
  --config_file ./MAGI-1/example/4.5B/4.5B_base_config.json \
  --prompt "..." \
  --init_image image.jpg \
  --output_path output.mp4 \
  --mode i2v
```

Wan2.2 目标入口：

```bash
python generate_wan2_2.py \
  --model_path ../../models/Wan2.2-I2V-A14B-Diffusers \
  --prompt "..." \
  --init_image image.jpg \
  --output_path output.mp4
```

MAGI-1 通过 `MagiPipeline(args.config_file)` 加载配置和权重；Wan2.2 通过 Diffusers 的 `from_pretrained(model_path)` 直接加载本地模型目录。当前 `generate_magi1.py` 和 `generator_i2v_multinode.py` 已经默认使用 `4.5B_base`，可以用 `--magi_model_variant` 切换到 `4.5B_distill`、`4.5B_distill_quant` 或 `24B_base`。

### 4.3 Guidance 能力不同

MAGI-1 原始 `guidance` 路径依赖子模块中的专用实现：

```python
from inference.pipeline.pipeline_w_guidance import MagiPipeline
```

Wan2.2 Diffusers pipeline 默认没有接入 WMReward guidance。轻量复现时应先把 `guidance` 改成不可用或退化为 `rejection`。也就是说：

```bash
--sampling_method rejection
```

是 Wan2.2 替换后的推荐默认值。

如果要完整复现论文中的 `gradient guidance`，需要修改 Wan2.2 的 denoising loop：在每个 guidance step 解码或近似得到当前 `x0` 视频，计算 VJEPA loss，对当前 latent 求梯度，再把这个梯度和 CFG 的更新项合并。这比黑盒 BoN 复杂得多，也会显著增加显存和时间开销。

### 4.4 PhysicsIQ 条件输入不同点

当前 `generator_i2v_multinode.py` 会读取 `prompts/physics_iq.json`，每条数据包含：

```json
{
  "input_video": "...",
  "input_image": "...",
  "prompt": "...",
  "output_video": "..."
}
```

MAGI-1 现在用 `load_first_frame(input_image_abs, input_video_abs)` 取首帧作为 I2V 条件。Wan2.2 I2V 可以继续沿用这个输入方式：把首帧 PIL image 传给 Diffusers pipeline 的 `image` 参数。

如果使用 `Wan2.2-T2V-A14B-Diffusers`，则无法利用 PhysicsIQ 的首帧条件，结果不再是同一个 I2V 任务。因此 PhysicsIQ 轻量复现应使用 I2V 版模型。

## 5. 需要修改哪些代码

### 5.1 新增下载器

已新增：

```text
downloader/download_magi1_4_5b.py
downloader/download_wan2_2_a14b_diffusers.py
downloader/command.sh
```

作用：

- `download_magi1_4_5b.py`：下载 MAGI-1 `4.5B_base`，同时下载共享 VAE/T5，并整理到 `downloads/`。该脚本只在 `../PhyT2V/downloader/download_cogvideox5b.py` 的基础上做了 include 过滤和路径整理。
- `download_wan2_2_a14b_diffusers.py`：下载 Wan2.2 I2V/T2V A14B Diffusers 权重到 `../../models`。
- 两个下载器都用 `curl -C -` 支持断点续传；Wan2.2 下载器额外支持 HF token 和 `HF_ENDPOINT` 镜像。

MAGI-1 4.5B 默认下载命令：

```bash
python -u downloader/download_magi1_4_5b.py
```

### 5.2 MAGI-1 4.5B 适配改动

已修改：

```text
generate_magi1.py
generator_i2v_multinode.py
generation/generate_i2v_magi1_multinode.sh
test.sh
```

具体变化：

1. `generate_magi1.py` 和 `generator_i2v_multinode.py` 中，`--config_file` 不再强制必填，默认解析为：

```text
./MAGI-1/example/4.5B/4.5B_base_config.json
```

2. 新增 `--magi_model_variant`，可选：

```text
4.5B_base
4.5B_distill
4.5B_distill_quant
24B_base
```

3. 默认设置 MAGI-1 官方 4.5B 单卡脚本中使用的显存相关环境变量：

```bash
PAD_HQ=1
PAD_DURATION=1
OFFLOAD_T5_CACHE=true
OFFLOAD_VAE_CACHE=true
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

4. `generation/generate_i2v_magi1_multinode.sh` 默认从 8 GPU 改为 1 GPU，并把输出目录从 `MAGI-1` 改成 `MAGI-1-4.5B_base`。

5. `test.sh` 从 `torchrun --nproc_per_node=8` 改为单进程 `python generate_magi1.py`，并默认使用 4.5B 配置。

### 5.3 新增 `generate_wan2_2.py`

建议新增一个单视频生成脚本，职责和 `generate_magi1.py` 对齐：

```text
generate_magi1.py      -> MAGI-1 单样本入口
generate_wan2_2.py     -> Wan2.2 单样本入口
```

需要实现的参数：

```text
--model_path
--prompt
--init_image
--output_path
--height
--width
--num_frames
--fps
--num_inference_steps
--guidance_scale
--dtype
--seed
--cpu_offload
```

它只负责生成视频，不负责 VJEPA 打分。

### 5.4 改造 `generator_i2v_multinode.py`

推荐保留原文件，复制出新文件：

```bash
cp generator_i2v_multinode.py generator_i2v_wan2_2_multinode.py
```

然后修改这些位置：

1. 删除或绕过 MAGI-1 子模块路径依赖：

```python
MAGI1_PATH = os.path.join(..., "MAGI-1")
sys.path.insert(0, MAGI1_PATH)
```

2. 把 `init_pipeline(args)` 改成 Wan2.2 pipeline 初始化：

```python
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    args.wan_model_path,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
```

3. 把 `pipe.run_image_to_video(...)` 替换成 Diffusers 调用：

```python
result = pipe(
    image=init_frame,
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=args.height,
    width=args.width,
    num_frames=args.num_frames,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.cfg_scale,
    generator=generator,
)
frames = result.frames[0]
export_to_video(frames, video_path, fps=args.fps)
```

4. 保留 `rejection` 分支中的 VJEPA 打分逻辑：

```python
compute_vjepa_loss_sliding_window(...)
```

5. 禁用或改名 `guidance`：

```text
choices=["vanilla", "rejection"]
```

或者在用户传入 `--sampling_method guidance` 时直接报错，提示 Wan2.2 guidance 尚未实现。

### 5.5 新增 Wan2.2 批量运行脚本

建议复制原 shell：

```bash
cp generation/generate_i2v_magi1_multinode.sh \
  generation/generate_i2v_wan2_2_multinode.sh
```

需要修改：

```text
MAGI1_CONFIG_FILE      删除
MODEL_OUTPUT_FOLDER    改成 generated_videos/<group>/Wan2.2
SAMPLE_METHODS         改成 ("rejection") 或 ("vanilla" "rejection")
python 脚本名          改成 generator_i2v_wan2_2_multinode.py
--config_file          删除
--wan_model_path       新增，指向 ../../models/Wan2.2-I2V-A14B-Diffusers
--rejection_samples    建议设为 16，对齐论文 BoN 设置
```

### 5.6 修改环境依赖

`environment.yml` 中建议增加或升级：

```text
huggingface_hub
accelerate
safetensors
diffusers
transformers
imageio-ffmpeg
```

如果担心影响 MAGI-1 原始环境，建议单独建一个 Wan2.2 环境，例如：

```bash
conda create -n wmreward-wan python=3.10 -y
conda activate wmreward-wan
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -U diffusers transformers accelerate safetensors decord opencv-python-headless einops imageio-ffmpeg
```

## 6. 推荐复现顺序

### 最小可行版本

```bash
python -u downloader/download_magi1_4_5b.py

python generate_magi1.py \
  --prompt "A ball falls from the table onto the floor" \
  --init_image ./example/0001_switch-frames_anyFPS_perspective-left_trimmed-ball-and-block-fall.jpg \
  --output_path ./results/magi1_4_5b_output.mp4 \
  --mode i2v

python compute_wmreward.py \
  --video_path ./results/magi1_4_5b_output.mp4
```

先确认 MAGI-1 4.5B 能单卡生成视频，且 WMReward 能给输出打分。

### Wan2.2 最小可行版本

```bash
python -u downloader/download_wan2_2_a14b_diffusers.py --variant i2v
python generate_wan2_2.py --model_path ../../models/Wan2.2-I2V-A14B-Diffusers ...
python compute_wmreward.py --video_path ./results/wan2_2_output.mp4
```

先确认 Wan2.2 能生成视频，且 WMReward 能给 Wan2.2 输出打分。

### 轻量论文复现版本

```bash
bash generation/generate_i2v_wan2_2_multinode.sh
```

批量跑 PhysicsIQ，使用 `rejection` / `Best-of-N`，每条样本生成 16 个候选并选择 VJEPA loss 最低的视频。

### 完整 guidance 版本

```text
不建议作为第一阶段目标。
```

需要深入修改 Diffusers Wan2.2 pipeline 的 denoising loop，并把 VJEPA loss 的梯度接回 latent 更新。实现复杂度、显存和耗时都明显高于 Best-of-N。

## 7. 参考

- WMReward paper: https://arxiv.org/pdf/2601.10553
- WMReward code: https://github.com/facebookresearch/WMReward
- Wan2.2 I2V A14B Diffusers: https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B-Diffusers
- Wan2.2 T2V A14B Diffusers: https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers
- Diffusers Wan documentation: https://huggingface.co/docs/diffusers/en/api/pipelines/wan
