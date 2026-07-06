#!/usr/bin/env bash

# Run from the WMReward repo root.

# MAGI-1 4.5B is the lightweight MAGI-1 default used by this repo.
nohup python -u downloader/download_magi1_4_5b.py \
  > downloader/download_magi1_4_5b.log 2>&1 &

# Progress:
# tail -f downloader/download_magi1_4_5b.log
# watch -n 5 'du -sh downloads/4.5B_* downloads/vae downloads/t5_pretrained 2>/dev/null'

# I2V is the model needed for PhysicsIQ-style image-conditioned generation.
# nohup python -u downloader/download_wan2_2_a14b_diffusers.py \
#   --variant i2v \
#   > downloader/download_wan2_2_a14b_diffusers.log 2>&1 &

# Optional: also fetch text-to-video weights.
# nohup python -u downloader/download_wan2_2_a14b_diffusers.py \
#   --variant t2v \
#   > downloader/download_wan2_2_t2v_a14b_diffusers.log 2>&1 &

# Wan2.2 progress:
# tail -f downloader/download_wan2_2_a14b_diffusers.log
# watch -n 5 'du -sh ../../models/Wan2.2-*A14B-Diffusers 2>/dev/null'
