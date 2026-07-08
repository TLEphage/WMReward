# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.s


#!/bin/bash

# SLURM job array configuration for multi-node MAGI-1 execution
#SBATCH --job-name=magi1_phy
#SBATCH --array=0-0                    # 1 nodes (0)
#SBATCH --nodes=1                      # Each job uses 1 node
#SBATCH --qos=h200_dream_high
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --gres=gpu:1                  # MAGI-1 4.5B can run on a single GPU
#SBATCH --cpus-per-task=48            # Adjust based on your cluster
#SBATCH --mem=128G                    # Adjust based on your cluster
#SBATCH --time=24:00:00               # Adjust based on expected runtime
#SBATCH --output=jobs/job_%A_%a.out
#SBATCH --error=jobs/job_%A_%a.err

# Activate your conda environment before running this script, e.g.:
#   conda activate wmreward

nvidia-smi

# Multi-node configuration
NUM_NODES=${NUM_NODES:-1}             # Total number of nodes
NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-1}  # GPUs per node
TOTAL_GPUS=$((NUM_NODES * NUM_GPUS_PER_NODE))
NODE_ID=${SLURM_ARRAY_TASK_ID:-0}     # Current node ID
# NODE_ID=0

echo "Starting node ${NODE_ID} of ${NUM_NODES} (GPUs per node: ${NUM_GPUS_PER_NODE}, Total GPUs: ${TOTAL_GPUS})"

# Legacy/rejection-sampling hyperparameter triplets
# The current MAGI guidance path ignores these and only consumes guidance_scale/frequency/backbone.
TRIPLETS=(
    "16 8 8"   # window=16, context_frames=8, stride=8
)

SEED_LIST=(42)

GUIDANCE_STEP_PATTERN="0x5,1x45"
GUIDANCE_LR_PATTERNS=("0.001x50")
GUIDANCE_SCALE="${GUIDANCE_SCALE:-0.001}"
GUIDANCE_FREQUENCY="${GUIDANCE_FREQUENCY:-1}"
VJEPA_GUIDANCE_MAX_CALLS="${VJEPA_GUIDANCE_MAX_CALLS:-1}"
ENABLE_ADAPTIVE_GUIDANCE="${ENABLE_ADAPTIVE_GUIDANCE:-0}"
ADAPTIVE_GUIDANCE_T_MIN="${ADAPTIVE_GUIDANCE_T_MIN:-0.25}"
ADAPTIVE_GUIDANCE_T_MAX="${ADAPTIVE_GUIDANCE_T_MAX:-0.75}"
ADAPTIVE_GUIDANCE_TARGET_T="${ADAPTIVE_GUIDANCE_TARGET_T:-0.50}"
VJEPA_GUIDANCE_TARGET_FPS="${VJEPA_GUIDANCE_TARGET_FPS:-16}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-6009}"
ADAPTIVE_GUIDANCE_ARGS=()
if [[ "$ENABLE_ADAPTIVE_GUIDANCE" == "1" || "$ENABLE_ADAPTIVE_GUIDANCE" == "true" ]]; then
    ADAPTIVE_GUIDANCE_ARGS=(
        --enable_adaptive_guidance
        --adaptive_guidance_t_min "$ADAPTIVE_GUIDANCE_T_MIN"
        --adaptive_guidance_t_max "$ADAPTIVE_GUIDANCE_T_MAX"
        --adaptive_guidance_target_t "$ADAPTIVE_GUIDANCE_TARGET_T"
    )
fi

# CFG scale values for classifier-free guidance ablation
CFG_SCALES=("6.0")

# Disable Time Travel for simple algorithm
GUIDANCE_RANGES=("0 0")

# Path to MAGI-1 config file. Override MAGI1_MODEL_VARIANT with
# 4.5B_distill, 4.5B_distill_quant, or 24B_base if needed.
MAGI1_MODEL_VARIANT="${MAGI1_MODEL_VARIANT:-4.5B_base}"
if [[ "$MAGI1_MODEL_VARIANT" == 4.5B_* ]]; then
    MAGI1_CONFIG_FILE="./MAGI-1/example/4.5B/${MAGI1_MODEL_VARIANT}_config.json"
elif [[ "$MAGI1_MODEL_VARIANT" == "24B_base" ]]; then
    MAGI1_CONFIG_FILE="./MAGI-1/example/24B/24B_base_config.json"
else
    echo "Unsupported MAGI1_MODEL_VARIANT=$MAGI1_MODEL_VARIANT"
    exit 1
fi

# JSON batch describing entries with input image/video, prompt, and output path
# Add or remove batch JSON files as needed
BATCH_JSON_LIST=(
    # Physics-IQ dataset
    "./prompts/physics_iq.json"
)
BASEDIR="${BASEDIR:-./PhysicsIQ/code}"
OUTPUT_FOLDER="./generated_videos"

# A100 40G friendly default. Override, for example:
#   SAMPLE_METHODS_OVERRIDE="rejection" bash generation/generate_i2v_magi1_multinode.sh
#   SAMPLE_METHODS_OVERRIDE="guidance" bash generation/generate_i2v_magi1_multinode.sh
if [[ -n "${SAMPLE_METHODS_OVERRIDE:-}" ]]; then
    read -r -a SAMPLE_METHODS <<< "$SAMPLE_METHODS_OVERRIDE"
else
    SAMPLE_METHODS=("vanilla")
fi
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-48}"
NUM_FRAMES="${NUM_FRAMES:-49}"
VIDEO_HEIGHT="${VIDEO_HEIGHT:-480}"
VIDEO_WIDTH="${VIDEO_WIDTH:-720}"
BATCH_START_IDX="${BATCH_START_IDX:-0}"
BATCH_MAX_ENTRIES="${BATCH_MAX_ENTRIES:-0}"
REJECTION_SAMPLES="${REJECTION_SAMPLES:-10}"  # Number of candidates to generate for rejection sampling
ENABLE_PROMPT_EXPANSION="${ENABLE_PROMPT_EXPANSION:-0}"
PROMPT_EXPANSION_MODE="${PROMPT_EXPANSION_MODE:-coect}"
PROMPT_EXPANSION_MAX_EVENTS="${PROMPT_EXPANSION_MAX_EVENTS:-4}"
PROMPT_EXPANSION_ARGS=()
if [[ "$ENABLE_PROMPT_EXPANSION" == "1" || "$ENABLE_PROMPT_EXPANSION" == "true" ]]; then
    PROMPT_EXPANSION_ARGS=(
        --enable_prompt_expansion
        --prompt_expansion_mode "$PROMPT_EXPANSION_MODE"
        --prompt_expansion_max_events "$PROMPT_EXPANSION_MAX_EVENTS"
    )
fi

# I2V conditioning comes from JSON (input_video or image); no static INIT_IMAGE here

# V-JEPA slice-pred fixed settings
if [[ -n "${VJEPA_VARIANT_OVERRIDE:-}" ]]; then
    read -r -a VJEPA_VARIANTS <<< "$VJEPA_VARIANT_OVERRIDE"
else
    VJEPA_VARIANTS=("vit_giant")
fi
VJEPA_IMG_SIZE=256
VJEPA_MASKING_MODE="causal"
# Loss aggregation modes to iterate over
LOSS_MODES=("mean")


mkdir -p "$OUTPUT_FOLDER"

for BATCH_JSON in "${BATCH_JSON_LIST[@]}"; do
for SAMPLE_METHOD in "${SAMPLE_METHODS[@]}"; do

    if [[ "$SAMPLE_METHOD" == "guidance" ]]; then
        ACTIVE_TRIPLETS=("${TRIPLETS[0]}")
        ACTIVE_GUIDANCE_RANGES=("0 0")
        ACTIVE_LOSS_MODES=("${LOSS_MODES[0]}")
        ACTIVE_GUIDANCE_LR_PATTERNS=("${GUIDANCE_LR_PATTERNS[0]}")
    else
        ACTIVE_TRIPLETS=("${TRIPLETS[@]}")
        ACTIVE_GUIDANCE_RANGES=("${GUIDANCE_RANGES[@]}")
        ACTIVE_LOSS_MODES=("${LOSS_MODES[@]}")
        ACTIVE_GUIDANCE_LR_PATTERNS=("${GUIDANCE_LR_PATTERNS[@]}")
    fi

    for triplet in "${ACTIVE_TRIPLETS[@]}"; do
            # Split triplet into individual variables
            read -r SLICE_WINDOW_SIZE CONTEXT_LENGTH STRIDE <<< "$triplet"
            for guidance_range in "${ACTIVE_GUIDANCE_RANGES[@]}"; do
                # Split guidance range into start and end values (GLOBAL 0..49)
                read -r GUIDANCE_START GUIDANCE_END <<< "$guidance_range"
                TRAVEL_TIME="${GUIDANCE_START},${GUIDANCE_END}"
                for CFG_SCALE in "${CFG_SCALES[@]}"; do
                    for LOSS_MODE in "${ACTIVE_LOSS_MODES[@]}"; do
                    for VJEPA_VARIANT in "${VJEPA_VARIANTS[@]}"; do
                    echo "Config: Method=$SAMPLE_METHOD, GuidanceScale=$GUIDANCE_SCALE, GuidanceFreq=$GUIDANCE_FREQUENCY, VJEPA_MAX_CALLS=$VJEPA_GUIDANCE_MAX_CALLS, Adaptive=$ENABLE_ADAPTIVE_GUIDANCE, CFG=$CFG_SCALE, VJEPA=$VJEPA_VARIANT"

                    # Match structure: <OUTPUT_FOLDER>/<group>/<experiment>/<name>.mp4
                    if [[ "$(basename "$BATCH_JSON")" == "physics_iq.json" ]]; then
                        GROUP_NAME="physics_iq"
                    elif [[ "$(basename "$BATCH_JSON")" == "physics_iq_multiframe.json" ]]; then
                        GROUP_NAME="physics_iq_multiframe"
                    else
                        GROUP_NAME=$(basename "$(dirname "$BATCH_JSON")")
                    fi
                    MODEL_OUTPUT_FOLDER="${OUTPUT_FOLDER}/${GROUP_NAME}/MAGI-1-${MAGI1_MODEL_VARIANT}"
                    mkdir -p "$MODEL_OUTPUT_FOLDER"

                    # Loop over LR patterns; pass base output folder and let Python name runs
                    for GUIDANCE_LR_PATTERN in "${ACTIVE_GUIDANCE_LR_PATTERNS[@]}"; do
                        RUN_OUTPUT_FOLDER="$MODEL_OUTPUT_FOLDER"
                        mkdir -p "$RUN_OUTPUT_FOLDER"

                        # Launch one worker per GPU on this node; each worker shards the JSON by global index
                        for SEED in "${SEED_LIST[@]}"; do
                        for ((g=0; g<NUM_GPUS_PER_NODE; g++)); do
                            # Calculate global GPU index across all nodes
                            GLOBAL_GPU_IDX=$((NODE_ID * NUM_GPUS_PER_NODE + g))
                            WORKER_MASTER_PORT=$((MASTER_PORT_BASE + GLOBAL_GPU_IDX))
                            echo "  -> Launching worker on Node $NODE_ID, Local GPU $g (Global GPU $GLOBAL_GPU_IDX, MASTER_PORT=$WORKER_MASTER_PORT) with LR pattern $GUIDANCE_LR_PATTERN"
                            CUDA_VISIBLE_DEVICES=$g MASTER_ADDR=localhost MASTER_PORT=$WORKER_MASTER_PORT GPUS_PER_NODE=1 NNODES=1 WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 python generator_i2v_multinode.py \
                                --config_file "$MAGI1_CONFIG_FILE" \
                                --magi_model_variant "$MAGI1_MODEL_VARIANT" \
                                --output_folder "$RUN_OUTPUT_FOLDER" \
                                --batch_json "$BATCH_JSON" \
                                --base_dir "$BASEDIR" \
                                --start_idx "$BATCH_START_IDX" \
                                --max_entries "$BATCH_MAX_ENTRIES" \
                                --num_gpus $TOTAL_GPUS \
                                --gpu_idx $GLOBAL_GPU_IDX \
                                --num_nodes $NUM_NODES \
                                --node_id $NODE_ID \
                                --gpus_per_node $NUM_GPUS_PER_NODE \
                                --sampling_method "$SAMPLE_METHOD" \
                                --num_inference_steps $NUM_SAMPLING_STEPS \
                                --num_frames $NUM_FRAMES \
                                --height $VIDEO_HEIGHT \
                                --width $VIDEO_WIDTH \
                                --cfg_scale $CFG_SCALE \
                                "${PROMPT_EXPANSION_ARGS[@]}" \
                                --guidance_scale $GUIDANCE_SCALE \
                                --vjepa_variant $VJEPA_VARIANT \
                                --vjepa_img_size $VJEPA_IMG_SIZE \
                                --vjepa_masking_mode $VJEPA_MASKING_MODE \
                                --vjepa_context_frames $CONTEXT_LENGTH \
                                --slice_stride $STRIDE \
                                --slice_window_size $SLICE_WINDOW_SIZE \
                                --guidance_step_pattern "$GUIDANCE_STEP_PATTERN" \
                                --guidance_lr_pattern "$GUIDANCE_LR_PATTERN" \
                                --guidance_frequency $GUIDANCE_FREQUENCY \
                                --vjepa_guidance_max_calls $VJEPA_GUIDANCE_MAX_CALLS \
                                --vjepa_guidance_target_fps $VJEPA_GUIDANCE_TARGET_FPS \
                                "${ADAPTIVE_GUIDANCE_ARGS[@]}" \
                                --loss_mode "$LOSS_MODE" \
                                --rejection_samples $REJECTION_SAMPLES \
                                --config_version "v2" \
                                --seed $SEED &
                        done
                        wait
                    done
                    done
                    done
                    done
                done
            done
    done
done
done

echo "Node ${NODE_ID} experiments completed! Results saved to: $OUTPUT_FOLDER"
