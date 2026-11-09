#!/bin/bash
# Run FINAL (no cribado): loss_vel_focus con pose_weight_scale=2 (ganador del
# barrido fino: robusto en 0.5-8, 2.0 el mejor puntual) sobre el curriculum
# AMPLIADO (840 train / 224 val / 224 test, ~2.5x el original de 336/112),
# generado por build_expanded_curriculum.py con la misma metodologia de
# muestreo por rareza de pose que el curriculum original.
#
# Uso: bash run_final_focus.sh <gpu>
set -e
cd "$(dirname "$0")"

GPU=${1:-2}
PY=/home/jupyter-antoniodetoro/entornos_venv/env_qwen/bin/python
LATENTS_DIR=/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512
EXP_JSON=experimentos/dataset_splits_thermal_expanded.json
BASE_MODEL=/mnt/nas_dataset/antonio/.qwen/qwen_model_real
TRANSFORMER_MODEL=/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles
OUTPUT_DIR=/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/final_focus_scale2_expanded

mkdir -p "$OUTPUT_DIR"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=$GPU $PY finetune_focus_vel.py \
    --reset_patience \
    --experiment_json $EXP_JSON \
    --latents_dir $LATENTS_DIR \
    --base_model $BASE_MODEL \
    --transformer_model $TRANSFORMER_MODEL \
    --output_dir $OUTPUT_DIR \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --pose_weight_scale 2 \
    --use_t_scaling \
    --inference_every 5 \
    --inference_steps 4 \
    --inference_samples 6 \
    --epochs 15 \
    --patience 15
