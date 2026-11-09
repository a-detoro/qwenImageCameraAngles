#!/bin/bash
# Barrido DEFINITIVO de pose_weight_scale sobre el curriculum completo 336/112
# (mismo que run_final_focus_336.sh / ablation_no_heatmap), 15 epocas,
# patience=15, mismo batch efectivo (batch_size=1, sin grad accumulation).
#
# A diferencia del barrido rapido anterior (focus_sweep, 4 epocas sobre
# curriculum reducido de cribado), este corre el curriculum y presupuesto de
# epocas COMPLETOS para poder comparar PCK/MSE/SSIM por epoca de forma
# consistente con no_heatmap_ablation y final_focus_scale2_336 (scale=2).
#
# inference_every=3 (epocas 2,5,8,11,14) e inference_samples muy alto para
# forzar que se use TODO el test set disponible (min(samples, len(dataset))),
# en vez de las 6 imagenes que se guardaban antes - necesario para que el PCK
# agregado no dependa de una muestra tan pequena.
#
# Uso: bash run_scale_sweep_full.sh <gpu> <scale>
set -e
cd "$(dirname "$0")"

GPU=${1:-2}
SCALE=${2:?"Uso: run_scale_sweep_full.sh <gpu> <scale>"}
PY=/home/jupyter-antoniodetoro/entornos_venv/env_qwen/bin/python
LATENTS_DIR=/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512
EXP_JSON=experimentos/dataset_splits_thermal.json
BASE_MODEL=/mnt/nas_dataset/antonio/.qwen/qwen_model_real
TRANSFORMER_MODEL=/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles
OUTPUT_DIR=/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/scale_sweep_full/scale_${SCALE}

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
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --pose_weight_scale $SCALE \
    --use_t_scaling \
    --inference_every 3 \
    --inference_steps 4 \
    --inference_samples 10000 \
    --epochs 15 \
    --patience 15
