#!/bin/bash
# Comparacion CONTROLADA y limpia contra run_ablation_no_heatmap.sh: misma
# curriculum (336 train / 112 val, dataset_splits_thermal.json), mismo
# batch_size=1, SIN gradient_accumulation (efectivo batch=1 igual que
# no_heatmap), mismas epochs/patience. Unica diferencia: loss_vel_focus
# (scale=2, el ganador del barrido) en vez de pura velocity loss.
#
# Esta es la comparacion que de verdad aisla el efecto de la loss de pose,
# a diferencia de run_final_focus.sh (curriculum ampliado, no comparable
# directamente con no_heatmap por el tamanio de datos distinto).
#
# Uso: bash run_final_focus_336.sh <gpu>
set -e
cd "$(dirname "$0")"

GPU=${1:-2}
PY=/home/jupyter-antoniodetoro/entornos_venv/env_qwen/bin/python
LATENTS_DIR=/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512
EXP_JSON=experimentos/dataset_splits_thermal.json
BASE_MODEL=/mnt/nas_dataset/antonio/.qwen/qwen_model_real
TRANSFORMER_MODEL=/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles
OUTPUT_DIR=/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/final_focus_scale2_336

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
    --pose_weight_scale 2 \
    --use_t_scaling \
    --inference_every 5 \
    --inference_steps 4 \
    --inference_samples 6 \
    --epochs 15 \
    --patience 15
