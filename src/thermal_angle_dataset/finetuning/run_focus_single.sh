#!/bin/bash
# Corre UNA sola escala de pose_weight_scale del cribado de loss_vel_focus.
# Pensado para ser lanzado como job independiente por el scheduler (uno por
# escala), de forma que se repartan en paralelo entre las GPUs libres en vez
# de correr las 5 secuencialmente en una sola GPU (ver run_focus_sweep.sh,
# ahora obsoleto para este uso).
#
# Uso: bash run_focus_single.sh <gpu> <scale>
set -e
cd "$(dirname "$0")"

GPU=$1
SCALE=$2
if [ -z "$GPU" ] || [ -z "$SCALE" ]; then
    echo "Uso: bash run_focus_single.sh <gpu> <scale>"
    exit 1
fi

PY=/home/jupyter-antoniodetoro/entornos_venv/env_qwen/bin/python
LATENTS_DIR=/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512
EXP_JSON=experimentos/dataset_splits_thermal_screening.json
BASE_MODEL=/mnt/nas_dataset/antonio/.qwen/qwen_model_real
TRANSFORMER_MODEL=/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles
OUTPUT_DIR=/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/focus_sweep/scale_${SCALE}
EPOCHS=4
PATIENCE=99

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
    --pose_weight_scale $SCALE \
    --use_t_scaling \
    --inference_every 0 \
    --epochs $EPOCHS \
    --patience $PATIENCE
