#!/bin/bash
# Prueba corta de ablación: SOLO velocity loss (sin HRNet/heatmap), térmicas.
# Objetivo: comparar curva de val_loss contra run_ablation_with_heatmap.sh
# para decidir si la loss de pose ayuda o no en el dominio térmico.
set -e
cd "$(dirname "$0")"

GPU=${1:-2}
PY=/home/jupyter-antoniodetoro/entornos_venv/env_qwen/bin/python
LATENTS_DIR=/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512
EXP_JSON=experimentos/dataset_splits_thermal.json
OUTPUT_DIR=/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/ablation_no_heatmap
TRANSFORMER_MODEL=/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles
BASE_MODEL=/mnt/nas_dataset/antonio/.qwen/qwen_model_real
# Ruta que NO existe a propósito: finetune.py solo carga HRNet (y por tanto
# loss.py solo decodifica con el VAE para la rama de heatmap) si el archivo
# existe. Con heatmap_loss_weight=0.0 esa rama no aporta nada al gradiente,
# así que nos la ahorramos del todo -> baseline de velocity pura más rápido.
HRNET_PATH=/dev/null/hrnet_disabled_for_pure_velocity_baseline.pth

mkdir -p "$OUTPUT_DIR"

# batch_size=1: la A30 tiene solo 24GB (vs los 32GB de V100 usados en los
# experimentos RGB originales via SLURM) y además es una GPU compartida con
# otros procesos -> batch_size=2 provoca CUDA OOM aquí.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=$GPU $PY finetune.py \
    --reset_patience \
    --experiment_json $EXP_JSON \
    --latents_dir $LATENTS_DIR \
    --base_model $BASE_MODEL \
    --hrnet_model_path $HRNET_PATH \
    --output_dir $OUTPUT_DIR \
    --transformer_model $TRANSFORMER_MODEL \
    --batch_size 1 \
    --learning_rate 1e-4 \
    --heatmap_loss_weight 0.0 \
    --velocity_loss_weight 1.0 \
    --heatmap_loss_type mse \
    --inference_every 5 \
    --inference_steps 4 \
    --inference_samples 6 \
    --epochs 15 \
    --patience 15
