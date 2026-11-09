#!/bin/bash
# Control de seleccion de datos: EXACTAMENTE la misma configuracion que
# run_ablation_no_heatmap.sh (velocity loss pura, sin heatmap), mismo tamanio
# 336 train / 112 val / 112 test, mismo test set. Unica diferencia: el
# curriculum de train/val se eligio uniformemente al azar (build_random_curriculum.py)
# en vez de por rareza de pose (knn_score).
#
# Esta es LA comparacion controlada que aisla si el curriculum por rareza de
# pose aporta algo frente a una seleccion aleatoria de igual tamanio:
# ablation_no_heatmap (rareza) vs ablation_random_curriculum (random).
#
# Uso: bash run_ablation_random_curriculum.sh <gpu>
set -e
cd "$(dirname "$0")"

GPU=${1:-2}
PY=/home/jupyter-antoniodetoro/entornos_venv/env_qwen/bin/python
LATENTS_DIR=/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512
EXP_JSON=experimentos/dataset_splits_thermal_random.json
OUTPUT_DIR=/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/ablation_random_curriculum
TRANSFORMER_MODEL=/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles
BASE_MODEL=/mnt/nas_dataset/antonio/.qwen/qwen_model_real
HRNET_PATH=/dev/null/hrnet_disabled_for_pure_velocity_baseline.pth

mkdir -p "$OUTPUT_DIR"

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
    --inference_every 3 \
    --inference_steps 4 \
    --inference_samples 10000 \
    --epochs 15 \
    --patience 15
