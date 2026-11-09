#!/bin/bash
# Prueba corta de ablación: velocity + heatmap loss (HRNet), térmicas.
# Mismos hiperparámetros que run_ablation_no_heatmap.sh salvo los pesos de loss,
# para que la comparación sea limpia. Pesos 0.9/0.1 + weighted_mse = la combinación
# ganadora en los experimentos RGB (exp_largo_hrnet_90_10).
# IMPORTANTE: requiere haber corrido antes generate_heatmaps_thermal.py (añade
# "target_heatmaps" a los .pt del subconjunto del curriculum).
set -e
cd "$(dirname "$0")"

GPU=${1:-2}
PY=/home/jupyter-antoniodetoro/entornos_venv/env_qwen/bin/python
LATENTS_DIR=/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512
EXP_JSON=experimentos/dataset_splits_thermal.json
OUTPUT_DIR=/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/ablation_with_heatmap
TRANSFORMER_MODEL=/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles
BASE_MODEL=/mnt/nas_dataset/antonio/.qwen/qwen_model_real
HRNET_PATH=/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/src/hr_net/outputs/27kp_sucio_v1/hrnet_finetuned_best.pth

mkdir -p "$OUTPUT_DIR"

# batch_size=1: ver nota en run_ablation_no_heatmap.sh (A30 24GB compartida).
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
    --heatmap_loss_weight 0.1 \
    --velocity_loss_weight 0.9 \
    --heatmap_loss_type weighted_mse \
    --inference_every 5 \
    --inference_steps 4 \
    --inference_samples 6 \
    --epochs 15 \
    --patience 15
