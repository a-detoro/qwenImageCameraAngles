#!/bin/bash
# Cribado rápido de pose_weight_scale para loss_vel_focus, sobre un subconjunto
# reducido (3 usuarios, 72 train / 24 val timestamps) y pocas épocas, en vez del
# curriculum completo (336/112, 15+ épocas) que tarda >30h por configuración.
#
# Objetivo: NO buscar el valor óptimo final aquí, solo descartar rápido y ver
# si hay divergencia de tendencia entre escalas antes de gastar tiempo en el
# run completo con el/los candidatos que sobrevivan.
#
# Uso: bash run_focus_sweep.sh [gpu_id]
set -e
cd "$(dirname "$0")"

PY=/home/jupyter-antoniodetoro/entornos_venv/env_qwen/bin/python
LATENTS_DIR=/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512
EXP_JSON=experimentos/dataset_splits_thermal_screening.json
BASE_MODEL=/mnt/nas_dataset/antonio/.qwen/qwen_model_real
TRANSFORMER_MODEL=/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles
OUTPUT_ROOT=/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/focus_sweep
GPU=${1:-2}

# Barrido logarítmico amplio: si ni en los extremos hay diferencia, el problema
# no es la escala del peso sino el mecanismo (ver conversación: con 4-5 en RGB
# no se notó diferencia, puede ser que el efecto esté diluido por la fracción de
# tokens que caen en la zona de pose, no que el enfoque no sirva).
SCALES=(0.5 2 8 32 64)
EPOCHS=4
PATIENCE=99   # no queremos early stopping durante el cribado, solo ver tendencia

mkdir -p "$OUTPUT_ROOT"

for scale in "${SCALES[@]}"; do
    OUTPUT_DIR="$OUTPUT_ROOT/scale_${scale}"
    mkdir -p "$OUTPUT_DIR"
    echo "========================================"
    echo "pose_weight_scale = $scale -> $OUTPUT_DIR"
    echo "========================================"

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
        --pose_weight_scale $scale \
        --use_t_scaling \
        --inference_every 0 \
        --epochs $EPOCHS \
        --patience $PATIENCE \
        2>&1 | tee -a "$OUTPUT_DIR/log.txt"
done

echo "Barrido completo. Analiza con: python compare_focus_sweep.py"
