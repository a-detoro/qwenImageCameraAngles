#!/bin/bash
# Lanza el precómputo de latentes térmicos completo (train+val+test) repartido
# entre las dos GPUs libres (0 = V100, 2 = A30), dentro de una sesión tmux
# llamada "thermal_precompute" para que sobreviva aunque se cierre esta shell.
#
# Uso:
#   bash launch_thermal_precompute.sh
#
# Seguimiento:
#   tmux attach -t thermal_precompute       # ver en vivo (Ctrl+B, D para salir sin matar)
#   tmux capture-pane -t thermal_precompute:gpu0 -p | tail -50   # ver progreso GPU0 sin adjuntarse
#   tmux capture-pane -t thermal_precompute:gpu2 -p | tail -50   # ver progreso GPU2 sin adjuntarse

set -e

PROJECT_ROOT="/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2"
SCRIPT_DIR="$PROJECT_ROOT/src/thermal_angle_dataset"
PYTHON="/home/jupyter-antoniodetoro/entornos_venv/env_qwen/bin/python"
DATASET_ROOT="$PROJECT_ROOT/datos/MultiViewVisibleThermalImagesHPE"
OUTPUT_DIR="/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512"
RESOLUTION=512
SESSION="thermal_precompute"

mkdir -p "$OUTPUT_DIR" "$SCRIPT_DIR/logs"

# Tamaños de cada split (calculados de antemano con la misma lógica de muestreo
# que usa el dataset: 91794 pares train_val -> 82614 train / 9180 val; 18930 test)
TRAIN_TOTAL=82614
VAL_TOTAL=9180
TEST_TOTAL=18930

TRAIN_HALF=$((TRAIN_TOTAL / 2))
VAL_HALF=$((VAL_TOTAL / 2))
TEST_HALF=$((TEST_TOTAL / 2))

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "La sesión tmux '$SESSION' ya existe. Adjúntate con: tmux attach -t $SESSION"
    exit 1
fi

tmux new-session -d -s "$SESSION" -n gpu0
tmux new-window -t "$SESSION" -n gpu2

run_cmd() {
    local gpu=$1 split=$2 start=$3 end=$4
    echo "CUDA_VISIBLE_DEVICES=$gpu $PYTHON precompute_latents_thermal.py --dataset_root $DATASET_ROOT --output_dir $OUTPUT_DIR --split $split --start_idx $start --end_idx $end --resolution $RESOLUTION 2>&1 | tee -a $SCRIPT_DIR/logs/gpu${gpu}_${split}.log"
}

# GPU 0: primera mitad de cada split
GPU0_CMDS="cd $SCRIPT_DIR && \
$(run_cmd 0 train 0 $TRAIN_HALF) && \
$(run_cmd 0 val 0 $VAL_HALF) && \
$(run_cmd 0 test 0 $TEST_HALF) ; \
echo '=== GPU0 TERMINADO ==='; bash"

# GPU 2: segunda mitad de cada split
GPU2_CMDS="cd $SCRIPT_DIR && \
$(run_cmd 2 train $TRAIN_HALF -1) && \
$(run_cmd 2 val $VAL_HALF -1) && \
$(run_cmd 2 test $TEST_HALF -1) ; \
echo '=== GPU2 TERMINADO ==='; bash"

tmux send-keys -t "$SESSION:gpu0" "$GPU0_CMDS" C-m
tmux send-keys -t "$SESSION:gpu2" "$GPU2_CMDS" C-m

echo "Lanzado en tmux. Sesión: $SESSION (ventanas: gpu0, gpu2)"
echo "Adjuntarse:   tmux attach -t $SESSION"
echo "Ver logs:     tail -f $SCRIPT_DIR/logs/gpu0_train.log"
echo "              tail -f $SCRIPT_DIR/logs/gpu2_train.log"
echo "Salida final: $OUTPUT_DIR"
