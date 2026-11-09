# Contexto para continuar: finetuning Qwen-Image-Edit-Angles sobre imágenes térmicas

Resumen de estado a 2026-11-09, pensado para retomar el trabajo en otro servidor
(sin acceso al NAS/GPUs de este, al menos de entrada). Todo el código relevante
está en este repo bajo `src/thermal_angle_dataset/`; lo que NO está en el repo
(pesos, datasets, latentes, salidas de inferencia) se describe más abajo para
poder reconstruirlo o reapuntarlo.

## 1. Objetivo del proyecto

`Qwen-Image-Edit-Angles-2` es un finetuning de Qwen-Image-Edit para cambio de
perspectiva/ángulo de cámara. Esta rama de trabajo lo extiende a **imágenes
térmicas** (mismo dataset multivista, pero en el dominio térmico en vez de
RGB), reutilizando los keypoints de pose extraídos de las imágenes RGB
(RGB y térmica comparten timestamps exactos, así que la pose es idéntica).

Dos preguntas abiertas que se han estado investigando con experimentos
controlados:

1. **¿Ayuda una loss de velocidad ponderada por pose (`loss_vel_focus`,
   parámetro `pose_weight_scale`) frente a la velocity loss pura?**
2. **¿Aporta algo seleccionar el curriculum de entrenamiento por rareza de
   pose (KNN-density) frente a una selección aleatoria del mismo tamaño?**

**Lección metodológica importante, ya verificada dos veces:** la loss de
entrenamiento NO es un proxy fiable de si un método "funciona" de cara al
resultado real. Hay que medir sobre las imágenes generadas con PCK (pose),
MSE y SSIM (fidelidad visual), no solo con la loss. Ambas preguntas de arriba
se han revertido en su conclusión al pasar de "comparar por loss" a
"comparar por PCK/MSE/SSIM".

## 2. Qué hay ya decidido/concluido (con datos)

### 2.1 Selección de curriculum por rareza de pose vs. aleatorio — RESUELTO

Comparación controlada (misma loss de velocidad pura, mismo tamaño 336
train / 112 val, mismo test set fijo, mismo nº de épocas):

- `ablation_no_heatmap` = curriculum por rareza de pose (KNN-density)
- `ablation_random_curriculum` = mismo tamaño pero elegido al azar
  (`build_random_curriculum.py`, seed=1234)

Resultado (`src/thermal_angle_dataset/finetuning/pck_eval_curriculum/pck_results_summary.csv`),
en la época 14 (checkpoint común):

| métrica | rareza (no_heatmap) | aleatorio |
|---|---|---|
| PCK val (n=28) | 46.85% | **75.00%** |
| PCK test | 65.69% (n=6) | **70.27%** (n=672) |
| MSE val | 406.1 | **259.6** |
| SSIM val | 0.894 | **0.908** |

**Conclusión: el curriculum aleatorio gana consistentemente al de rareza de
pose**, en val y test, en toda la trayectoria de épocas, no solo al final.
Hipótesis no verificadas: las poses "raras" seleccionadas pueden ser en
realidad outliers/ruido (oclusiones, estimaciones de pose erróneas) en vez
de señal útil, o el muestreo por rareza sesga la distribución de ángulos/
posturas de forma que generaliza peor al test fijo. Si se quiere profundizar,
este sería el siguiente paso natural (no iniciado).

### 2.2 `loss_vel_focus` con `pose_weight_scale=2`, curriculum controlado 336/112 — parcialmente resuelto

(`src/thermal_angle_dataset/finetuning/pck_eval/pck_results_summary.csv`,
experimentos `ablation_no_heatmap` vs `final_focus_scale2_336`)

| época | PCK test no_heatmap (vel. pura) | PCK test focus scale=2 |
|---|---|---|
| 4 | 13.73 | **34.31** (focus gana) |
| 9 | **58.82** | 42.16 (no_heatmap ya gana) |
| 14 | **65.69** | 48.04 (no_heatmap gana claramente) |

**Patrón: la focus loss ayuda en épocas tempranas pero pierde frente a la
velocity loss pura según avanza el entrenamiento**, en igualdad de
condiciones de datos. `final_focus_scale2_expanded` (840 train, 2.5x más
datos) llega a PCK test=81.37 en época 14, pero mezcla más datos + la loss,
así que no aísla el efecto de la loss por sí sola.

### 2.3 Barrido de escalas (0.5 / 1 / 5) sobre el curriculum completo — **ENTRENADO PERO NO EVALUADO** (ver sección 3, es el siguiente paso más obvio)

## 3. Qué falta por hacer (siguiente paso más directo)

El entrenamiento del barrido definitivo de escalas **ya terminó con éxito**
(`scheduler/scheduler_state.json`: `scale_sweep_full_0.5`, `scale_sweep_full_1`,
`scale_sweep_full_5` todos `status: done, exit_code: 0`), sobre el curriculum
completo 336/112, 15 épocas, con inferencia cada 3 épocas sobre todo el test
set. Las imágenes generadas existen en
`OUTPUTS_EXPERIMENTOS_THERMAL/scale_sweep_full/scale_{0.5,1,5}/` (en el NAS,
ver sección 4).

**Lo que NO se ha hecho todavía**: correr `evaluate_pck_rtmw.py` sobre esas
tres escalas para obtener PCK/MSE/SSIM por época y así cerrar definitivamente
la pregunta de la focus loss con las 5 escalas (0.5, 1, 2, 5 — el 2 ya está en
`pck_eval/`). Solo existe una evaluación parcial e incompleta en CPU de
`scale_0.5` (épocas 2 y 5, `pck_eval_sweep/`), hecha ANTES de arreglar el
proveedor GPU de onnxruntime, y abandonada quedó a medias cuando la atención
se desvió a la comparación de curriculum (sección 2.1). Es segura de repetir/
completar en GPU ahora que el fix de onnxruntime está aplicado.

Comando de referencia (mismo patrón que se usó para `pck_eval_curriculum`,
con symlinks en `_root/` para no reprocesar lo ya evaluado):

```bash
cd src/thermal_angle_dataset/finetuning
python evaluate_pck_rtmw.py \
    --root <symlink_dir_con_scale_0.5_1_5> \
    --out_dir pck_eval_sweep_full \
    --device cuda --viz_sample_n 12
```

Con esto se tendría la tabla completa (0.5, 1, 2, 5) para responder con
certeza "¿a qué escala, si alguna, ayuda `loss_vel_focus`".

## 4. Qué NO está en el repo (pesos, datasets, salidas) y hay que resolver en el otro servidor

Todo esto vive en `/mnt/nas_dataset/antonio/qwen/...` en este servidor (montaje
NAS), **no está en git** (por tamaño) y no existirá en el otro servidor salvo
que se copie/remonte aparte:

- `BASE_MODEL=/mnt/nas_dataset/antonio/.qwen/qwen_model_real` — pesos base de Qwen.
- `TRANSFORMER_MODEL=/mnt/nas_dataset/antonio/qwen/Qwen-Fused-Angles` — checkpoint fusionado de angles.
- `LATENTS_DIR=/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512` — latentes precomputados del dataset térmico.
- `OUTPUT_DIR=/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/...` — checkpoints + imágenes de inferencia de cada experimento.
- HRNet: `src/hr_net/models/pose_hrnet_w48_384x288.pth`, `hrnet_finetuned_best.pth`, etc. (ignorados por `.gitignore`, `*.pth`).
- Dataset multivista original (`datos/`, ignorado por `.gitignore`).

Antes de poder relanzar cualquier entrenamiento/evaluación en el otro
servidor, hay que decidir cómo traer (o regenerar) al menos: pesos base de
Qwen + el `Qwen-Fused-Angles` checkpoint, y los latentes térmicos
precomputados (`src/thermal_angle_dataset/precompute_latents_thermal.py` es
el script que los generó, por si hay que regenerarlos desde cero a partir del
dataset multivista térmico en vez de copiar los `.pt`).

Para empezar en el otro servidor sin pesos/datos, lo que sí se puede hacer de
inmediato es trabajo de análisis/código: leer `pck_eval*/pck_results_*.csv`
(sí están en git) y seguir con la sección 3 una vez los datos estén
disponibles, o avanzar en el guion/redacción de resultados para el TFG/paper.

## 5. Mapa de archivos clave

```
src/thermal_angle_dataset/
├── precompute_latents_thermal.py      # genera los .pt de latentes térmicos
├── launch_thermal_precompute.sh
├── sanity_check/check_vae_roundtrip.py
└── finetuning/
    ├── finetune.py                    # baseline: velocity loss pura (HRNet desactivado a propósito)
    ├── finetune_focus_vel.py          # loss_vel_focus (pose_weight_scale)
    ├── loss_vel_focus.py              # weighting espacial estático de la velocity MSE por pose
    ├── evaluate_pck_rtmw.py           # RTMW -> PCK/MSE/SSIM + triptychs + gráficas (script central, reusar)
    ├── build_random_curriculum.py     # curriculum de control aleatorio
    ├── build_expanded_curriculum.py   # reconstrucción del algoritmo de curriculum por rareza (ampliado)
    ├── experimentos/*.json            # definiciones de splits train/val/test
    ├── run_*.sh                       # scripts de lanzamiento parametrizados por experimento
    ├── scheduler/
    │   ├── gpu_scheduler.py           # scheduler casero de GPUs (poll 45s, prioriza GPU A30/idx2, fallback V100 por mem libre)
    │   ├── jobs_queue.json            # cola de jobs (hot-reload)
    │   └── scheduler_state.json       # estado/resultado de cada job
    ├── pck_eval/                      # resultados: no_heatmap vs focus_scale2 (336 y expanded)
    └── pck_eval_curriculum/           # resultados: rareza vs aleatorio (sección 2.1)
```

`notebooks/viendo_diferentes_con_k-NN+t-NSE.ipynb` (no está en este repo
todavía) es el notebook origen de `timestamps_by_rarity.csv`, que alimenta
tanto el curriculum original por rareza como `build_random_curriculum.py`.
Si hace falta reconstruir pools de timestamps por usuario, es la referencia.

## 6. Notas de entorno

- Dependencias: `requirements.txt` / `requirements_qwen_nuevo.txt` en la raíz del repo.
- RTMW (`rtmlib`) necesita `onnxruntime-gpu` SOLO (no instalar `onnxruntime` a la vez: coexistir con el paquete CPU hace que caiga silenciosamente a `CPUExecutionProvider` aunque se pida `--device cuda`). Verificar con `onnxruntime.get_available_providers()` que aparece `CUDAExecutionProvider`.
- El scheduler asume la política: preferir siempre la GPU A30 (índice 2 en este servidor), si no, la V100 con más memoria libre, luego la de menor utilización. Los índices de GPU no tienen por qué coincidir en el otro servidor — revisar antes de reusar `jobs_queue.json` tal cual.
