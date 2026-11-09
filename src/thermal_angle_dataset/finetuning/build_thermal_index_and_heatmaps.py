"""
Construye ts_index_cache_{split}.json para el dataset térmico de forma analítica
(sin escanear/leer los 110k .pt), y copia los heatmaps GT ya calculados en los
experimentos RGB del TFM en vez de recalcularlos con HRNet.

Por qué es seguro reusar los heatmaps RGB: el heatmap GT es la pose 2D de la
persona vista por una cámara concreta en un timestamp concreto. Esa pose física
es la misma independientemente de si la captura de esa cámara/timestamp se lee
como imagen RGB o térmica (mismo rig, mismo instante) - solo cambia la
modalidad de imagen, no la pose. Por tanto el heatmap ya calculado para
RGB en (timestamp, cámara_destino) es válido tal cual para la muestra térmica
con el mismo (timestamp, cámara_destino).

Por qué NO se puede reusar por abs_idx: la permutación aleatoria (np.random.seed(42)
sobre np.random.permutation(len(samples))) depende de len(samples), que difiere
entre el dataset RGB y el térmico (distinto nº de pares válidos) -> mismo seed,
pero permutación totalmente distinta. Verificado empíricamente: batch_00000.pt
de RGB y de térmicas tienen timestamps distintos.

En su lugar:
1. Reconstruye el índice térmico timestamp -> [(abs_idx, tgt_camera_id)] de forma
   analítica, reutilizando MultiViewThermalDataset de precompute_latents_thermal.py
   (mismo código, misma seed -> mismo resultado que el precómputo real, en
   segundos, sin tocar disco salvo para listar nombres de archivo).
2. Guarda ts_index_cache_{split}.json en el formato que espera dataset.py.
3. Para los timestamps del curriculum (experiment_json), busca en el índice RGB
   ya existente (ts_index_cache_{split}.json de dataset_local_latents_512_idx)
   los .pt cuyo (timestamp, cámara_destino) coincide, y lee su target_heatmaps.

NO se escribe nada en los .pt térmicos: los ~1700 heatmaps relevantes se vuelcan
en un único archivo "heatmap_lookup.pt" (dict {(timestamp, tgt_cam): tensor})
dentro de latents_dir. dataset.py (en esta misma carpeta) lo carga una vez y lo
usa en __getitem__ para rellenar "target_heatmaps" cuando el .pt térmico no lo
trae, sin tocar ni duplicar los 110k .pt originales.
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict

import torch
from tqdm import tqdm
from accelerate import PartialState

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dataset import load_experiment_json
from precompute_latents_thermal import MultiViewThermalDataset

_ = PartialState()  # requerido por accelerate.logging dentro de MultiViewThermalDataset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def camera_id_from_path(path: str) -> str:
    """'.../00_16/00_16_168...png' -> '00_16'"""
    return Path(path).parent.name


def _add_samples(ds, split_dir: Path, ts_index, tgt_lookup):
    for sample in ds.samples:
        ts = int(sample["timestamp"])
        pt_path = str(split_dir / f"batch_{sample['absolute_idx']:05d}.pt")
        ts_index[ts].append(pt_path)
        tgt_cam = camera_id_from_path(sample["tgt_path"])
        tgt_lookup[(ts, tgt_cam)].append(pt_path)


def build_analytic_index_train_val(dataset_root: str, latents_dir: Path):
    """train/ y val/ comparten índice combinado (dataset.py escanea ambas carpetas
    para resolver tanto 'train' como 'val', ya que distintos pares ordenados de un
    mismo timestamp pueden caer en lados distintos del split). Se reconstruyen las
    dos (train y val) y se fusionan en un único índice, igual que haría un escaneo
    real de ambas carpetas."""
    ts_index = defaultdict(list)
    tgt_lookup = defaultdict(list)

    ds_train = MultiViewThermalDataset(dataset_root=dataset_root, split="train")
    _add_samples(ds_train, latents_dir / "train", ts_index, tgt_lookup)

    ds_val = MultiViewThermalDataset(dataset_root=dataset_root, split="val")
    _add_samples(ds_val, latents_dir / "val", ts_index, tgt_lookup)

    return ts_index, tgt_lookup


def build_analytic_index_test(dataset_root: str, latents_dir: Path):
    ts_index = defaultdict(list)
    tgt_lookup = defaultdict(list)
    ds_test = MultiViewThermalDataset(dataset_root=dataset_root, split="test")
    _add_samples(ds_test, latents_dir / "test", ts_index, tgt_lookup)
    return ts_index, tgt_lookup


def save_ts_index_cache(ts_index, latents_dir: Path, split: str):
    cache_path = latents_dir / f"ts_index_cache_{split}.json"
    with open(cache_path, "w") as f:
        json.dump({str(k): v for k, v in ts_index.items()}, f)
    logger.info(f"[{split}] ts_index_cache guardado: {cache_path} ({len(ts_index)} timestamps)")


def load_rgb_heatmap_lookup(rgb_latents_dir: Path, split: str, timestamps: set):
    """Devuelve {(timestamp, tgt_cam): tensor_heatmap} leyendo SOLO los .pt de RGB
    cuyo timestamp está en el curriculum (vía el ts_index_cache ya existente de RGB)."""
    cache_path = rgb_latents_dir / f"ts_index_cache_{split}.json"
    with open(cache_path) as f:
        raw = json.load(f)

    relevant_files = []
    for ts_str, paths in raw.items():
        if int(ts_str) in timestamps:
            relevant_files.extend(paths)

    logger.info(f"[{split}] RGB: {len(relevant_files)} .pt relevantes de {len(raw)} timestamps indexados")

    lookup = {}
    missing_heatmap = 0
    missing_file = 0
    for p in tqdm(relevant_files, desc=f"RGB heatmaps [{split}]"):
        full_p = p if os.path.isabs(p) else str(rgb_latents_dir / p)
        try:
            data = torch.load(full_p, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            missing_file += 1
            continue
        ts = int(data["timestamp"])
        tgt_cam = camera_id_from_path(data["tgt_path"])
        if "target_heatmaps" not in data:
            missing_heatmap += 1
            continue
        lookup[(ts, tgt_cam)] = data["target_heatmaps"]
    if missing_heatmap:
        logger.warning(f"[{split}] {missing_heatmap} .pt RGB relevantes sin target_heatmaps")
    if missing_file:
        logger.warning(f"[{split}] {missing_file} .pt RGB relevantes no encontrados en disco")
    return lookup


def merge_into_lookup(tgt_lookup, rgb_heatmap_lookup, split: str, combined_lookup: dict):
    found, missing = 0, 0
    for (ts, tgt_cam) in tgt_lookup:
        heatmap = rgb_heatmap_lookup.get((ts, tgt_cam))
        if heatmap is None:
            missing += 1
            continue
        combined_lookup[(ts, tgt_cam)] = heatmap
        found += 1
    logger.info(f"[{split}] {found} pares (timestamp,cam) con heatmap | {missing} sin match en RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True,
                         help="Carpeta con train_val/ y test/ (imágenes térmicas fuente)")
    parser.add_argument("--latents_dir", type=str, required=True)
    parser.add_argument("--rgb_latents_dir", type=str, required=True)
    parser.add_argument("--experiment_json", type=str, required=True)
    args = parser.parse_args()

    latents_dir = Path(args.latents_dir)
    rgb_latents_dir = Path(args.rgb_latents_dir)

    train_ts, val_ts, test_ts, _ = load_experiment_json(args.experiment_json)
    curriculum_ts = {"train": set(train_ts), "val": set(val_ts), "test": set(test_ts)}

    combined_lookup = {}

    logger.info("\n=== train+val (índice combinado) ===")
    tv_ts_index, tv_tgt_lookup = build_analytic_index_train_val(args.dataset_root, latents_dir)
    save_ts_index_cache(tv_ts_index, latents_dir, "train")
    save_ts_index_cache(tv_ts_index, latents_dir, "val")

    for split in ["train", "val"]:
        rel_tgt_lookup = {k: v for k, v in tv_tgt_lookup.items() if k[0] in curriculum_ts[split]}
        logger.info(f"[{split}] {len(rel_tgt_lookup)} pares (timestamp,cam_destino) del curriculum")
        rgb_hm = load_rgb_heatmap_lookup(rgb_latents_dir, split, curriculum_ts[split])
        merge_into_lookup(rel_tgt_lookup, rgb_hm, split, combined_lookup)

    logger.info("\n=== test ===")
    test_ts_index, test_tgt_lookup = build_analytic_index_test(args.dataset_root, latents_dir)
    save_ts_index_cache(test_ts_index, latents_dir, "test")
    rel_tgt_lookup = {k: v for k, v in test_tgt_lookup.items() if k[0] in curriculum_ts["test"]}
    logger.info(f"[test] {len(rel_tgt_lookup)} pares (timestamp,cam_destino) del curriculum")
    rgb_hm = load_rgb_heatmap_lookup(rgb_latents_dir, "test", curriculum_ts["test"])
    merge_into_lookup(rel_tgt_lookup, rgb_hm, "test", combined_lookup)

    lookup_path = latents_dir / "heatmap_lookup.pt"
    torch.save(combined_lookup, lookup_path)
    logger.info(f"\nheatmap_lookup guardado: {lookup_path} ({len(combined_lookup)} entradas (timestamp,cam))")
    logger.info("Done.")


if __name__ == "__main__":
    main()
