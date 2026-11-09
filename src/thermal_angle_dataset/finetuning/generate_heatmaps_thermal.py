"""
Genera heatmaps GT de HRNet para el subconjunto térmico definido por el JSON de
curriculum (dataset_splits_thermal.json, mismos timestamps que el experimento
RGB "dataset_splits_uniformes_rareza_ususarios.json"), y los añade a los .pt
precomputados bajo la clave "target_heatmaps" (formato esperado por
dataset.py::make_collate_latents / loss.py::CombinedLossFn).

A diferencia del script RGB original (src/hr_net/add_heatmaps_to_latents.py),
que reconstruía el orden del dataset por índice de batch para encontrar la
imagen target de cada .pt, aquí no hace falta: nuestros .pt térmicos ya
guardan "timestamp"/"src_path"/"tgt_path" directamente (ver
precompute_latents_thermal.py), así que basta con leerlos.

Reutiliza dataset.py (load_experiment_json + LatentsDatasetFromJSON) para
resolver timestamp -> lista de .pt, lo que además construye/cachea
ts_index_cache_{split}.json en latents_dir (necesario para el training real).
"""
import os
import sys
import argparse
import logging

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/mnt/nas_dataset/antonio/.qwen/hf_cache"
os.environ["TMPDIR"] = "/mnt/nas_dataset/antonio/.qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import load_experiment_json, LatentsDatasetFromJSON
from hrnet import load_hrnet_model

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

HRNET_INPUT_SIZE = (288, 384)  # (H, W), igual que en latent_utils/config


def image_to_hrnet_input(pil_img: Image.Image, device) -> torch.Tensor:
    """Replica preprocess_image_for_hrnet pero partiendo de una imagen real en
    disco (ya en [0,1]) en vez de un latente decodificado en [-1,1]."""
    img = pil_img.convert("RGB")
    tensor = torch.from_numpy(
        __import__("numpy").array(img)
    ).float().permute(2, 0, 1).unsqueeze(0) / 255.0  # (1,3,H,W) en [0,1]
    tensor = tensor.to(device)
    tensor = F.interpolate(tensor, size=HRNET_INPUT_SIZE, mode="bilinear", align_corners=True)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


@torch.no_grad()
def process_split(files, hrnet, device, force=False):
    processed, skipped, errors = 0, 0, 0
    for pt_path in tqdm(files, desc="heatmaps"):
        try:
            data = torch.load(pt_path, map_location="cpu", weights_only=False)
            if not force and "target_heatmaps" in data:
                skipped += 1
                continue

            tgt_path = data["tgt_path"]
            img = Image.open(tgt_path)
            hrnet_input = image_to_hrnet_input(img, device)
            heatmap = hrnet(hrnet_input)  # (1, 27, 72, 96)

            data["target_heatmaps"] = heatmap.squeeze(0).float().cpu()
            torch.save(data, pt_path)
            processed += 1
        except Exception as e:
            logger.error(f"Error en {pt_path}: {e}")
            errors += 1
    return processed, skipped, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_dir", type=str, required=True)
    parser.add_argument("--experiment_json", type=str, required=True)
    parser.add_argument("--hrnet_model_path", type=str, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Cargando HRNet desde {args.hrnet_model_path}...")
    hrnet = load_hrnet_model(args.hrnet_model_path, width=48, num_joints=27, device=device)
    hrnet.requires_grad_(False)
    hrnet.eval()

    train_ts, val_ts, test_ts, _ = load_experiment_json(args.experiment_json)

    for split, ts_list in [("train", train_ts), ("val", val_ts), ("test", test_ts)]:
        logger.info(f"\n=== Split {split}: {len(ts_list)} timestamps ===")
        ds = LatentsDatasetFromJSON(args.latents_dir, ts_list, split=split)
        logger.info(f"[{split}] {len(ds.files)} .pt a procesar")
        processed, skipped, errors = process_split(ds.files, hrnet, device, force=args.force)
        logger.info(f"[{split}] procesados={processed} saltados={skipped} errores={errors}")

    logger.info("Done.")


if __name__ == "__main__":
    main()
