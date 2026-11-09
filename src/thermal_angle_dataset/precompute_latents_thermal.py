"""
Precómputo de latentes + embeddings para el dataset TÉRMICO (Qwen-Image-Edit),
análogo al pipeline usado para el dataset RGB (precompute_latents_fix_batch_BULL_timer_MSSE.py)
pero leyendo las imágenes *_thermal.png en lugar de *_rgb.png.

Vive en una carpeta propia (src/thermal_angle_dataset/) para no mezclarse con los
experimentos RGB ya existentes.

Diferencias respecto al script RGB original:
  - Glob de imágenes: "*_thermal.png" en vez de "*_rgb.png".
  - Rutas de HF_HOME/TMPDIR/modelo base adaptadas a este host (/mnt/nas_dataset/antonio/.qwen/...).
  - --split acepta "train" | "val" | "test" de forma independiente (en vez de procesar
    siempre train+val en la misma llamada), para poder repartir el trabajo entre varias
    GPUs mediante --start_idx/--end_idx sin que los rangos de un split interfieran con otro.
  - Los campos "timestamp", "src_path" y "tgt_path" se guardan directamente en el .pt
    (en el pipeline RGB esto se hacía con un script de parcheo a posteriori).
"""

import os
import sys

os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/mnt/nas_dataset/antonio/.qwen/hf_cache"
os.environ["TMPDIR"] = "/mnt/nas_dataset/antonio/.qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

import argparse
import torch
import csv
import time
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from dataclasses import dataclass
from PIL import Image
from pathlib import Path
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

logger = get_logger(__name__)


@dataclass
class PrecomputeConfig:
    dataset_root: str
    output_dir: str
    base_model: str = "/mnt/nas_dataset/antonio/.qwen/qwen_model_real"
    resolution: int = 512
    condition_resolution: int = 384
    batch_size: int = 1
    start_idx: int = 0
    end_idx: int = -1
    mixed_precision: str = "bf16"
    seed: int = 42
    dataloader_num_workers: int = 0


class MultiViewThermalDataset(Dataset):
    """
    Dataset de imágenes térmicas multi-vista sincronizadas por timestamp.
    Genera pares de (imagen_origen, imagen_destino, prompt_rotación).
    Misma lógica de construcción de pares que el dataset RGB, pero
    leyendo "*_thermal.png" en lugar de "*_rgb.png".
    """

    CAMERA_ANGLES = {
        "00_17": 0,
        "00_16": 90,
        "00_15": -90,
    }

    ANGLE_PROMPTS = {
        0: {
            90:   "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            180:  "将镜头旋转180度 Rotate the camera 180 degrees.",
            -90:  "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
        },
        90: {
            90:   "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            -90:  "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
            -180: "将镜头旋转180度 Rotate the camera 180 degrees.",
        },
        -90: {
            90:   "将镜头向右旋转90度 Rotate the camera 90 degrees to the right.",
            -90:  "将镜头向左旋转90度 Rotate the camera 90 degrees to the left.",
            180:  "将镜头旋转180度 Rotate the camera 180 degrees.",
        },
    }

    def __init__(
        self,
        dataset_root: str,
        resolution: int = 512,
        split: str = "train",
        train_ratio: float = 0.9,
        start_idx: int = 0,
        end_idx: int = -1,
    ):
        self.dataset_root = Path(dataset_root)
        self.resolution = resolution
        self.split = split

        self.samples = self._build_sample_pairs()

        np.random.seed(42)
        indices = np.random.permutation(len(self.samples))
        split_idx = int(len(indices) * train_ratio)

        if split == "train":
            self.samples = [self.samples[i] for i in indices[:split_idx]]
        elif split == "val":
            self.samples = [self.samples[i] for i in indices[split_idx:]]
        # si split == "test", no se hace nada → usa todos los samples

        # Tatuar índice absoluto (estable, ligado al split) y cortar por rango
        for i, sample in enumerate(self.samples):
            sample["absolute_idx"] = i

        total_samples = len(self.samples)
        end = end_idx if end_idx != -1 else total_samples
        self.samples = self.samples[start_idx:end]

        logger.info(f"Loaded {len(self.samples)} samples for {split} split (range {start_idx}:{end} of {total_samples})")

    def _build_sample_pairs(self) -> List[Dict]:
        samples = []
        timestamp_data = {}

        subdir = "test" if self.split == "test" else "train_val"
        base_dir = self.dataset_root / subdir
        if not base_dir.exists():
            logger.warning(f"No {subdir} folder at {self.dataset_root}, using root")
            base_dir = self.dataset_root

        for camera_dir in sorted(base_dir.glob("*")):
            if not camera_dir.is_dir():
                continue
            camera_id = camera_dir.name
            if camera_id not in self.CAMERA_ANGLES:
                continue
            images = sorted(camera_dir.glob("*_thermal.png"))
            if not images:
                images = sorted(camera_dir.glob("*_thermal*"))
            for img_path in images:
                parts = img_path.stem.split("_")
                if len(parts) >= 3:
                    timestamp = parts[2]
                    if timestamp not in timestamp_data:
                        timestamp_data[timestamp] = {}
                    timestamp_data[timestamp][camera_id] = img_path

        for timestamp, cameras in timestamp_data.items():
            for src_cam, src_angle in self.CAMERA_ANGLES.items():
                if src_cam not in cameras:
                    continue
                src_img = cameras[src_cam]
                for tgt_cam, tgt_angle in self.CAMERA_ANGLES.items():
                    if src_cam == tgt_cam or tgt_cam not in cameras:
                        continue
                    tgt_img = cameras[tgt_cam]
                    angle_diff = (tgt_angle - src_angle) % 360
                    if angle_diff > 180:
                        angle_diff -= 360
                    if src_angle in self.ANGLE_PROMPTS and angle_diff in self.ANGLE_PROMPTS[src_angle]:
                        prompt = self.ANGLE_PROMPTS[src_angle][angle_diff]
                    else:
                        prompt = f"将镜头旋转{angle_diff}度 Rotate the camera {angle_diff} degrees."
                    samples.append({
                        "src_path": str(src_img),
                        "tgt_path": str(tgt_img),
                        "src_angle": src_angle,
                        "tgt_angle": tgt_angle,
                        "angle_diff": angle_diff,
                        "prompt": prompt,
                        "timestamp": timestamp,
                    })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        # Las térmicas están en modo "L" (grises); convert("RGB") las replica en 3 canales
        # para que encajen con el VAE/text-encoder sin tocar su arquitectura.
        src_pil = Image.open(sample["src_path"]).convert("RGB")
        tgt_pil = Image.open(sample["tgt_path"]).convert("RGB")
        return {
            "source_pil": src_pil,
            "target_pil": tgt_pil,
            "prompt": sample["prompt"],
            "angle_diff": sample["angle_diff"],
            "absolute_idx": sample["absolute_idx"],
            "timestamp": sample["timestamp"],
            "src_path": sample["src_path"],
            "tgt_path": sample["tgt_path"],
        }


def collate_fn(examples):
    """Collate function: imágenes PIL se pasan como listas (no se pueden stack)."""
    return {
        "source_pils": [ex["source_pil"] for ex in examples],
        "target_pils": [ex["target_pil"] for ex in examples],
        "prompts":     [ex["prompt"] for ex in examples],
        "angle_diffs": torch.tensor([ex["angle_diff"] for ex in examples]),
        "absolute_idxs": [ex["absolute_idx"] for ex in examples],
        "timestamps":  [ex["timestamp"] for ex in examples],
        "src_paths":   [ex["src_path"] for ex in examples],
        "tgt_paths":   [ex["tgt_path"] for ex in examples],
    }


def pil_to_vae_tensor(pil_img: Image.Image, resolution: int, device, dtype) -> torch.Tensor:
    """
    Convierte una imagen PIL a tensor listo para el VAE.
    Salida: (1, C, 1, H, W) en rango [-1, 1], con la dimensión de frame requerida por
    AutoencoderKLQwenImage (que opera sobre video 5D).
    """
    transform = T.Compose([
        T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ])
    tensor = transform(pil_img).unsqueeze(0)
    tensor = tensor.unsqueeze(2)
    return tensor.to(device=device, dtype=dtype)


def pil_to_condition_pil(pil_img: Image.Image, condition_resolution: int) -> Image.Image:
    """
    Redimensiona a la resolución que usa el text encoder (384x384 por defecto).
    Se devuelve PIL porque Qwen2VLProcessor espera PIL/array en [0,255], NO tensores.
    """
    return pil_img.resize((condition_resolution, condition_resolution), Image.BILINEAR)


@torch.no_grad()
def precompute_data(
    pipeline: QwenImageEditPlusPipeline,
    dataloader: DataLoader,
    accelerator: Accelerator,
    config: PrecomputeConfig,
    split: str = "train",
):
    """Pre-computa y guarda en disco latentes + embeddings para un split."""
    split_dir = os.path.join(config.output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    csv_path = os.path.join(config.output_dir, f"timings_{split}.csv")
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if os.path.getsize(csv_path) == 0:
        csv_writer.writerow(["abs_idx", "time_compute_s", "time_total_s"])

    pipeline.vae.eval()
    pipeline.text_encoder.eval()

    device = accelerator.device
    store_dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16
    vae_dtype = next(pipeline.vae.parameters()).dtype

    logger.info(f"Processing {split} split...")

    total_samples = len(dataloader.dataset)
    progress_bar = tqdm(total=total_samples, disable=not accelerator.is_local_main_process)

    data_iter = iter(dataloader)
    while True:
        try:
            batch = next(data_iter)
        except StopIteration:
            break
        except Exception as e:
            # Imagen corrupta/ilegible (p.ej. PIL.UnidentifiedImageError): se salta
            # esa muestra en vez de abortar todo el proceso.
            logger.warning(f"Skipping unreadable sample: {e}")
            progress_bar.update(1)
            continue

        source_pils = batch["source_pils"]
        target_pils = batch["target_pils"]
        prompts     = batch["prompts"]
        angle_diffs = batch["angle_diffs"]
        abs_idx     = batch["absolute_idxs"][0]  # batch_size=1
        timestamp   = batch["timestamps"][0]
        src_path    = batch["src_paths"][0]
        tgt_path    = batch["tgt_paths"][0]

        save_path = os.path.join(split_dir, f"batch_{abs_idx:05d}.pt")
        if os.path.exists(save_path):
            progress_bar.update(1)
            continue

        t_start = time.perf_counter()
        # ── 1. TARGET LATENTS ────────────────────────────────────────────────
        target_latents_list = []
        for pil in target_pils:
            t_vae = pil_to_vae_tensor(pil, config.resolution, device, dtype=vae_dtype)
            enc = pipeline._encode_vae_image(t_vae, generator=None)
            target_latents_list.append(enc)

        target_latents_5d = torch.cat(target_latents_list, dim=0)
        target_latents_4d = target_latents_5d.squeeze(2)
        B, C, H_lat, W_lat = target_latents_4d.shape
        target_latents_packed = pipeline._pack_latents(
            target_latents_4d, B, C, H_lat, W_lat
        )

        # ── 2. SOURCE LATENTS ────────────────────────────────────────────────
        source_latents_list = []
        for pil in source_pils:
            s_vae = pil_to_vae_tensor(pil, config.resolution, device, dtype=vae_dtype)
            enc = pipeline._encode_vae_image(s_vae, generator=None)
            source_latents_list.append(enc)

        source_latents_5d = torch.cat(source_latents_list, dim=0)
        source_latents_4d = source_latents_5d.squeeze(2)
        source_latents_packed = pipeline._pack_latents(
            source_latents_4d, B, C, H_lat, W_lat
        )

        # ── 3. TEXT + VISION EMBEDDINGS ──────────────────────────────────────
        condition_pils = [pil_to_condition_pil(p, config.condition_resolution) for p in source_pils]

        prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
            prompt=prompts,
            image=condition_pils,
            device=device,
            num_images_per_prompt=1,
        )
        t_after_compute = time.perf_counter()
        # ── 4. GUARDAR ────────────────────────────────────────────────────────
        data_to_save = {
            "target_latents_packed":  target_latents_packed.to(dtype=store_dtype).cpu(),
            "source_latents_packed":  source_latents_packed.to(dtype=store_dtype).cpu(),
            "prompt_embeds":          prompt_embeds.to(dtype=store_dtype).cpu(),
            "prompt_embeds_mask":     prompt_embeds_mask.cpu(),
            "angle_diffs":            angle_diffs.cpu(),
            "resolution":             config.resolution,
            "timestamp":              timestamp,
            "src_path":               src_path,
            "tgt_path":               tgt_path,
        }

        torch.save(data_to_save, save_path)
        t_after_save = time.perf_counter()
        csv_writer.writerow([
            abs_idx,
            round(t_after_compute - t_start, 4),
            round(t_after_save   - t_start, 4),
        ])
        csv_file.flush()
        progress_bar.update(1)

    progress_bar.close()
    csv_file.close()
    logger.info(f"Finished processing {split} split.")


def main():
    parser = argparse.ArgumentParser(description="Precompute thermal latents for Qwen-Image-Edit")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--split",        type=str, required=True, choices=["train", "val", "test"],
                         help="Qué split procesar en esta invocación (permite repartir cada split entre varias GPUs de forma independiente)")
    parser.add_argument("--batch_size",   type=int, default=1)
    parser.add_argument("--start_idx",    type=int, default=0)
    parser.add_argument("--end_idx",      type=int, default=-1)
    parser.add_argument("--resolution",   type=int, default=512)
    parser.add_argument("--condition_resolution", type=int, default=384)
    args = parser.parse_args()

    config = PrecomputeConfig(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        resolution=args.resolution,
        condition_resolution=args.condition_resolution,
    )

    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    set_seed(config.seed)

    logger.info("Loading pipeline model...")
    dtype = torch.bfloat16 if config.mixed_precision == "bf16" else torch.float16

    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        config.base_model,
        torch_dtype=dtype,
        local_files_only=True,
    )

    logger.info("Deleting transformer to save memory...")
    del pipeline.transformer
    pipeline.transformer = None
    torch.cuda.empty_cache()

    pipeline.vae.to(accelerator.device)
    pipeline.text_encoder.to(accelerator.device)

    logger.info(f"Loading dataset (split={args.split})...")
    dataset = MultiViewThermalDataset(
        dataset_root=config.dataset_root,
        resolution=config.resolution,
        split=args.split,
        start_idx=config.start_idx,
        end_idx=config.end_idx,
    )

    if len(dataset) == 0:
        logger.info(f"No samples to process for split={args.split} in range {args.start_idx}:{args.end_idx}")
        return

    batch_size = config.batch_size if args.split == "train" else 1
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.dataloader_num_workers,
        collate_fn=collate_fn,
    )
    precompute_data(pipeline, dataloader, accelerator, config, split=args.split)

    logger.info("Done!")


if __name__ == "__main__":
    main()
