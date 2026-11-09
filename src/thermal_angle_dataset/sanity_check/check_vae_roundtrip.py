"""
Decodifica una muestra aleatoria de varios .pt precomputados con el VAE y compara
contra la imagen original (src y tgt) para confirmar que los latentes codifican
correctamente la imagen térmica (sin pérdida de información relevante).
Guarda imágenes side-by-side (original vs reconstruida) para inspección visual.
"""
import os, sys, random
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/mnt/nas_dataset/antonio/.qwen/hf_cache"
os.environ["TMPDIR"] = "/mnt/nas_dataset/antonio/.qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

import torch
import numpy as np
from PIL import Image
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from qwenimage.pipeline_qwenimage_edit_plus import QwenImageEditPlusPipeline

MODEL_PATH = "/mnt/nas_dataset/antonio/.qwen/qwen_model_real"
LATENTS_DIR = Path("/mnt/nas_dataset/antonio/qwen/datasets/dataset_thermal_latents_512")
OUT_DIR = Path(os.path.dirname(__file__)) / "outputs"
OUT_DIR.mkdir(exist_ok=True)

random.seed(123)

def pick_samples(split, n):
    files = sorted((LATENTS_DIR / split).glob("batch_*.pt"))
    return random.sample(files, n)

def unpack_and_denormalize(pipeline, packed, resolution, device, dtype):
    """
    Revierte _pack_latents (-> 5D: B,C,1,H,W) y revierte la normalización
    aplicada en _encode_vae_image: (x - mean) / std  =>  x = decoded*std + mean.
    """
    latents_5d = pipeline._unpack_latents(packed, resolution, resolution, pipeline.vae_scale_factor)
    latents_5d = latents_5d.to(device=device, dtype=pipeline.vae.dtype)

    z_dim = pipeline.vae.config.z_dim
    latents_mean = torch.tensor(pipeline.vae.config.latents_mean).view(1, z_dim, 1, 1, 1).to(device, pipeline.vae.dtype)
    latents_std = torch.tensor(pipeline.vae.config.latents_std).view(1, z_dim, 1, 1, 1).to(device, pipeline.vae.dtype)

    latents_5d = latents_5d * latents_std + latents_mean
    return latents_5d

def main():
    dtype = torch.bfloat16
    device = "cuda:0"
    print("Cargando pipeline (solo VAE)...")
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_PATH, torch_dtype=dtype, local_files_only=True,
    )
    del pipeline.transformer
    pipeline.transformer = None
    torch.cuda.empty_cache()
    pipeline.vae.to(device)
    pipeline.vae.eval()

    samples = []
    for split, n in [("train", 3), ("val", 2), ("test", 2)]:
        samples += [(split, f) for f in pick_samples(split, n)]

    for split, f in samples:
        data = torch.load(f, map_location="cpu", weights_only=False)
        resolution = data["resolution"]
        for key, orig_path_key in [("source_latents_packed", "src_path"), ("target_latents_packed", "tgt_path")]:
            packed = data[key].to(device=device, dtype=dtype)
            with torch.no_grad():
                lat_5d = unpack_and_denormalize(pipeline, packed, resolution, device, dtype)
                decoded = pipeline.vae.decode(lat_5d, return_dict=False)[0]
            decoded = decoded.squeeze(2).squeeze(0).float().cpu()
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded_np = (decoded.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            recon_img = Image.fromarray(decoded_np)

            orig_path = data[orig_path_key]
            orig_img = Image.open(orig_path).convert("RGB").resize((resolution, resolution), Image.BILINEAR)

            combo = Image.new("RGB", (resolution * 2 + 10, resolution), "white")
            combo.paste(orig_img, (0, 0))
            combo.paste(recon_img, (resolution + 10, 0))
            ts = data["timestamp"]
            out_path = OUT_DIR / f"{split}_{f.stem}_{orig_path_key}_{ts}.png"
            combo.save(out_path)

            orig_arr = np.array(orig_img).astype(np.float32)
            recon_arr = decoded_np.astype(np.float32)
            mse = float(((orig_arr - recon_arr) ** 2).mean())
            print(f"[{split}] {f.name} {orig_path_key} ts={ts} MSE={mse:.2f} -> {out_path}")

    print("Done.")

if __name__ == "__main__":
    main()
