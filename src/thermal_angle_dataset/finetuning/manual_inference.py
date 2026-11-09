"""
Inferencia MANUAL (fuera del bucle de entrenamiento) para inspeccionar un
checkpoint intermedio sin esperar al callback automático (--inference_every).

Pensado para correr en una GPU distinta a la que está entrenando (p.ej. una
V100 con más memoria libre que la A30 ocupada por finetune.py), cargando el
checkpoint LoRA ya guardado en disco.

Las imágenes se guardan DENTRO del mismo output_dir del experimento de
entrenamiento, en una carpeta "manual_inference_epoch{N:03d}" (prefijo
"manual_" a propósito para no confundirlas con "inference_epoch_XXX", que son
las que genera automáticamente el callback configurado por --inference_every).
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
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLQwenImage

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import TrainingConfig
from dataset import load_experiment_json, LatentsDatasetFromJSON
from finetune import build_model
from inference import run_inference_callback

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                         help="stage_XX_.../ que contiene qwen_lora_best.pt (o --checkpoint_file explícito)")
    parser.add_argument("--checkpoint_file", type=str, default=None,
                         help="Ruta explícita a un .pt (p.ej. checkpoints/xxx_final_epoch000.pt). Por defecto usa qwen_lora_best.pt del checkpoint_dir")
    parser.add_argument("--latents_dir", type=str, required=True)
    parser.add_argument("--experiment_json", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--transformer_model", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--num_steps", type=int, default=4)
    parser.add_argument("--epoch_label", type=int, required=True,
                         help="Época del checkpoint usado, solo para nombrar la carpeta/logs")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    config = TrainingConfig(
        latents_dir=args.latents_dir,
        output_dir=args.checkpoint_dir,
        base_model=args.base_model,
        transformer_model=args.transformer_model,
        experiment_json=args.experiment_json,
    )

    logger.info("Cargando modelo base + LoRA (vacía) ...")
    model = build_model(config, device)

    # IMPORTANTE: qwen_lora_best.pt es un archivo "mejor hasta ahora" que el
    # entrenamiento sigue sobreescribiendo mientras esta inferencia manual
    # carga el modelo base (varios minutos). Para no etiquetar mal las
    # imágenes, la época real se lee SIEMPRE del propio checkpoint cargado
    # (campo "epoch"/"global_epoch"), ignorando --epoch_label si no coincide.
    ckpt_path = args.checkpoint_file or os.path.join(args.checkpoint_dir, "qwen_lora_best.pt")
    logger.info(f"Cargando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    lora_state = ckpt["lora_state_dict"] if "lora_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(lora_state, strict=False)
    logger.info(f"{len(lora_state)-len(unexpected)}/{len(lora_state)} pesos LoRA cargados")

    actual_epoch = ckpt.get("global_epoch", ckpt.get("epoch"))
    if actual_epoch is not None and actual_epoch != args.epoch_label:
        logger.warning(
            f"--epoch_label pedía época {args.epoch_label} pero el checkpoint "
            f"cargado es de la época real {actual_epoch} (probable sobreescritura "
            f"de qwen_lora_best.pt mientras cargaba el modelo). Se usa {actual_epoch} "
            f"para nombrar la carpeta de salida."
        )
        args.epoch_label = actual_epoch

    model.eval()

    logger.info("Cargando VAE y scheduler...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model, subfolder="scheduler"
    )

    logger.info(f"Cargando dataset split={args.split} ...")
    train_ts, val_ts, test_ts, _ = load_experiment_json(args.experiment_json)
    ts_by_split = {"train": train_ts, "val": val_ts, "test": test_ts}
    dataset = LatentsDatasetFromJSON(args.latents_dir, ts_by_split[args.split], split=args.split)

    save_dir = os.path.join(args.checkpoint_dir, f"manual_inference_epoch{args.epoch_label:03d}_{args.split}")
    logger.info(f"Guardando en: {save_dir}  (NOTA: inferencia MANUAL, no es el callback automático de --inference_every)")

    run_inference_callback(
        model=model, vae=vae, base_scheduler=scheduler,
        device=device, dataset=dataset,
        save_dir=save_dir,
        epoch=args.epoch_label,
        img_height=512, img_width=512,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        dtype=dtype,
    )

    with open(os.path.join(save_dir, "README_MANUAL.txt"), "w") as f:
        f.write(
            f"Imagenes generadas con inferencia MANUAL (manual_inference.py), "
            f"NO por el callback automático configurado via --inference_every.\n"
            f"Checkpoint usado: epoch {args.epoch_label} "
            f"({'explicito: ' + args.checkpoint_file if args.checkpoint_file else 'qwen_lora_best.pt'})\n"
            f"Split: {args.split} | num_samples: {args.num_samples} | num_steps: {args.num_steps}\n"
        )

    logger.info("Done.")


if __name__ == "__main__":
    main()
