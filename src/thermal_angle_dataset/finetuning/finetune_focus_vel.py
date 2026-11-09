# train.py
import os
import sys
import csv
import argparse
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(current_dir)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["HF_HOME"] = "/mnt/nas_dataset/antonio/.qwen/hf_cache"
os.environ["TMPDIR"] = "/mnt/nas_dataset/antonio/.qwen/tmp"
os.environ["PYTHONNOUSERSITE"] = "1"

import torch
from tqdm import tqdm
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.models import AutoencoderKLQwenImage
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import json
from dataclasses import asdict

from config import TrainingConfig
from dataset import build_dataloaders, build_val_inference_subset
from inference import run_inference_callback
from loss_vel_focus import CombinedLossFn
from model import QwenSingleGPUWrapper
from validate import validate


logging.basicConfig(level=logging.INFO, force=True, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents_dir",       type=str, required=True)
    parser.add_argument("--synth_latents_dir", type=str, default=None)
    parser.add_argument("--experiment_json",   type=str, nargs="+", required=True)
    parser.add_argument("--output_dir",        type=str, default="../models/finetuned_pp")
    parser.add_argument("--base_model",        type=str, required=True)
    parser.add_argument("--transformer_model", type=str, required=True)
    parser.add_argument("--reset_patience", action="store_true")

    parser.add_argument("--epochs",            type=int,   default=3)
    parser.add_argument("--batch_size",        type=int,   default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate",     type=float, default=1e-4)
    parser.add_argument("--lora_rank",         type=int,   default=16)
    parser.add_argument("--lora_alpha",        type=int,   default=32)
    parser.add_argument("--lora_dropout",      type=float, default=0.1)
    parser.add_argument("--inference_every",   type=int,   default=2)
    parser.add_argument("--inference_steps",   type=int,   default=4)
    parser.add_argument("--inference_samples", type=int,   default=5)
    parser.add_argument("--val_inference_per_user", type=int, default=2)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pose_weight_scale", type=float, default=4.0)
    parser.add_argument("--use_t_scaling",     action="store_true")
    return parser.parse_args()


def make_config(args, experiment_json: str, stage_output_dir: str) -> TrainingConfig:
    return TrainingConfig(
        latents_dir=args.latents_dir,
        synth_latents_dir=args.synth_latents_dir,
        experiment_json=experiment_json,
        output_dir=stage_output_dir,
        base_model=args.base_model,
        transformer_model=args.transformer_model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.learning_rate,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        inference_every=args.inference_every,
        inference_steps=args.inference_steps,
        inference_samples=args.inference_samples,
        patience=args.patience,
        reset_patience=args.reset_patience,
        pose_weight_scale=args.pose_weight_scale,
        use_t_scaling=args.use_t_scaling,
    )


# ---------------------------------------------------------------------------
# Setup de modelo, checkpoints
# ---------------------------------------------------------------------------

def build_model(config: TrainingConfig, device: torch.device) -> QwenSingleGPUWrapper:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    logger.info(f"Cargando modelo desde {config.transformer_model} → {device}")
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.transformer_model,
        subfolder=None,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map=str(device),
        attn_implementation="sdpa",
        local_files_only=True,
    )
    transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        init_lora_weights=True,
        target_modules=["to_q", "to_k", "to_v", "to_out.0",
                        "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"],
        lora_dropout=config.lora_dropout,
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()
    logger.info("Creando QwenSingleGPUWrapper...")
    return QwenSingleGPUWrapper(transformer)


def load_checkpoint(model, config, device, optimizer=None, diff_scheduler=None):
    checkpoint_path = os.path.join(config.output_dir, "qwen_lora_best.pt")
    if not os.path.exists(checkpoint_path):
        logger.info("No se encontró checkpoint previo. Iniciando desde cero.")
        return 0, float('inf'), 0
    try:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "lora_state_dict" in ckpt:
            lora_state  = ckpt["lora_state_dict"]
            start_epoch = ckpt.get("epoch", -1) + 1
            best_loss   = ckpt.get("best_loss", float('inf'))
            patience_counter = ckpt.get("patience_counter", 0)
            if optimizer is not None and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if diff_scheduler is not None and "scheduler_state_dict" in ckpt and ckpt["scheduler_state_dict"] is not None:
                diff_scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            logger.info(f"✓ Reanudando desde época local {start_epoch} | best_loss={best_loss:.6f}")
        else:
            lora_state, start_epoch, best_loss, patience_counter = ckpt, 0, float('inf'), 0
        missing, unexpected = model.load_state_dict(lora_state, strict=False)
        logger.info(f"{len(lora_state)-len(unexpected)}/{len(lora_state)} pesos LoRA cargados")
        return start_epoch, best_loss, patience_counter
    except Exception as e:
        logger.error(f"Error cargando checkpoint: {e}. Iniciando desde cero.")
        return 0, float('inf'), 0


def save_checkpoint(model, config, global_epoch, local_epoch, best_loss, optimizer, diff_scheduler, patience_counter=0, extra_name=None):
    local_lora = {k: v.cpu() for k, v in model.state_dict().items() if "lora" in k}
    ckpt_data  = {
        "lora_state_dict":      local_lora,
        "epoch":                local_epoch,
        "global_epoch":         global_epoch,
        "best_loss":            best_loss,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": diff_scheduler.state_dict() if hasattr(diff_scheduler, 'state_dict') else None,
        "learning_rate":        config.lr,
        "patience_counter":     patience_counter,
    }
    save_path = os.path.join(config.output_dir, "qwen_lora_best.pt")
    torch.save(ckpt_data, save_path)
    logger.info(f"LoRA guardado en: {save_path} (época global {global_epoch}, val_loss={best_loss:.6f})")
    if extra_name:
        ckpt_dir = os.path.join(config.output_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(ckpt_data, os.path.join(ckpt_dir, f"{extra_name}_epoch{global_epoch:03d}.pt"))


# ---------------------------------------------------------------------------
# Función de entrenamiento por etapa
# ---------------------------------------------------------------------------

def train_stage(
    config: TrainingConfig,
    model: QwenSingleGPUWrapper,
    vae,
    device: torch.device,
    dtype: torch.dtype,
    stage_idx: int,
    optimizer: torch.optim.Optimizer,
    diff_scheduler,
    epoch_offset: int,
    patience: int = 5,
):
    exp_name = os.path.splitext(os.path.basename(config.experiment_json))[0]
    logger.info(f"\n{'='*60}\n  ETAPA {stage_idx}  |  {exp_name}\n{'='*60}")

    os.makedirs(config.output_dir, exist_ok=True)

    start_local_epoch, best_loss, patience_counter = load_checkpoint(model, config, device, optimizer, diff_scheduler)

    if config.reset_patience:
        patience_counter = 0
        best_loss = float('inf')

    train_dataset, val_dataset, test_dataset, \
    train_dataloader, val_dataloader, test_dataloader = build_dataloaders(config)

    first_sample   = torch.load(train_dataset.files[0], weights_only=True)
    img_resolution = first_sample.get("resolution", 1024)
    logger.info(f"Resolución: {img_resolution}")

    loss_fn = CombinedLossFn(
        img_height=img_resolution, img_width=img_resolution,
        pose_weight_scale=config.pose_weight_scale,
        use_t_scaling=config.use_t_scaling,
    )

    csv_file_path = os.path.join(config.output_dir, "training_metrics.csv")
    csv_mode = "a" if start_local_epoch > 0 else "w"
    with open(csv_file_path, mode=csv_mode, newline="") as f:
        if csv_mode == "w":
            csv.writer(f).writerow([
                "global_epoch", "local_epoch", "train_loss", "val_loss",
                "train_pck", "val_pck",
                "train_velocity_loss", "train_heatmap_loss",
                "val_velocity_loss", "val_heatmap_loss",
                "pose_weight_mean",
            ])

    config_path = os.path.join(config.output_dir, "experiment_config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(asdict(config), f, indent=2)

    model.train()

    accum_steps = max(1, config.gradient_accumulation_steps)
    num_batches = len(train_dataloader)

    local_epoch = start_local_epoch - 1
    for local_epoch in range(start_local_epoch, config.epochs):
        global_epoch = epoch_offset + local_epoch
        logger.info(f"[Etapa {stage_idx}] Global epoch {global_epoch} | Local epoch {local_epoch} start")

        iterator = tqdm(train_dataloader, desc=f"[Stage {stage_idx}] GEpoch {global_epoch}")
        avg_loss, avg_pck, avg_velocity_loss, avg_heatmap_loss, avg_pose_weight, steps = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        optimizer.zero_grad()
        for step, batch in enumerate(iterator):
            target = batch["target_latents_packed"].to(device, dtype=dtype)
            source = batch["source_latents_packed"].to(device, dtype=dtype)
            prompt = batch["prompt_embeds"].to(device, dtype=dtype)
            mask   = batch["prompt_embeds_mask"].to(device)
            target_heatmaps = batch["target_heatmaps"]

            bsz    = target.shape[0]
            g_seed = 42 + global_epoch * 10000 + step
            gen    = torch.Generator(device=device).manual_seed(g_seed)

            timesteps = torch.randint(
                0, diff_scheduler.config.num_train_timesteps,
                (bsz,), generator=gen, device=device
            ).long()
            noise = torch.randn(target.shape, generator=gen, device=device, dtype=dtype)

            t_norm = (timesteps.float() / diff_scheduler.config.num_train_timesteps).to(dtype).view(-1, 1, 1)
            noisy  = (1.0 - t_norm) * target + t_norm * noise

            velocity_target = noise - target
            timestep_norm   = (timesteps.float() / 1000.0).to(dtype)

            latent_model_input = torch.cat([noisy, source], dim=1)
            v_pred = model(latent_model_input, prompt, mask, timestep_norm)

            loss_fn.set_batch_context(target_heatmaps, timesteps)
            combined_target = torch.stack([velocity_target, noisy], dim=1)
            loss = loss_fn(v_pred, combined_target)

            (loss / accum_steps).backward()

            is_accum_boundary = ((step + 1) % accum_steps == 0) or ((step + 1) == num_batches)
            if is_accum_boundary:
                torch.cuda.reset_peak_memory_stats(device)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            avg_loss          += loss.item()
            avg_pck           += 0.0
            avg_heatmap_loss  += loss_fn.last_weighted_loss
            avg_velocity_loss += loss_fn.last_velocity_loss
            avg_pose_weight   += loss_fn.last_pose_weight_mean
            steps             += 1

        logger.info(f"[Etapa {stage_idx}] Ejecutando validación...")
        val_loss, val_pck, val_vel_loss, val_heat_loss = validate(
            model=model, val_dataloader=val_dataloader,
            loss_fn=loss_fn, diff_scheduler=diff_scheduler,
            device=device, dtype=dtype,
        )

        global_avg_loss = avg_loss / steps if steps > 0 else 0.0
        global_avg_pck  = avg_pck  / steps if steps > 0 else 0.0
        global_avg_pose_weight = avg_pose_weight / steps if steps > 0 else 0.0

        logger.info(
            f"[Stage {stage_idx}] GEpoch {global_epoch} | "
            f"Train Loss: {global_avg_loss:.6f} | Val Loss: {val_loss:.6f} | "
            f"pose_weight_mean: {global_avg_pose_weight:.3f} | Best: {best_loss:.6f}"
        )

        with open(csv_file_path, mode="a", newline="") as f:
            csv.writer(f).writerow([
                global_epoch, local_epoch, global_avg_loss, val_loss,
                global_avg_pck, val_pck,
                avg_velocity_loss / steps, avg_heatmap_loss / steps,
                val_vel_loss, val_heat_loss,
                global_avg_pose_weight,
            ])

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            logger.info("🟢 Nueva mejor val_loss. Guardando checkpoint...")
            save_checkpoint(model, config, global_epoch, local_epoch, best_loss,
                            optimizer, diff_scheduler, patience_counter, extra_name=exp_name)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"🔴 Early stopping en epoch global {global_epoch}.")
                break

        if config.inference_every > 0 and (global_epoch + 1) % config.inference_every == 0:
            infer_dir = os.path.join(config.output_dir, f"inference_epoch_{global_epoch:03d}")
            run_inference_callback(
                model=model, vae=vae, base_scheduler=diff_scheduler,
                device=device, dataset=test_dataset,
                save_dir=os.path.join(infer_dir, "test"),
                epoch=global_epoch,
                img_height=img_resolution, img_width=img_resolution,
                num_steps=config.inference_steps,
                num_samples=config.inference_samples,
                dtype=dtype,
            )

    logger.info(f"[Etapa {stage_idx}] Evaluando en test set...")
    test_loss, test_pck, test_vel_loss, test_heat_loss = validate(
        model=model, val_dataloader=test_dataloader,
        loss_fn=loss_fn, diff_scheduler=diff_scheduler,
        device=device, dtype=dtype,
    )
    logger.info(f"[Stage {stage_idx}] TEST | Loss: {test_loss:.6f}")
    with open(csv_file_path, mode="a", newline="") as f:
        csv.writer(f).writerow(["test", "-", "-", test_loss, "-", test_pck, "-", "-", test_vel_loss, test_heat_loss, "-"])

    logger.info(f"[Etapa {stage_idx}] Finalizada.")
    return local_epoch + 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16

    logger.info(f"Training: 1 GPU ({device})")
    os.makedirs(args.output_dir, exist_ok=True)

    first_config = make_config(args, args.experiment_json[0], args.output_dir)
    model = build_model(first_config, device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate, eps=1e-6)

    diff_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model, subfolder="scheduler"
    )

    logger.info(f"Cargando VAE en {device}...")
    vae = AutoencoderKLQwenImage.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False)
    vae.eval()

    global_config_path = os.path.join(args.output_dir, "experiment_config_global.json")
    if not os.path.exists(global_config_path):
        with open(global_config_path, "w") as f:
            json.dump({
                "experiment_jsons": args.experiment_json,
                "epochs_per_stage": args.epochs,
                "total_epochs":     args.epochs * len(args.experiment_json),
                "pose_weight_scale": args.pose_weight_scale,
                "use_t_scaling":     args.use_t_scaling,
            }, f, indent=2)

    epoch_offset = 0
    for stage_idx, json_path in enumerate(args.experiment_json):
        exp_name     = os.path.splitext(os.path.basename(json_path))[0]
        stage_dir    = os.path.join(args.output_dir, f"stage_{stage_idx:02d}_{exp_name}")
        stage_config = make_config(args, json_path, stage_dir)

        epochs_executed = train_stage(
            config=stage_config,
            model=model,
            vae=vae,
            device=device, dtype=dtype,
            stage_idx=stage_idx,
            optimizer=optimizer,
            diff_scheduler=diff_scheduler,
            epoch_offset=epoch_offset,
            patience=stage_config.patience,
        )
        epoch_offset += epochs_executed

    logger.info(f"Todas las etapas completadas. Total épocas: {epoch_offset}")


if __name__ == "__main__":
    main()
