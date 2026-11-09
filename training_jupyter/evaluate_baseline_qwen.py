import os, sys, argparse, logging
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(current_dir)

os.environ["CUDA_DEVICE_ORDER"]   = "PCI_BUS_ID"
os.environ["HF_HOME"]             = "/nas/antoniodetoro/qwen/hf_cache"
os.environ["TMPDIR"]              = "/dev/shm"
os.environ["PYTHONNOUSERSITE"]    = "1"

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader
from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
from diffusers.models import AutoencoderKLQwenImage

from config import TrainingConfig
from dataset import LatentsDatasetFromJSON, load_experiment_json, make_collate_latents
from model import QwenSingleGPUWrapper
from latent_utils import unpack_latents

logging.basicConfig(level=logging.INFO, force=True, format="%(message)s")
logger = logging.getLogger(__name__)


def build_ts_reverse_index(test_dataset):
    """Construye un dict {path_str -> timestamp} invirtiendo ts_index."""
    reverse = {}
    for ts, paths in test_dataset.ts_index.items():
        for p in paths:
            reverse[str(p)] = ts
    return reverse


def build_test_dataloader(config):
    _, _, test_ts = load_experiment_json(config.experiment_json)
    test_dataset  = LatentsDatasetFromJSON(config.latents_dir, test_ts, split="test")
    collate_fn    = make_collate_latents(test_dataset.global_max_seq_len)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        shuffle=False,
        num_workers=4,
    )
    return test_dataset, test_dataloader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir",        type=str, required=True)
    parser.add_argument("--experiment_json",   type=str, required=True)
    parser.add_argument("--latents_dir",       type=str, required=True)
    parser.add_argument("--base_model",        type=str, default="Qwen/Qwen-Image-Edit-2509")
    parser.add_argument("--transformer_model", type=str, default="../models/Qwen-Fused-Angles")
    parser.add_argument("--batch_size",        type=int, default=1)
    parser.add_argument("--inference_steps",   type=int, default=6)
    return parser.parse_args()


@torch.no_grad()
def run_inference_baseline(
    model,
    vae,
    base_scheduler,
    device,
    test_dataset,
    reverse_ts_index,
    save_dir,
    img_height,
    img_width,
    num_steps,
    dtype=torch.float32,
):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    infer_scheduler = FlowMatchEulerDiscreteScheduler.from_config(base_scheduler.config)

    vae_mean = torch.tensor(vae.config.latents_mean).view(
        1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
    vae_std  = torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)

    for sample_idx in range(len(test_dataset)):
        sample = test_dataset[sample_idx]

        # Timestamp desde el índice inverso
        pt_path = str(test_dataset.files[sample_idx])
        timestamp = reverse_ts_index.get(pt_path, sample_idx)
        angle_diff = int(sample.get("angle_diffs", torch.tensor(0)).item())

        prompt = sample["prompt_embeds"].unsqueeze(0).to(device, dtype=dtype)
        mask   = sample["prompt_embeds_mask"].unsqueeze(0).to(device)

        target_packed = sample["target_latents_packed"].unsqueeze(0).to(device, dtype=dtype)
        src_packed    = sample.get("source_latents_packed")
        if src_packed is not None:
            src_packed = src_packed.unsqueeze(0).to(device, dtype=dtype)

        latent_shape   = target_packed.shape
        image_seq_len  = latent_shape[1]

        # Timesteps con dynamic shifting si aplica
        if infer_scheduler.config.get("use_dynamic_shifting", False):
            base_seq   = infer_scheduler.config.get("base_image_seq_len", 256)
            max_seq    = infer_scheduler.config.get("max_image_seq_len", 4096)
            base_shift = infer_scheduler.config.get("base_shift", 0.5)
            max_shift  = infer_scheduler.config.get("max_shift", 1.15)
            m  = (max_shift - base_shift) / (max_seq - base_seq)
            b  = base_shift - m * base_seq
            mu = image_seq_len * m + b
            infer_scheduler.set_timesteps(num_steps, device=device, mu=mu)
        else:
            infer_scheduler.set_timesteps(num_steps, device=device)

        gen     = torch.Generator(device=device).manual_seed(42 + sample_idx)
        latents = torch.randn(latent_shape, generator=gen, device=device, dtype=dtype)

        for t in infer_scheduler.timesteps:
            timestep      = t.view(1).to(device)
            timestep_norm = (timestep.float() / 1000.0).to(dtype)

            if src_packed is not None:
                latent_input = torch.cat([latents, src_packed], dim=1)
            else:
                latent_input = latents

            v_pred = model(latent_input, prompt, mask, timestep_norm)
            v_pred = v_pred[:, :latent_shape[1], :]

            latents = infer_scheduler.step(
                v_pred.float(), t, latents.float()
            ).prev_sample.to(dtype)

        # Decode
        try:
            def decode_packed(packed):
                spatial = unpack_latents(packed.float(), img_height, img_width).to(vae.dtype)
                z_raw   = spatial * vae_std + vae_mean
                return vae.decode(z_raw, return_dict=False)[0][:, :, 0]

            def to_pil(t_img):
                img = torch.clamp(t_img[0].cpu().float() * 0.5 + 0.5, 0.0, 1.0)
                return T.ToPILImage()(img)

            pred_pil = to_pil(decode_packed(latents))
            gt_pil   = to_pil(decode_packed(target_packed))
            w, h     = pred_pil.size

            if src_packed is not None:
                src_pil  = to_pil(decode_packed(src_packed))
                combined = Image.new("RGB", (w * 3, h))
                combined.paste(src_pil,  (0, 0))
                combined.paste(pred_pil, (w, 0))
                combined.paste(gt_pil,   (w * 2, 0))
            else:
                combined = Image.new("RGB", (w * 2, h))
                combined.paste(pred_pil, (0, 0))
                combined.paste(gt_pil,   (w, 0))

            fname = os.path.join(save_dir, f"{timestamp}_angle{angle_diff:+d}_rgb.png")
            combined.save(fname)
            logger.info(f"[{sample_idx+1}/{len(test_dataset)}] Guardado: {fname}")

        except Exception as e:
            logger.warning(f"Error sample {sample_idx} (ts={timestamp}, angle={angle_diff:+d}): {e}")

    logger.info("Inferencia baseline completada.")


def main():
    args   = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32

    inference_dir = os.path.join(args.output_dir, "inference_results_baseline")
    os.makedirs(inference_dir, exist_ok=True)

    config = TrainingConfig(
        latents_dir=args.latents_dir,
        experiment_json=args.experiment_json,
        output_dir=args.output_dir,
        base_model=args.base_model,
        transformer_model=args.transformer_model,
        batch_size=args.batch_size,
    )

    test_dataset, _ = build_test_dataloader(config)
    reverse_ts_index = build_ts_reverse_index(test_dataset)

    first_sample   = torch.load(test_dataset.files[0], weights_only=True)
    img_resolution = first_sample.get("resolution", 1024)

    # Transformer original, float32, sin LoRA
    transformer = QwenImageTransformer2DModel.from_pretrained(
        config.transformer_model,
        subfolder=None,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    model = QwenSingleGPUWrapper(transformer)
    model.enable_sequential_cpu_offload()   # offload a CPU entre capas
    model.eval()

    vae = AutoencoderKLQwenImage.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=torch.float32
    ).to(device)
    vae.requires_grad_(False).eval()

    diff_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.base_model, subfolder="scheduler"
    )

    logger.info(f"Generando inferencia baseline en {inference_dir}...")
    run_inference_baseline(
        model=model,
        vae=vae,
        base_scheduler=diff_scheduler,
        device=device,
        test_dataset=test_dataset,
        reverse_ts_index=reverse_ts_index,
        save_dir=inference_dir,
        img_height=img_resolution,
        img_width=img_resolution,
        num_steps=args.inference_steps,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()