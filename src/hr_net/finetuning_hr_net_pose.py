#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HRNet Fine-Tuning Script
==========================================================
Descripción:
    Script para re-entrenar (fine-tune) el modelo HRNet utilizando
    imágenes ensuciadas (con ruido)en espacio latente y sus heatmaps originales generados
    previamente.

Asume que:
    - Ruta 1: Imágenes .jpg/.png (Ej: 00_15_1680259460407_rgb_noise_0.4.jpg)
    - Ruta 2: Heatmaps .npy (Ej: 00_15_1680259460407_rgb.npy)
"""

import os
import argparse
import csv
import logging
import re
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
from tqdm import tqdm

# Importamos la arquitectura desde tu script original
# Asegúrate de que tu archivo subido se llame 'hrnet_inference.py'
try:
    from hrnet_inference import PoseHRNet
except ImportError:
    raise ImportError("No se pudo importar 'PoseHRNet'. Asegúrate de que el script original se llama 'hrnet_inference.py' y está en esta misma carpeta.")

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Conexiones del esqueleto para visualización (17 kp COCO body)
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


class NoiseHeatmapDataset(Dataset):
    """
    Dataset personalizado para mapear imágenes con ruido a sus heatmaps originales.
    """
    def __init__(self, imgs_dir: str, heatmaps_dir: str, input_size=(288, 384)):
        self.imgs_dir = Path(imgs_dir)
        self.heatmaps_dir = Path(heatmaps_dir)
        self.input_size = input_size
        
        # Buscar todas las imágenes válidas en ruta1
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        self.img_paths = [p for p in self.imgs_dir.rglob('*') if p.suffix.lower() in valid_extensions]
        
        if not self.img_paths:
            raise ValueError(f"No se encontraron imágenes en {imgs_dir}")
            
        # Valores de normalización estándar de ImageNet (usados en HRNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # LÓGICA DE EMPAREJAMIENTO
        # {base}_rgb_clean.png           → {base}_rgb.npy
        # {base}_rgb_clean_flip.png      → {base}_rgb_flip.npy
        # {base}_rgb_clean_aug1.png      → {base}_rgb.npy
        # {base}_rgb_clean_aug1_flip.png → {base}_rgb_flip.npy
        # {base}_rgb_noise_X.XX.png      → {base}_rgb.npy
        stem = img_path.stem
        m = re.match(r'^(.+)_rgb_(clean(?:_aug\d+)?|noise_[\d.]+)(_flip)?$', stem)
        if m is None:
            raise ValueError(f"Nombre de imagen no reconocido: {img_path.name}")
        base    = m.group(1)
        is_flip = m.group(3) is not None
        npy_name     = f"{base}_rgb_flip.npy" if is_flip else f"{base}_rgb.npy"
        heatmap_path = self.heatmaps_dir / npy_name
        
        if not heatmap_path.exists():
            raise FileNotFoundError(f"Falta el heatmap esperado: {heatmap_path}")

        # 1. Cargar y preprocesar imagen
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w = self.input_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        # Formato canal primero (C, H, W)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

        # 2. Cargar heatmap (.npy)
        heatmap = np.load(heatmap_path)
        heatmap_tensor = torch.from_numpy(heatmap).float()

        return image_tensor, heatmap_tensor


def load_pretrained_weights(model, weights_path, device):
    """Carga los pesos iniciales ignorando errores de 'weights_only'."""
    logger.info(f"Cargando pesos pre-entrenados desde {weights_path}")
    try:
        state_dict = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        state_dict = torch.load(weights_path, map_location=device)
        
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
        
    model.load_state_dict(state_dict, strict=True)
    return model


def evaluate(model, dataloader, criterion, device):
    """Evalúa el modelo en un dataloader y devuelve la pérdida promedio."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, heatmaps in dataloader:
            images   = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            outputs  = model(images)
            loss     = criterion(outputs, heatmaps)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def save_plots(stats, output_dir):
    """Genera y sobreescribe las gráficas de loss por época."""
    output_dir = Path(output_dir)
    epochs = [s['epoch'] for s in stats]

    # — Gráfica de losses —
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, [s['train_loss'] for s in stats], label='train_loss')
    ax.plot(epochs, [s['val_loss_real'] for s in stats], label='val_loss_real')
    if stats[0].get('val_loss_synth') is not None:
        ax.plot(epochs, [s['val_loss_synth'] for s in stats], label='val_loss_synth')
    ax.set_xlabel('Época')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(str(output_dir / 'training_loss.png'))
    plt.close(fig)


def load_checkpoint(ckpt_path, model, optimizer, device):
    """Carga un checkpoint y devuelve el estado del entrenamiento."""
    logger.info(f"Resumiendo desde checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(ckpt['state_dict'])
    optimizer.load_state_dict(ckpt['optimizer'])

    return ckpt  # contiene epoch, best_val_loss, patience_counter, stats, lr


def collect_sample_variants(imgs_dir: Path, base: str) -> list[Path]:
    """
    Dado un base (ej: '00_15_1680259460105'), devuelve todas las variantes
    de imagen ordenadas: clean primero, luego noise de menor a mayor nivel.
    Solo incluye imágenes sin _flip y sin _aug (la imagen original y sus ruidos).
    """
    valid_ext = {'.jpg', '.jpeg', '.png'}
    pat = re.compile(r'^(.+)_rgb_(clean|noise_[\d.]+)$')
    candidates = []
    for f in imgs_dir.glob(f"{base}_rgb_*"):
        if f.suffix.lower() not in valid_ext:
            continue
        m = pat.match(f.stem)
        if m is None:
            continue
        variant = m.group(2)
        # nivel de ruido para ordenar: clean=0.0, noise_X.XX=X.XX
        noise_level = 0.0 if variant == 'clean' else float(variant.split('_')[1])
        candidates.append((noise_level, f))
    candidates.sort(key=lambda x: x[0])
    return [f for _, f in candidates]


def preprocess_image(img_path: Path, input_size=(288, 384)) -> np.ndarray:
    """Carga y normaliza una imagen para inferencia. Devuelve array (C,H,W) float32."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = input_size
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - mean) / std
    return img.transpose(2, 0, 1)  # (C, H, W)


def draw_pose_on_ax(ax, img_rgb: np.ndarray, heatmap: np.ndarray, title: str):
    """Pinta imagen + keypoints + esqueleto en un eje matplotlib."""
    coords = []
    for j in range(heatmap.shape[0]):
        y, x = np.unravel_index(np.argmax(heatmap[j]), heatmap[j].shape)
        coords.append((x, y))
    ax.imshow(img_rgb)
    xs, ys = zip(*coords)
    ax.scatter(xs, ys, c='red', s=15, zorder=3)
    for j, (x, y) in enumerate(coords):
        ax.text(x, y, str(j), fontsize=5, color='yellow', zorder=4)
    for (a, b) in SKELETON:
        if a < len(coords) and b < len(coords):
            ax.plot([coords[a][0], coords[b][0]], [coords[a][1], coords[b][1]],
                    c='cyan', linewidth=1.0, zorder=2)
    ax.set_title(title, fontsize=7)
    ax.axis('off')


def save_inference_grid(model, device, epoch: int, domain: str,
                        base: str, imgs_dir: Path, out_dir: Path,
                        input_size=(288, 384)):
    """
    Genera y guarda una figura con una columna por variante de ruido de la muestra `base`.
    Fila superior: imagen. Fila inferior: keypoints predichos sobre la imagen.
    """
    variants = collect_sample_variants(imgs_dir, base)
    if not variants:
        logger.warning(f"No se encontraron variantes para {base} en {imgs_dir}")
        return

    n = len(variants)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = [[axes[0]], [axes[1]]]

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for col, img_path in enumerate(variants):
            # Preprocesar
            arr = preprocess_image(img_path, input_size)           # (C,H,W) normalizado
            tensor = torch.from_numpy(arr).unsqueeze(0).float().to(device)
            heatmap = model(tensor).squeeze(0).cpu().numpy()       # (J,H,W)

            # Imagen desnormalizada para visualizar
            img_vis = arr.transpose(1, 2, 0) * std + mean          # (H,W,C)
            img_vis = np.clip(img_vis, 0, 1)

            # Nombre de la variante para el título
            m = re.match(r'^.+_rgb_(clean|noise_[\d.]+)$', img_path.stem)
            variant_name = m.group(1) if m else img_path.stem

            axes[0][col].imshow(img_vis)
            axes[0][col].set_title(variant_name, fontsize=7)
            axes[0][col].axis('off')

            draw_pose_on_ax(axes[1][col], img_vis, heatmap, '')

    fig.suptitle(f"Época {epoch:04d} | {domain} | {base}", fontsize=9)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"epoch_{epoch:04d}_{domain}.png"
    fig.savefig(str(out_path), dpi=100)
    plt.close(fig)
    logger.info(f"  → Inferencia visual guardada: {out_path}")


def pick_fixed_base(imgs_dir: Path) -> str | None:
    """Elige deterministamente el primer base disponible en imgs_dir (orden lexicográfico)."""
    pat = re.compile(r'^(.+)_rgb_clean$')
    valid_ext = {'.jpg', '.jpeg', '.png'}
    bases = set()
    for f in imgs_dir.glob('*'):
        if f.suffix.lower() not in valid_ext:
            continue
        m = pat.match(f.stem)
        if m:
            bases.add(m.group(1))
    if not bases:
        return None
    return sorted(bases)[0]


def main():
    parser = argparse.ArgumentParser(description='Fine-Tuning de HRNet')
    parser.add_argument('--imgs-dir', type=str, required=True,
                        help='Directorio base de las imágenes (debe contener train_val/ y test/)')
    parser.add_argument('--heatmaps-dir', type=str, required=True,
                        help='Directorio base de los heatmaps .npy (debe contener train_val/ y test/)')
    # Dataset sintético opcional
    parser.add_argument('--synth-imgs-dir', type=str, default=None,
                        help='(Opcional) Directorio base del dataset sintético (debe contener train_val/ y test/)')
    parser.add_argument('--synth-heatmaps-dir', type=str, default=None,
                        help='(Opcional) Directorio base de los heatmaps .npy sintéticos (debe contener train_val/ y test/)')
    parser.add_argument('--model-path', type=str, default='./models/pose_hrnet_w48_384x288.pth',
                        help='Ruta al modelo base para inicializar los pesos')
    parser.add_argument('--output-model', type=str, default='./models/hrnet_finetuned_best.pth',
                        help='Ruta donde guardar el mejor modelo (checkpoint por val_loss)')
    parser.add_argument('--output-stats', type=str, default='./training_stats.csv',
                        help='Ruta donde guardar las estadísticas de entrenamiento (CSV)')
    parser.add_argument('--epochs', type=int, default=100, help='Número máximo de épocas')
    parser.add_argument('--batch-size', type=int, default=12, help='Tamaño del batch')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Fracción de train_val destinada a validación (por defecto 0.2 = 20%%)')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Épocas sin mejora en val_loss antes de detener el entrenamiento')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla aleatoria para el split train/val reproducible')
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='IDs de GPU a usar para DataParallel (ej: --gpus 0 1). '
                             'Si no se indica, se usa la GPU disponible o CPU.')
    parser.add_argument('--resume', type=str, default=None,
                        help='Ruta a un checkpoint .pth para continuar el entrenamiento desde ese punto')
    parser.add_argument('--num-joints', type=int, default=27,
                        help='Número de joints del modelo (default: 27 para body+manos+pies)')
    # Visualización de inferencia periódica
    parser.add_argument('--vis-every', type=int, default=5,
                        help='Guardar visualización de inferencia cada N épocas (0 = desactivado)')
    parser.add_argument('--vis-dir', type=str, default=None,
                        help='Directorio donde guardar las visualizaciones de inferencia. '
                             'Por defecto: <carpeta del output-model>/inference_vis/')
    parser.add_argument('--vis-base-real', type=str, default=None,
                        help='Base de la muestra real fija para visualización (ej: 00_15_1680259460105). '
                             'Si no se indica, se elige automáticamente el primero del test.')
    parser.add_argument('--vis-base-synth', type=str, default=None,
                        help='Base de la muestra sintética fija para visualización. '
                             'Si no se indica, se elige automáticamente el primero del test sintético.')

    args = parser.parse_args()

    # ── Definición de rutas separadas para imágenes y heatmaps ──
    imgs_dir = Path(args.imgs_dir)
    heatmaps_dir = Path(args.heatmaps_dir)

    train_imgs_dir = imgs_dir / 'train_val'
    train_heatmaps_dir = heatmaps_dir / 'train_val'
    
    test_imgs_dir = imgs_dir / 'test'
    test_heatmaps_dir = heatmaps_dir / 'test'

    use_synth = args.synth_imgs_dir is not None and args.synth_heatmaps_dir is not None
    if use_synth:
        synth_imgs_dir     = Path(args.synth_imgs_dir)
        synth_heatmaps_dir = Path(args.synth_heatmaps_dir)
        synth_train_imgs_dir     = synth_imgs_dir / 'train_val'
        synth_train_heatmaps_dir = synth_heatmaps_dir / 'train_val'
        synth_test_imgs_dir      = synth_imgs_dir / 'test'
        synth_test_heatmaps_dir  = synth_heatmaps_dir / 'test'

    # Directorio de salida para gráficas (mismo que el del modelo)
    plots_dir = Path(args.output_model).parent
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Directorio de visualizaciones de inferencia
    vis_dir = Path(args.vis_dir) if args.vis_dir else plots_dir / 'inference_vis'

    # ── Dispositivo y paralelismo ──────────────────────────────────────────
    use_parallel = False
    if args.gpus is not None and len(args.gpus) > 1 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpus[0]}')
        use_parallel = True
        logger.info(f"Modo DataParallel activado en GPUs: {args.gpus}")
    elif torch.cuda.is_available():
        gpu_id = args.gpus[0] if args.gpus else 0
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Usando GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
    logger.info(f"Iniciando fine-tuning en dispositivo: {device}")

    # ── 1. Datasets ──────────────────────────────────────────────────────────
    logger.info("Cargando dataset train_val (real)...")
    full_trainval_real = NoiseHeatmapDataset(
        imgs_dir=str(train_imgs_dir),
        heatmaps_dir=str(train_heatmaps_dir),
        input_size=(288, 384),
    )
    n_total_real = len(full_trainval_real)
    n_val_real   = int(args.val_split * n_total_real)
    n_train_real = n_total_real - n_val_real
    logger.info(f"Total train_val real: {n_total_real}  →  train: {n_train_real}, val: {n_val_real}")
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset_real, val_dataset_real = random_split(
        full_trainval_real, [n_train_real, n_val_real], generator=generator
    )

    logger.info("Cargando dataset test (real)...")
    test_dataset_real = NoiseHeatmapDataset(
        imgs_dir=str(test_imgs_dir),
        heatmaps_dir=str(test_heatmaps_dir),
        input_size=(288, 384),
    )
    logger.info(f"Total test real: {len(test_dataset_real)}")

    num_workers = 4
    val_loader_real  = DataLoader(val_dataset_real,  batch_size=args.batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)
    test_loader_real = DataLoader(test_dataset_real, batch_size=args.batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=True)

    # Dataset sintético (opcional)
    val_loader_synth  = None
    test_loader_synth = None
    synth_test_imgs_dir_resolved = None
    if use_synth:
        logger.info("Cargando dataset train_val (sintético)...")
        full_trainval_synth = NoiseHeatmapDataset(
            imgs_dir=str(synth_train_imgs_dir),
            heatmaps_dir=str(synth_train_heatmaps_dir),
            input_size=(288, 384),
        )
        n_total_synth = len(full_trainval_synth)
        n_val_synth   = int(args.val_split * n_total_synth)
        n_train_synth = n_total_synth - n_val_synth
        logger.info(f"Total train_val sintético: {n_total_synth}  →  train: {n_train_synth}, val: {n_val_synth}")
        generator_synth = torch.Generator().manual_seed(args.seed)
        train_dataset_synth, val_dataset_synth = random_split(
            full_trainval_synth, [n_train_synth, n_val_synth], generator=generator_synth
        )
        logger.info("Cargando dataset test (sintético)...")
        test_dataset_synth = NoiseHeatmapDataset(
            imgs_dir=str(synth_test_imgs_dir),
            heatmaps_dir=str(synth_test_heatmaps_dir),
            input_size=(288, 384),
        )
        logger.info(f"Total test sintético: {len(test_dataset_synth)}")
        val_loader_synth  = DataLoader(val_dataset_synth,  batch_size=args.batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True)
        test_loader_synth = DataLoader(test_dataset_synth, batch_size=args.batch_size, shuffle=False,
                                       num_workers=num_workers, pin_memory=True)
        synth_test_imgs_dir_resolved = synth_test_imgs_dir
        # Mezclar real + sintético en un único DataLoader de entrenamiento
        train_dataset_combined = ConcatDataset([train_dataset_real, train_dataset_synth])
        logger.info(f"Train combinado (real + sintético): {len(train_dataset_combined)} muestras")
    else:
        train_dataset_combined = train_dataset_real

    train_loader = DataLoader(train_dataset_combined, batch_size=args.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)

    # ── Elegir muestras fijas para visualización ──────────────────────────────
    vis_base_real = args.vis_base_real or pick_fixed_base(test_imgs_dir)
    vis_base_synth = None
    if use_synth:
        vis_base_synth = args.vis_base_synth or pick_fixed_base(synth_test_imgs_dir_resolved)
    if args.vis_every > 0:
        logger.info(f"Visualización cada {args.vis_every} épocas → {vis_dir}")
        logger.info(f"  Muestra real:     {vis_base_real}")
        logger.info(f"  Muestra sintética: {vis_base_synth}")

    # ── 2. Modelo ────────────────────────────────────────────────────────────
    model = PoseHRNet(width=48, num_joints=args.num_joints)
    if Path(args.model_path).exists():
        model = load_pretrained_weights(model, args.model_path, device)
    else:
        logger.warning(f"No se encontró {args.model_path}. Entrenando desde cero.")
    model = model.to(device)
    if use_parallel:
        model = nn.DataParallel(model, device_ids=args.gpus)
        logger.info(f"Modelo envuelto en DataParallel con device_ids={args.gpus}")

    # ── 3. Función de pérdida y optimizador ──────────────────────────────────
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ── 4. Variables de seguimiento ───────────────────────────────────────────
    stats            = []
    best_val_loss    = float('inf')
    patience_counter = 0
    start_epoch      = 0
    output_path      = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path = Path(args.output_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    # Checkpoint de época (último, para resume)
    epoch_ckpt_path = output_path.parent / 'hrnet_epoch_last.pth'

    # ── 4b. Resume desde checkpoint ───────────────────────────────────────────
    resume_path = args.resume or (epoch_ckpt_path if epoch_ckpt_path.exists() else None)
    if resume_path and Path(resume_path).exists():
        ckpt = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch      = ckpt.get('epoch', 0)           # época siguiente a reanudar
        best_val_loss    = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        stats            = ckpt.get('stats', [])
        logger.info(
            f"Entrenamiento reanudado desde época {start_epoch} "
            f"(best_val_loss={best_val_loss:.6f}, paciencia={patience_counter})"
        )
        # Restaurar LR guardado
        saved_lr = ckpt.get('lr', args.lr)
        for pg in optimizer.param_groups:
            pg['lr'] = saved_lr
        logger.info(f"LR restaurado: {saved_lr}")

    # ── 5. Bucle de entrenamiento ─────────────────────────────────────────────
    logger.info("Comenzando el entrenamiento...")
    for epoch in range(start_epoch, args.epochs):
        # — Train (real + sintético mezclados) —
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Época {epoch + 1}/{args.epochs} [train]")
        for images, heatmaps in progress_bar:
            images   = images.to(device, non_blocking=True)
            heatmaps = heatmaps.to(device, non_blocking=True)
            outputs  = model(images)
            loss     = criterion(outputs, heatmaps)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        avg_train_loss = running_loss / len(train_loader)

        # — Validation —
        # Early stopping solo sobre val_loss_real; val_loss_synth es solo monitorización
        avg_val_loss_real = evaluate(model, val_loader_real, criterion, device)
        avg_val_loss_synth = None
        if val_loader_synth is not None:
            avg_val_loss_synth = evaluate(model, val_loader_synth, criterion, device)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(
            f"Época [{epoch + 1}/{args.epochs}] — "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss real: {avg_val_loss_real:.6f}" +
            (f" | Val Loss synth: {avg_val_loss_synth:.6f}" if avg_val_loss_synth is not None else "") +
            f" | LR: {current_lr:.2e}"
        )

        # — Guardar estadísticas de esta época —
        epoch_stats = {
            'epoch':          epoch + 1,
            'train_loss':     avg_train_loss,
            'val_loss_real':  avg_val_loss_real,
            'val_loss_synth': avg_val_loss_synth,
            'lr':             current_lr,
        }
        stats.append(epoch_stats)

        # — Actualizar CSV por época —
        with open(stats_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['epoch', 'train_loss', 'val_loss_real', 'val_loss_synth', 'lr']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(stats)

        # — Actualizar gráficas —
        save_plots(stats, plots_dir)

        # — Visualización de inferencia periódica ─────────────────────────────
        if args.vis_every > 0 and (epoch + 1) % args.vis_every == 0:
            raw_model_vis = model.module if isinstance(model, nn.DataParallel) else model
            if vis_base_real:
                save_inference_grid(
                    raw_model_vis, device, epoch + 1, 'real',
                    vis_base_real, test_imgs_dir, vis_dir,
                )
            if vis_base_synth and synth_test_imgs_dir_resolved:
                save_inference_grid(
                    raw_model_vis, device, epoch + 1, 'synth',
                    vis_base_synth, synth_test_imgs_dir_resolved, vis_dir,
                )

        # — Checkpoint de época (siempre, para resume) —
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        epoch_ckpt = {
            'epoch':            epoch + 1,
            'state_dict':       raw_model.state_dict(),
            'optimizer':        optimizer.state_dict(),
            'lr':               current_lr,
            'train_loss':       avg_train_loss,
            'val_loss_real':    avg_val_loss_real,
            'val_loss_synth':   avg_val_loss_synth,
            'best_val_loss':    best_val_loss,
            'patience_counter': patience_counter,
            'stats':            stats,
        }
        torch.save(epoch_ckpt, str(epoch_ckpt_path))

        # — Checkpoint del mejor modelo (por val_loss_real) —
        if avg_val_loss_real < best_val_loss:
            best_val_loss    = avg_val_loss_real
            patience_counter = 0
            torch.save(epoch_ckpt, str(output_path))
            logger.info(f"  → Mejor modelo guardado (val_loss_real={best_val_loss:.6f})")
        else:
            patience_counter += 1
            logger.info(
                f"  → Sin mejora en val_loss_real. "
                f"Paciencia: {patience_counter}/{args.early_stopping}"
            )
            if patience_counter >= args.early_stopping:
                logger.info("Early stopping activado. Finalizando entrenamiento anticipado.")
                break

    # ── 6. Guardar estadísticas finales en CSV ─────────────────────────────────
    # (el CSV ya está actualizado por época; esta escritura final garantiza consistencia)
    with open(stats_path, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['epoch', 'train_loss', 'val_loss_real', 'val_loss_synth', 'lr']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
    logger.info(f"Estadísticas de entrenamiento guardadas en: {stats_path}")

    # ── 7. Evaluación final en el conjunto de test ────────────────────────────
    logger.info("Evaluando el mejor modelo en el conjunto de test...")
    try:
        best_ckpt = torch.load(str(output_path), map_location=device, weights_only=False)
    except TypeError:
        best_ckpt = torch.load(str(output_path), map_location=device)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(best_ckpt['state_dict'])

    test_loss_real = evaluate(raw_model, test_loader_real, criterion, device)
    logger.info(f"Test Loss real (mejor modelo): {test_loss_real:.6f}")

    test_loss_synth = None
    if test_loader_synth is not None:
        test_loss_synth = evaluate(raw_model, test_loader_synth, criterion, device)
        logger.info(f"Test Loss sintético (mejor modelo): {test_loss_synth:.6f}")

    # Añadir resultado de test al CSV
    with open(stats_path, 'a', newline='', encoding='utf-8') as f:
        fieldnames = ['epoch', 'train_loss', 'val_loss_real', 'val_loss_synth', 'lr']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow({
            'epoch':          'test',
            'train_loss':     '',
            'val_loss_real':  test_loss_real,
            'val_loss_synth': test_loss_synth if test_loss_synth is not None else '',
            'lr':             '',
        })

    logger.info("✅ Proceso finalizado.")


if __name__ == '__main__':
    main()