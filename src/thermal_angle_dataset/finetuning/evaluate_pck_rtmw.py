"""
Evalua PCK (Percentage of Correct Keypoints), MSE y SSIM de las imagenes
generadas durante inferencia, frente a su GT correspondiente, usando RTMW
(rtmlib Wholebody) como extractor de pose. Recorre todos los experimentos
THERMAL que tengan carpetas inference_epoch_* guardadas (una imagen por
muestra, formada por 3 paneles pegados horizontalmente: src | generada | GT,
cada uno de 512x512).

PCK@0.2 sobre los 17 keypoints de cuerpo (subconjunto COCO-body de los 133 de
RTMW Wholebody), normalizado por el tamano del torso (distancia hombro-cadera
promediada entre ambos lados). Se ignoran cara/manos para el calculo de PCK:
en imagenes termicas borrosas de baja resolucion esos puntos son ruido, no
senal (pero SI se dibujan si tienen confianza suficiente, para inspeccion
visual).

MSE y SSIM se calculan en escala de grises sobre el panel completo
(generada vs GT), como medida global de fidelidad de imagen independiente de
la pose.

Ademas de los CSV de resultados, genera:
  - pck_eval/visualizations/<experimento>/epoch_XXX/<split>/<file>.png
    triptico [src | generada+esqueleto RTMW | GT+esqueleto RTMW] con texto de
    PCK/MSE/SSIM superpuesto.
  - pck_eval/charts/*.png
    evolucion de PCK/MSE/SSIM medios por epoca, una linea por experimento.

Uso:
    python evaluate_pck_rtmw.py --root /mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL \
        --out_dir pck_eval --conf_thresh 0.3 --pck_thresh 0.2
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Indices COCO-body dentro del esquema COCO-WholeBody (133 kpts) que usa RTMW.
N_BODY = 17
L_SHOULDER, R_SHOULDER = 5, 6
L_HIP, R_HIP = 11, 12


def torso_size(kpts, scores, conf_thresh):
    sizes = []
    if scores[L_SHOULDER] > conf_thresh and scores[L_HIP] > conf_thresh:
        sizes.append(np.linalg.norm(kpts[L_SHOULDER] - kpts[L_HIP]))
    if scores[R_SHOULDER] > conf_thresh and scores[R_HIP] > conf_thresh:
        sizes.append(np.linalg.norm(kpts[R_SHOULDER] - kpts[R_HIP]))
    if not sizes:
        return None
    return float(np.mean(sizes))


def best_person_idx(scores, n_kpts):
    body_scores = scores[:, :n_kpts].mean(axis=1)
    return int(np.argmax(body_scores))


def compute_pck(kpts_gt, scores_gt, kpts_gen, scores_gen, conf_thresh, pck_thresh):
    size = torso_size(kpts_gt, scores_gt, conf_thresh)
    if size is None or size == 0:
        return None, 0, 0
    valid = scores_gt > conf_thresh
    n_valid = int(valid.sum())
    if n_valid == 0:
        return None, 0, 0
    dists = np.linalg.norm(kpts_gt[valid] - kpts_gen[valid], axis=1)
    n_correct = int((dists / size <= pck_thresh).sum())
    return 100.0 * n_correct / n_valid, n_correct, n_valid


def compute_mse_ssim(gen_panel_bgr, gt_panel_bgr):
    from skimage.metrics import structural_similarity
    gen_gray = cv2.cvtColor(gen_panel_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gt_gray = cv2.cvtColor(gt_panel_bgr, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mse = float(np.mean((gen_gray - gt_gray) ** 2))
    ssim = float(structural_similarity(gt_gray, gen_gray, data_range=255.0))
    return mse, ssim


def find_inference_dirs(root: Path):
    out = []
    for exp_dir in sorted(root.iterdir()):
        if not exp_dir.is_dir():
            continue
        for stage_dir in sorted(exp_dir.glob("stage_*")):
            for inf_dir in sorted(stage_dir.glob("inference_epoch_*")):
                try:
                    epoch = int(inf_dir.name.split("_")[-1])
                except ValueError:
                    continue
                out.append((exp_dir.name, stage_dir.name, epoch, inf_dir))
    return out


def make_triptych(src_panel, gen_panel, gt_panel,
                   kpts_gen_full, scores_gen_full, kpts_gt_full, scores_gt_full,
                   conf_thresh, header_text):
    from rtmlib import draw_skeleton

    gen_vis = draw_skeleton(gen_panel.copy(), kpts_gen_full[None, ...], scores_gen_full[None, ...],
                             kpt_thr=conf_thresh, radius=3, line_width=2)
    gt_vis = draw_skeleton(gt_panel.copy(), kpts_gt_full[None, ...], scores_gt_full[None, ...],
                            kpt_thr=conf_thresh, radius=3, line_width=2)

    def label(img, text):
        img = img.copy()
        cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return img

    src_l = label(src_panel, "SRC")
    gen_l = label(gen_vis, "GENERADA + pose RTMW")
    gt_l = label(gt_vis, "GT + pose RTMW")

    triptych = np.concatenate([src_l, gen_l, gt_l], axis=1)

    header = np.zeros((28, triptych.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, header_text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    return np.concatenate([header, triptych], axis=0)


def make_charts(summary_rows, charts_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    charts_dir.mkdir(parents=True, exist_ok=True)

    by_exp_split = defaultdict(list)
    for r in summary_rows:
        by_exp_split[(r["experiment"], r["split"])].append(r)
    for key in by_exp_split:
        by_exp_split[key].sort(key=lambda r: r["epoch"])

    metrics = [("pck_mean", "PCK medio (%) @0.2", "PCK (%)"),
               ("mse_mean", "MSE medio (grises, 0-255^2)", "MSE"),
               ("ssim_mean", "SSIM medio (grises)", "SSIM")]

    for metric_key, title, ylabel in metrics:
        plt.figure(figsize=(7, 5))
        for (exp, split), rows in sorted(by_exp_split.items()):
            epochs = [r["epoch"] for r in rows]
            values = [r[metric_key] for r in rows]
            style = "-o" if split == "test" else "--s"
            plt.plot(epochs, values, style, label=f"{exp} ({split})")
        plt.title(title)
        plt.xlabel("Epoca")
        plt.ylabel(ylabel)
        plt.legend(fontsize=8)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = charts_dir / f"{metric_key}_vs_epoch.png"
        plt.savefig(out_path, dpi=130)
        plt.close()
        print(f"Grafica guardada: {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL")
    ap.add_argument("--out_dir", default="pck_eval")
    ap.add_argument("--conf_thresh", type=float, default=0.3)
    ap.add_argument("--pck_thresh", type=float, default=0.2)
    ap.add_argument("--mode", default="performance", choices=["performance", "balanced", "lightweight"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no_viz", action="store_true", help="No generar imagenes de visualizacion")
    ap.add_argument("--viz_sample_n", type=int, default=12,
                     help="Cuantas imagenes (aleatorias, seed fija) visualizar por grupo experimento/epoca/split. "
                          "Las metricas SIEMPRE se calculan sobre todas las imagenes; esto solo limita cuantos "
                          "tripticos PNG se guardan a disco, para no generar miles de ficheros redundantes.")
    args = ap.parse_args()

    from rtmlib import Wholebody

    wb = Wholebody(mode=args.mode, backend="onnxruntime", device=args.device)

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_root = out_dir / "visualizations"

    jobs = find_inference_dirs(root)
    print(f"Encontradas {len(jobs)} carpetas inference_epoch_* en {len(set(j[0] for j in jobs))} experimentos.")

    rows = []
    for exp_name, stage_name, epoch, inf_dir in jobs:
        for split_dir in sorted(inf_dir.iterdir()):
            if not split_dir.is_dir():
                continue
            split = split_dir.name
            img_paths = sorted(split_dir.glob("*.png"))
            print(f"[{exp_name} | {stage_name} | epoch={epoch} | {split}] {len(img_paths)} imagenes")

            import random
            rng = random.Random(f"{exp_name}|{epoch}|{split}")
            viz_names = set(p.name for p in rng.sample(img_paths, min(args.viz_sample_n, len(img_paths))))

            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                row = dict(experiment=exp_name, stage=stage_name, epoch=epoch, split=split,
                           file=img_path.name, pck=None, n_correct=0, n_valid=0,
                           mse=None, ssim=None,
                           gt_detected=False, gen_detected=False, fail_reason="")
                if img is None:
                    row["fail_reason"] = "no_se_pudo_leer"
                    rows.append(row)
                    continue

                h, w = img.shape[:2]
                third = w // 3
                src_panel = img[:, 0:third]
                gen_panel = img[:, third:2 * third]
                gt_panel = img[:, 2 * third:]

                kpts_gen_all, scores_gen_all = wb(gen_panel)
                kpts_gt_all, scores_gt_all = wb(gt_panel)

                gen_ok = kpts_gen_all is not None and len(kpts_gen_all) > 0
                gt_ok = kpts_gt_all is not None and len(kpts_gt_all) > 0
                row["gen_detected"] = gen_ok
                row["gt_detected"] = gt_ok

                mse, ssim = compute_mse_ssim(gen_panel, gt_panel)
                row["mse"] = mse
                row["ssim"] = ssim

                if not gen_ok or not gt_ok:
                    row["fail_reason"] = "sin_persona_detectada"
                    rows.append(row)
                    continue

                idx_gen = best_person_idx(scores_gen_all, N_BODY)
                idx_gt = best_person_idx(scores_gt_all, N_BODY)
                kpts_gen_full, scores_gen_full = kpts_gen_all[idx_gen], scores_gen_all[idx_gen]
                kpts_gt_full, scores_gt_full = kpts_gt_all[idx_gt], scores_gt_all[idx_gt]

                pck, n_correct, n_valid = compute_pck(
                    kpts_gt_full[:N_BODY], scores_gt_full[:N_BODY],
                    kpts_gen_full[:N_BODY], scores_gen_full[:N_BODY],
                    args.conf_thresh, args.pck_thresh,
                )
                row["pck"], row["n_correct"], row["n_valid"] = pck, n_correct, n_valid
                if pck is None:
                    row["fail_reason"] = "torso_no_fiable"
                rows.append(row)

                if not args.no_viz and img_path.name in viz_names:
                    pck_txt = f"{pck:.1f}%" if pck is not None else "N/A"
                    header_text = (f"{exp_name} | epoch {epoch} | {split} | {img_path.name}  "
                                    f"PCK={pck_txt}  MSE={mse:.1f}  SSIM={ssim:.3f}")
                    triptych = make_triptych(src_panel, gen_panel, gt_panel,
                                              kpts_gen_full, scores_gen_full,
                                              kpts_gt_full, scores_gt_full,
                                              args.conf_thresh, header_text)
                    out_img_dir = viz_root / exp_name / f"epoch_{epoch:03d}" / split
                    out_img_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_img_dir / img_path.name), triptych)

    out_csv = out_dir / "pck_results_per_image.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nGuardado detalle por imagen en {out_csv} ({len(rows)} filas)")

    groups_pck = defaultdict(list)
    groups_mse = defaultdict(list)
    groups_ssim = defaultdict(list)
    for r in rows:
        key = (r["experiment"], r["epoch"], r["split"])
        if r["pck"] is not None:
            groups_pck[key].append(r["pck"])
        if r["mse"] is not None:
            groups_mse[key].append(r["mse"])
        if r["ssim"] is not None:
            groups_ssim[key].append(r["ssim"])

    all_keys = sorted(set(groups_pck) | set(groups_mse) | set(groups_ssim))
    summary_rows = []
    for key in all_keys:
        exp, epoch, split = key
        n_total = sum(1 for r in rows if (r["experiment"], r["epoch"], r["split"]) == key)
        pcks = np.array(groups_pck[key]) if groups_pck[key] else None
        mses = np.array(groups_mse[key]) if groups_mse[key] else None
        ssims = np.array(groups_ssim[key]) if groups_ssim[key] else None
        summary_rows.append(dict(
            experiment=exp, epoch=epoch, split=split,
            pck_mean=round(float(pcks.mean()), 2) if pcks is not None else None,
            pck_median=round(float(np.median(pcks)), 2) if pcks is not None else None,
            pck_std=round(float(pcks.std()), 2) if pcks is not None else None,
            n_pck_ok=len(pcks) if pcks is not None else 0,
            mse_mean=round(float(mses.mean()), 3) if mses is not None else None,
            ssim_mean=round(float(ssims.mean()), 4) if ssims is not None else None,
            n_total=n_total,
        ))

    out_summary = out_dir / "pck_results_summary.csv"
    with out_summary.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nResumen guardado en {out_summary}\n")

    print(f"{'experimento':30s} {'epoch':>5s} {'split':>5s} {'PCK':>8s} {'MSE':>9s} {'SSIM':>7s} {'n':>4s}")
    for r in summary_rows:
        pck_s = f"{r['pck_mean']:.2f}%" if r['pck_mean'] is not None else "N/A"
        mse_s = f"{r['mse_mean']:.2f}" if r['mse_mean'] is not None else "N/A"
        ssim_s = f"{r['ssim_mean']:.4f}" if r['ssim_mean'] is not None else "N/A"
        print(f"{r['experiment']:30s} {r['epoch']:5d} {r['split']:>5s} {pck_s:>8s} {mse_s:>9s} {ssim_s:>7s} {r['n_total']:4d}")

    if not args.no_viz:
        make_charts(summary_rows, out_dir / "charts")


if __name__ == "__main__":
    main()
