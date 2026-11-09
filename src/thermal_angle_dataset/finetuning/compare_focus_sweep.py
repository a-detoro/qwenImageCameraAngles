"""
Compara las corridas de cribado de run_focus_sweep.sh: para cada pose_weight_scale,
lee training_metrics.csv y resume la tendencia de val_loss / val PCK / pose_weight_mean,
para decidir si merece la pena una corrida completa (336/112, 15+ epochs) con algun
candidato, o si el mecanismo de loss_vel_focus no muestra señal ni en los extremos.
"""
import csv
from pathlib import Path

SWEEP_ROOT = Path("/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/focus_sweep")


def read_metrics(output_dir: Path):
    stage_dirs = sorted(output_dir.glob("stage_*"))
    if not stage_dirs:
        return None
    csv_path = stage_dirs[0] / "training_metrics.csv"
    if not csv_path.exists():
        return None
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["global_epoch"] == "test":
                continue
            rows.append(row)
    return rows


def main():
    if not SWEEP_ROOT.exists():
        print(f"No existe {SWEEP_ROOT} todavia.")
        return

    results = []
    for scale_dir in sorted(SWEEP_ROOT.glob("scale_*"), key=lambda p: float(p.name.split("_")[1])):
        scale = scale_dir.name.split("_")[1]
        rows = read_metrics(scale_dir)
        if not rows:
            print(f"[scale={scale}] sin datos todavia")
            continue

        first_val = float(rows[0]["val_loss"])
        last_val = float(rows[-1]["val_loss"])
        best_val = min(float(r["val_loss"]) for r in rows)
        pose_weights = [float(r["pose_weight_mean"]) for r in rows if r.get("pose_weight_mean") not in (None, "", "-")]
        mean_pose_weight = sum(pose_weights) / len(pose_weights) if pose_weights else float("nan")

        results.append({
            "scale": float(scale),
            "epochs": len(rows),
            "first_val_loss": first_val,
            "last_val_loss": last_val,
            "best_val_loss": best_val,
            "delta": first_val - last_val,
            "mean_pose_weight": mean_pose_weight,
        })

    if not results:
        print("Ningun resultado todavia.")
        return

    print(f"{'scale':>8} | {'epochs':>6} | {'val_loss ini':>12} | {'val_loss fin':>12} | {'mejora':>8} | {'peso medio':>10}")
    print("-" * 70)
    for r in results:
        print(f"{r['scale']:>8} | {r['epochs']:>6} | {r['first_val_loss']:>12.6f} | "
              f"{r['last_val_loss']:>12.6f} | {r['delta']:>8.5f} | {r['mean_pose_weight']:>10.3f}")

    print()
    best = min(results, key=lambda r: r["best_val_loss"])
    print(f"Mejor val_loss absoluta: scale={best['scale']} ({best['best_val_loss']:.6f})")

    deltas = [r["delta"] for r in results]
    if max(deltas) - min(deltas) < 1e-4:
        print(
            "\n⚠️  La mejora (val_loss inicial - final) es prácticamente idéntica en "
            "todas las escalas, incluso en los extremos (0.5 vs 64). Esto sugiere que "
            "el problema NO es la magnitud del peso, sino que la señal de pose se está "
            "diluyendo en el promedio global (pocos tokens con peso alto frente al total), "
            "o que el rango de LoRA no tiene margen para aprovechar la diferencia. "
            "Antes de descartar el enfoque, mira el PCK/pose_weight_mean real logueado "
            "y considera subir lora_rank en el cribado."
        )


if __name__ == "__main__":
    main()
