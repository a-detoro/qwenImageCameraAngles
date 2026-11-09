"""
Compara las dos pruebas cortas de ablación (con/sin heatmap loss) leyendo sus
training_metrics.csv y decide cuál tuvo mejor val_loss (y, si aplica, val_pck).
"""
import csv
import sys
from pathlib import Path

RUNS = {
    "no_heatmap":   "/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/ablation_no_heatmap",
    "with_heatmap": "/mnt/nas_dataset/antonio/qwen/OUTPUTS_EXPERIMENTOS_THERMAL/ablation_with_heatmap",
}


def read_metrics(output_dir):
    stage_dirs = sorted(Path(output_dir).glob("stage_*"))
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


def summarize(name, rows):
    if not rows:
        print(f"[{name}] sin datos todavía")
        return None
    best = min(rows, key=lambda r: float(r["val_loss"]))
    last = rows[-1]
    print(f"[{name}] epochs corridas: {len(rows)}")
    print(f"  mejor val_loss: {float(best['val_loss']):.6f} (epoch {best['global_epoch']}) | val_pck={float(best['val_pck']):.4f}")
    print(f"  última epoch:   val_loss={float(last['val_loss']):.6f} | val_pck={float(last['val_pck']):.4f}")
    return float(best["val_loss"])


def main():
    results = {}
    for name, out_dir in RUNS.items():
        rows = read_metrics(out_dir)
        results[name] = summarize(name, rows)

    print()
    if all(v is not None for v in results.values()):
        winner = min(results, key=results.get)
        print(f"GANADOR: {winner} (val_loss={results[winner]:.6f})")
    else:
        print("Aún faltan datos de alguna de las dos pruebas.")


if __name__ == "__main__":
    main()
