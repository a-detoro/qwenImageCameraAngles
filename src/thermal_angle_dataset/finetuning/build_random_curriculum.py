"""
Construye un curriculum de CONTROL: mismo numero de muestras train/val por
usuario que experimentos/dataset_splits_thermal.json (336 train / 112 val,
14 usuarios train/val, 24 train + 8 val cada uno), mismo test set (sin tocar),
pero eligiendo los timestamps de train/val de forma UNIFORMEMENTE ALEATORIA
en vez de por rareza de pose (muestreo estratificado via knn_score).

Objetivo: aislar si el metodo de seleccion de datos (curriculum por rareza de
pose) aporta algo frente a una seleccion aleatoria de igual tamano, usando el
mismo pool de timestamps disponibles por usuario (notebooks/timestamps_by_rarity.csv,
que contiene TODAS las poses de cada usuario, no solo las seleccionadas).

Uso:
    python build_random_curriculum.py --seed 1234 \
        --output experimentos/dataset_splits_thermal_random.json
"""
import csv
import json
import random
import argparse
from collections import defaultdict

RARITY_CSV = "/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/notebooks/timestamps_by_rarity.csv"
ORIGINAL_JSON = "experimentos/dataset_splits_thermal.json"


def load_pools(path):
    pools = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["split"] == "train_val":
                pools[row["user_id"]].append(int(row["timestamp"]))
    return pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--output", default="experimentos/dataset_splits_thermal_random.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    with open(ORIGINAL_JSON) as f:
        original = json.load(f)

    pools = load_pools(RARITY_CSV)

    train_entries, val_entries = [], []
    for train_u, val_u in zip(original["train"], original["val"]):
        uid = train_u["id"]
        assert val_u["id"] == uid
        n_train = train_u["n_timestamps"]
        n_val = val_u["n_timestamps"]

        pool = pools[uid][:]
        rng.shuffle(pool)
        needed = n_train + n_val
        assert len(pool) >= needed, f"{uid}: pool={len(pool)} < needed={needed}"

        chosen = pool[:needed]
        train_ts = chosen[:n_train]
        val_ts = chosen[n_train:]

        train_entries.append({"id": uid, "timestamps": train_ts, "n_timestamps": len(train_ts)})
        val_entries.append({"id": uid, "timestamps": val_ts, "n_timestamps": len(val_ts)})

    # Test set: identico al original, sin tocar (comparacion contra el mismo test fijo).
    test_entries = original["test"]

    result = {
        "name": "thermal_random_control",
        "description": (
            "Control experiment: mismo numero de muestras train/val por usuario que "
            "dataset_splits_thermal.json (336 train / 112 val, 24+8 por usuario), pero "
            "elegidas uniformemente al azar del pool de timestamps_by_rarity.csv en vez "
            "de por rareza de pose. Test set identico al original (sin modificar), para "
            "comparar de forma controlada contra ablation_no_heatmap (misma loss, mismo "
            "tamanio, solo cambia el metodo de seleccion de datos)."
        ),
        "seed": args.seed,
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }

    tot_train = sum(u["n_timestamps"] for u in train_entries)
    tot_val = sum(u["n_timestamps"] for u in val_entries)
    tot_test = sum(u["n_timestamps"] for u in test_entries)
    print(f"train: {tot_train} timestamps ({len(train_entries)} usuarios)")
    print(f"val:   {tot_val} timestamps ({len(val_entries)} usuarios)")
    print(f"test:  {tot_test} timestamps ({len(test_entries)} usuarios) [identico al original]")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Guardado en {args.output}")


if __name__ == "__main__":
    main()
