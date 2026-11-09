"""
Genera un curriculum térmico AMPLIADO (más muestras que las 336 train / 112 val
originales), manteniendo la misma metodología de selección: muestreo uniforme
estratificado a lo largo del eje de rareza de pose, por usuario.

Fuente: notebooks/timestamps_by_rarity.csv (user_id,timestamp,knn_score,rank,split),
el mismo CSV del que salio el curriculum original (ver investigacion previa:
rank=1 es la pose mas rara para ese usuario).

Algoritmo (version limpia, documentada; el original exacto no se encontro en el
repo, se reconstruyo por ingenieria inversa sobre el JSON ya publicado):
- Usuarios train/val (split=train_val en el CSV): se eligen K anclas con
  espaciado uniforme cubriendo TODO el rango de rangos [1, N_usuario]
  (linspace inclusive, llega hasta la pose mas rara). Cada ancla -> val.
  Sus vecinos inmediatos (ancla-1, ancla+1, recortados a [1,N] y sin duplicar)
  -> train. Con K anclas y espaciado >> 1, no hay solape train/val.
- Usuarios test (split=test en el CSV): se eligen M anclas con el mismo
  esquema, usadas directamente como conjunto de test (sin dividir).

K y M se escalan respecto a los originales (8 y 56) para controlar cuanto se
amplia el curriculum.
"""
import csv
import json
import argparse
from collections import defaultdict

RARITY_CSV = "/nas/antoniodetoro/qwen/Qwen-Image-Edit-Angles-2/notebooks/timestamps_by_rarity.csv"


def load_rarity_csv(path):
    by_user = defaultdict(list)
    user_split = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            uid = row["user_id"]
            by_user[uid].append((int(row["rank"]), int(row["timestamp"])))
            user_split[uid] = row["split"]
    for uid in by_user:
        by_user[uid].sort(key=lambda x: x[0])  # ordenar por rank ascendente
    return by_user, user_split


def pick_anchors(ranked_timestamps, k):
    """ranked_timestamps: lista (rank, timestamp) ordenada por rank asc, rank en [1,N].
    Devuelve k timestamps con rank uniformemente espaciado en [1, N] (linspace inclusive)."""
    n = len(ranked_timestamps)
    k = min(k, n)
    if k <= 1:
        return [ranked_timestamps[0][1]]
    rank_to_ts = {r: ts for r, ts in ranked_timestamps}
    anchors = []
    for i in range(k):
        target_rank = round(1 + i * (n - 1) / (k - 1))
        anchors.append(target_rank)
    anchors = sorted(set(anchors))
    return [rank_to_ts[r] for r in anchors], set(anchors)


def build_train_val_user(ranked_timestamps, k_val, train_neighbor_radius):
    n = len(ranked_timestamps)
    rank_to_ts = {r: ts for r, ts in ranked_timestamps}
    val_ts, val_ranks = pick_anchors(ranked_timestamps, k_val)

    train_ranks = set()
    for r in val_ranks:
        for offset in range(1, train_neighbor_radius + 1):
            for neighbor in (r - offset, r + offset):
                if 1 <= neighbor <= n and neighbor not in val_ranks:
                    train_ranks.add(neighbor)

    train_ts = [rank_to_ts[r] for r in sorted(train_ranks)]
    return train_ts, val_ts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_val", type=int, default=16,
                         help="Anclas de val por usuario train/val (original: 8)")
    parser.add_argument("--m_test", type=int, default=112,
                         help="Anclas de test por usuario test (original: 56)")
    parser.add_argument("--train_neighbor_radius", type=int, default=1,
                         help="Radio de vecinos de rank cosechados para train por cada ancla de val (original: 1)")
    parser.add_argument("--output", type=str,
                         default="experimentos/dataset_splits_thermal_expanded.json")
    args = parser.parse_args()

    by_user, user_split = load_rarity_csv(RARITY_CSV)

    train_entries, val_entries, test_entries = [], [], []

    for uid, ranked in by_user.items():
        if user_split[uid] == "train_val":
            train_ts, val_ts = build_train_val_user(ranked, args.k_val, args.train_neighbor_radius)
            train_entries.append({"id": uid, "timestamps": train_ts, "n_timestamps": len(train_ts)})
            val_entries.append({"id": uid, "timestamps": val_ts, "n_timestamps": len(val_ts)})
        else:  # test
            test_ts, _ = pick_anchors(ranked, args.m_test)
            test_entries.append({"id": uid, "timestamps": test_ts, "n_timestamps": len(test_ts)})

    result = {
        "name": "thermal_expanded",
        "description": (
            f"Curriculum termico ampliado: k_val={args.k_val} anclas/usuario train-val "
            f"(original 8), train_neighbor_radius={args.train_neighbor_radius} (original 1), "
            f"m_test={args.m_test} anclas/usuario test (original 56). "
            "Mismo metodo de muestreo uniforme por rareza de pose que el original, "
            "ver build_expanded_curriculum.py para el algoritmo exacto."
        ),
        "train": train_entries,
        "val": val_entries,
        "test": test_entries,
    }

    tot_train = sum(u["n_timestamps"] for u in train_entries)
    tot_val = sum(u["n_timestamps"] for u in val_entries)
    tot_test = sum(u["n_timestamps"] for u in test_entries)
    print(f"train: {tot_train} timestamps ({len(train_entries)} usuarios)")
    print(f"val:   {tot_val} timestamps ({len(val_entries)} usuarios)")
    print(f"test:  {tot_test} timestamps ({len(test_entries)} usuarios)")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Guardado en {args.output}")


if __name__ == "__main__":
    main()
