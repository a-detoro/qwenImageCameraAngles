#!/usr/bin/env python
"""
"SLURM casero": daemon que monitoriza las GPUs periódicamente y lanza
trabajos de una cola definida en un JSON editable en caliente.

Diseño:
- La cola (jobs_queue.json) es la fuente de verdad de QUÉ hay que correr.
  Se relee en cada ciclo, así que puedes añadir/activar/desactivar trabajos
  sin reiniciar el daemon.
- El estado (scheduler_state.json) es la fuente de verdad de QUÉ HA PASADO
  (pending/running/done/failed, GPU asignada, PID, exit code). Vive en
  disco, no en memoria del proceso, así que un reinicio del daemon no
  pierde la pista de los trabajos ya lanzados.
- Cada trabajo se lanza DETACHED (nueva sesión de proceso) con
  `start_new_session=True`, así que sigue vivo aunque el scheduler se
  pare o se reinicie. La finalización se detecta de forma robusta con un
  marcador en disco (exit_code.txt), no confiando en que el PID siga
  existiendo (podría haber sido reciclado por el SO).
- Cada GPU es un slot EXCLUSIVO: si el scheduler ya tiene un job "running"
  en la GPU X (según su propio estado), no lanza nada más ahí aunque
  nvidia-smi la muestre con memoria libre en ese instante (evita la
  doble-reserva mientras un modelo grande está cargando y todavía no ha
  alcanzado su pico de VRAM).
- Fallback: si la/s GPU/s preferidas de un job no están libres, prueba el
  resto de GPUs ordenadas por memoria libre descendente.

Uso:
    python gpu_scheduler.py --queue jobs_queue.json --poll-interval 45

Formato de jobs_queue.json:
{
  "jobs": [
    {
      "id": "nombre_unico_del_job",
      "command": "bash run_focus_sweep.sh {gpu}",
      "cwd": "/ruta/absoluta/de/trabajo",
      "preferred_gpus": [2, 0],
      "min_free_mem_mib": 18000,
      "enabled": true,
      "priority": 10
    }
  ]
}

- "{gpu}" en el comando se sustituye por el índice de GPU asignado. Si el
  comando no contiene "{gpu}", el scheduler igualmente exporta
  CUDA_VISIBLE_DEVICES=<gpu> en el entorno del subproceso.
- "priority": menor número = se intenta antes dentro de un mismo ciclo.
- Solo afecta a jobs "pending": si quitas/desactivas un job que ya está
  "running", el scheduler NO lo mata (ver nota de seguridad abajo).
"""
import os
import sys
import json
import time
import signal
import argparse
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timezone

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [scheduler] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

STATE_SCHEMA_VERSION = 1
_shutdown_requested = False


def _handle_signal(signum, frame):
    global _shutdown_requested
    logger.info(f"Señal {signum} recibida, terminando el ciclo actual y saliendo "
                f"(los trabajos ya lanzados SIGUEN corriendo, son procesos independientes)...")
    _shutdown_requested = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# ---------------------------------------------------------------------------
# GPU monitoring
# ---------------------------------------------------------------------------

def query_gpus():
    """Devuelve {idx: {"used_mib": int, "total_mib": int, "free_mib": int, "util_pct": int}}."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        )
    except Exception as e:
        logger.error(f"nvidia-smi falló: {e}")
        return {}

    gpus = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        idx, used, total, util = parts
        idx, used, total, util = int(idx), int(used), int(total), int(util)
        gpus[idx] = {
            "used_mib": used,
            "total_mib": total,
            "free_mib": total - used,
            "util_pct": util,
        }
    return gpus


def log_gpu_status(gpus: dict):
    parts = [f"GPU{idx}: {g['free_mib']}MiB libres/{g['total_mib']}MiB, util {g['util_pct']}%"
              for idx, g in sorted(gpus.items())]
    logger.info("Estado GPUs -> " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Estado persistente
# ---------------------------------------------------------------------------

def load_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"schema_version": STATE_SCHEMA_VERSION, "jobs": {}}


def save_state(state_path: Path, state: dict):
    tmp_path = state_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, default=str)
    tmp_path.replace(state_path)  # escritura atómica


def load_queue(queue_path: Path) -> list:
    if not queue_path.exists():
        return []
    try:
        with open(queue_path) as f:
            data = json.load(f)
        return data.get("jobs", [])
    except Exception as e:
        logger.error(f"No se pudo leer {queue_path}: {e} (se ignora este ciclo)")
        return None  # distinto de [] para no borrar jobs por un JSON mal editado a medias


# ---------------------------------------------------------------------------
# Lógica de scheduling
# ---------------------------------------------------------------------------

def gpus_reserved_by_running_jobs(state: dict) -> set:
    reserved = set()
    for job_state in state["jobs"].values():
        if job_state.get("status") == "running" and job_state.get("gpu") is not None:
            reserved.add(job_state["gpu"])
    return reserved


def pick_gpu_for_job(job_cfg: dict, gpus: dict, reserved: set) -> int | None:
    min_free = job_cfg.get("min_free_mem_mib", 16000)
    preferred = job_cfg.get("preferred_gpus", [])

    # Candidatos: preferidas (en orden declarado) + resto, ordenadas por
    # memoria libre descendente y, en caso de empate, por utilizacion
    # ascendente (menos cargada primero). Asi un job con preferred_gpus=[2]
    # siempre intenta la A30 primero y, si no cabe/esta ocupada, cae a la
    # V100 con mas margen real en vez de a una fija por configuracion.
    others = sorted(
        [idx for idx in gpus if idx not in preferred],
        key=lambda idx: (-gpus[idx]["free_mib"], gpus[idx]["util_pct"]),
    )
    candidates = list(preferred) + others

    for idx in candidates:
        if idx in reserved:
            continue
        info = gpus.get(idx)
        if info is None:
            continue
        if info["free_mib"] >= min_free:
            return idx
    return None


def job_log_dir(logs_root: Path, job_id: str) -> Path:
    d = logs_root / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def launch_job(job_cfg: dict, gpu: int, logs_root: Path) -> dict:
    job_id = job_cfg["id"]
    log_dir = job_log_dir(logs_root, job_id)
    log_path = log_dir / "output.log"
    exit_code_path = log_dir / "exit_code.txt"

    if exit_code_path.exists():
        exit_code_path.unlink()  # relanzamos: limpiar marcador de una corrida anterior

    command = job_cfg["command"].replace("{gpu}", str(gpu))
    cwd = job_cfg.get("cwd", str(logs_root))

    # Se envuelve en bash -c para poder: (a) fijar CUDA_VISIBLE_DEVICES aunque
    # el comando no use {gpu} explícitamente, (b) escribir el exit code real
    # a un marcador en disco, robusto a que el scheduler se reinicie o a que
    # el PID original sea reciclado por el SO.
    wrapped = (
        f"export CUDA_VISIBLE_DEVICES={gpu}; "
        f"{command}; "
        f"echo $? > {exit_code_path}"
    )

    logger.info(f"Lanzando job '{job_id}' en GPU {gpu}: {command}")
    with open(log_path, "a") as log_file:
        log_file.write(f"\n\n===== Lanzado {datetime.now(timezone.utc).isoformat()} en GPU {gpu} =====\n")
        log_file.flush()
        proc = subprocess.Popen(
            ["bash", "-c", wrapped],
            cwd=cwd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # detached: sobrevive a que el scheduler se pare
        )

    return {
        "status": "running",
        "gpu": gpu,
        "pid": proc.pid,
        "command": command,
        "cwd": cwd,
        "log_path": str(log_path),
        "exit_code_path": str(exit_code_path),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "exit_code": None,
    }


def check_running_jobs(state: dict):
    """Actualiza el estado de los jobs 'running' mirando su marcador de exit code."""
    for job_id, job_state in state["jobs"].items():
        if job_state.get("status") != "running":
            continue
        exit_code_path = Path(job_state["exit_code_path"])
        if exit_code_path.exists():
            try:
                code = int(exit_code_path.read_text().strip())
            except Exception:
                code = -1
            job_state["status"] = "done" if code == 0 else "failed"
            job_state["exit_code"] = code
            job_state["finished_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(
                f"Job '{job_id}' terminó con exit_code={code} "
                f"({'OK' if code == 0 else 'FALLO, revisa ' + job_state['log_path']})"
            )


def sync_queue_into_state(queue_jobs: list, state: dict):
    """Añade al estado los jobs nuevos de la cola (no toca los que ya existen)."""
    known_ids = set(state["jobs"].keys())
    for job_cfg in queue_jobs:
        job_id = job_cfg.get("id")
        if not job_id:
            logger.warning(f"Job sin 'id' en la cola, se ignora: {job_cfg}")
            continue
        if job_id not in known_ids:
            state["jobs"][job_id] = {"status": "pending", "gpu": None, "pid": None}
            logger.info(f"Nuevo job detectado en la cola: '{job_id}'")


def run_cycle(queue_path: Path, state_path: Path, logs_root: Path):
    state = load_state(state_path)
    queue_jobs = load_queue(queue_path)
    if queue_jobs is None:
        return  # error de parseo ya logueado, no tocar nada este ciclo

    check_running_jobs(state)
    sync_queue_into_state(queue_jobs, state)

    gpus = query_gpus()
    if gpus:
        log_gpu_status(gpus)

    jobs_by_id = {j["id"]: j for j in queue_jobs if j.get("id")}

    # Jobs pendientes, en orden de prioridad (menor = antes); si no se da,
    # se usa el orden en que aparecen en la cola.
    pending_ids = [
        jid for jid, js in state["jobs"].items()
        if js.get("status") == "pending" and jid in jobs_by_id and jobs_by_id[jid].get("enabled", True)
    ]
    pending_ids.sort(key=lambda jid: jobs_by_id[jid].get("priority", 100))

    reserved = gpus_reserved_by_running_jobs(state)

    for job_id in pending_ids:
        job_cfg = jobs_by_id[job_id]
        gpu = pick_gpu_for_job(job_cfg, gpus, reserved)
        if gpu is None:
            logger.info(f"Job '{job_id}' pendiente: ninguna GPU libre cumple "
                        f"min_free_mem_mib={job_cfg.get('min_free_mem_mib', 16000)} "
                        f"(preferidas={job_cfg.get('preferred_gpus', [])})")
            continue

        job_state = launch_job(job_cfg, gpu, logs_root)
        state["jobs"][job_id] = job_state
        reserved.add(gpu)  # no asignar esta misma GPU a otro job en este mismo ciclo

    save_state(state_path, state)


def print_status_table(state_path: Path):
    state = load_state(state_path)
    if not state["jobs"]:
        print("No hay jobs en el estado todavía.")
        return
    print(f"{'id':<30} {'status':<10} {'gpu':<5} {'pid':<8} {'started_at':<22} {'exit':<5}")
    print("-" * 90)
    for job_id, js in state["jobs"].items():
        print(f"{job_id:<30} {js.get('status',''):<10} {str(js.get('gpu','')):<5} "
              f"{str(js.get('pid','')):<8} {str(js.get('started_at',''))[:22]:<22} "
              f"{str(js.get('exit_code','')):<5}")


def main():
    parser = argparse.ArgumentParser(description="GPU scheduler casero tipo SLURM")
    parser.add_argument("--queue", type=str, default="jobs_queue.json",
                         help="JSON con la cola de trabajos (se relee cada ciclo)")
    parser.add_argument("--state", type=str, default="scheduler_state.json",
                         help="JSON donde se persiste el estado de los jobs")
    parser.add_argument("--logs-root", type=str, default="job_logs",
                         help="Carpeta donde cada job escribe su output.log/exit_code.txt")
    parser.add_argument("--poll-interval", type=int, default=45,
                         help="Segundos entre ciclos de monitorización/scheduling")
    parser.add_argument("--status", action="store_true",
                         help="Imprime el estado actual y termina (no corre el daemon)")
    parser.add_argument("--once", action="store_true",
                         help="Ejecuta un solo ciclo (monitorización + scheduling) y termina. Para pruebas.")
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    queue_path = Path(args.queue)
    if not queue_path.is_absolute():
        queue_path = base_dir / queue_path
    state_path = Path(args.state)
    if not state_path.is_absolute():
        state_path = base_dir / state_path
    logs_root = Path(args.logs_root)
    if not logs_root.is_absolute():
        logs_root = base_dir / logs_root
    logs_root.mkdir(parents=True, exist_ok=True)

    if args.status:
        print_status_table(state_path)
        return

    logger.info(f"Scheduler arrancado. cola={queue_path} estado={state_path} "
                f"logs={logs_root} poll_interval={args.poll_interval}s")

    if args.once:
        run_cycle(queue_path, state_path, logs_root)
        return

    while not _shutdown_requested:
        try:
            run_cycle(queue_path, state_path, logs_root)
        except Exception as e:
            logger.exception(f"Error en el ciclo del scheduler (se continúa en el siguiente): {e}")
        for _ in range(args.poll_interval):
            if _shutdown_requested:
                break
            time.sleep(1)

    logger.info("Scheduler parado. Los jobs ya lanzados siguen corriendo de forma independiente.")


if __name__ == "__main__":
    main()
