#!/usr/bin/env python3
"""Run the baseline queues in tmux and persist progress without retrying failures.

Example:
  python scripts/baseline_suite.py prepare --run-dir logs/baselines_20261003 \
      --python /home/qrh/miniconda3/envs/layer/bin/python
  python scripts/baseline_suite.py launch --run-dir logs/baselines_20261003
  python scripts/baseline_suite.py status --run-dir logs/baselines_20261003

Failures create NEEDS_CONFIRMATION.json and prevent new queued jobs from
starting. Running jobs may finish; nothing is retried or reconfigured.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
MODELS = ("Llama-3.2-1B-Instruct", "Meta-Llama-3-8B-Instruct")
TASKS = ("mmlu", "hellaswag", "winogrande", "gsm8k")
METHODS = ("shortgpt", "sleb", "tale", "none")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def process_start(pid):
    try:
        # Field 22; comm can itself contain spaces or parentheses.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def alive(pid, birth):
    return pid is not None and birth is not None and process_start(pid) == birth


def issue(run_dir, method, message, **details):
    entry = {"time": now(), "method": method, "message": message, **details}
    # One file per producer avoids concurrent failures overwriting each other.
    write_json(run_dir / f"issue_{method}.json", entry)
    # Any producer may set the common pause marker. Issue files remain canonical.
    write_json(run_dir / "NEEDS_CONFIRMATION.json", entry)
    print(json.dumps(entry, ensure_ascii=False), flush=True)


def prepare(args):
    run_dir = args.run_dir
    if (run_dir / "manifest.json").exists():
        raise SystemExit("Manifest already exists; use the existing run or a new directory.")
    python = str(Path(args.python).resolve())
    if not Path(python).is_file():
        raise SystemExit(f"Missing Python: {python}")
    hf_home = os.environ.get("HF_HOME")
    if not hf_home:
        raise SystemExit("HF_HOME must point to the existing local dataset tree.")
    jobs = {}
    output = REPO / "results" / run_dir.name
    for method, gpu in zip(METHODS, args.gpus):
        queue = []
        for model, num_remove in zip(MODELS, (4, 8)):
            command = [python, "-u", str(REPO / "eval.py"), "--model",
                       f"meta-llama/{model}", "--strategy", method,
                       "--tasks", *TASKS, "--local", "--device", "cuda",
                       "--dtype", "auto", "--batch_size", "1", "--seed", "42",
                       "--output", str(output)]
            if method in ("shortgpt", "sleb"):
                command += [f"--{method}_num_remove", str(num_remove)]
            if method == "tale":
                command += ["--tale_search_max_samples", str(args.tale_search_samples),
                            "--tale_threshold", "0.08", "--tale_variant", "threshold_final"]
            queue.append({"id": f"{method}_{model}", "model": model,
                          "command": command,
                          "log": str(run_dir / f"{method}_{model}.log")})
        jobs[method] = {"gpu": gpu, "jobs": queue}
    files = [REPO / "eval.py", *sorted((REPO / "evaluation").rglob("*.py")),
             Path(__file__).resolve()]
    source_hashes = {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in files}
    session = "baselines_" + run_dir.name.removeprefix("baselines_")
    manifest = {"created_at": now(), "session": session, "repo": str(REPO),
                "python": python, "results_dir": str(output), "tasks": TASKS,
                "environment": {"HF_HOME": hf_home, "HF_HUB_OFFLINE": "1",
                                "HF_DATASETS_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                                "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "8",
                                "LAYERSKIP_DISABLE_TQDM": "1"},
                "protocol": {"evaluation_max_samples": None, "prune_ratio": 0.25,
                             "tale_search_max_samples": args.tale_search_samples,
                             "tale_threshold": 0.08, "tale_variant": "threshold_final",
                             "num_fewshot": {"mmlu": 5, "hellaswag": 0,
                                             "winogrande": 0, "gsm8k": 8}},
                "queues": jobs, "source_sha256": source_hashes}
    write_json(run_dir / "manifest.json", manifest)
    run_dir.mkdir(parents=True, exist_ok=True)
    for command, filename in [(["git", "rev-parse", "HEAD"], "git_head.txt"),
                              (["git", "diff"], "source.diff")]:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True)
        (run_dir / filename).write_text(result.stdout)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def launch(args):
    manifest = read_json(args.run_dir / "manifest.json")
    # Fail if code changed since the recorded, reviewable commands were prepared.
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((REPO / relative).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Source changed after preparation: {relative}")
    if (args.run_dir / "NEEDS_CONFIRMATION.json").exists():
        raise SystemExit("Paused; inspect NEEDS_CONFIRMATION.json before continuing.")
    existing = subprocess.run(["tmux", "has-session", "-t", "=" + manifest["session"]],
                              capture_output=True)
    if existing.returncode == 0:
        raise SystemExit("The tmux session already exists; refusing duplicate launch.")
    for method in METHODS:
        if (args.run_dir / f"state_{method}.json").exists():
            raise SystemExit("Queue state already exists; refusing to rerun this directory.")
    for i, method in enumerate(METHODS):
        command = shlex.join([manifest["python"], "-u", str(Path(__file__).resolve()),
                              "worker", "--run-dir", str(args.run_dir), "--method", method])
        if i == 0:
            tmux_command = ["tmux", "new-session", "-d", "-s", manifest["session"],
                            "-n", method, "-c", str(REPO), command]
        else:
            tmux_command = ["tmux", "new-window", "-t", manifest["session"] + ":",
                            "-n", method, "-c", str(REPO), command]
        subprocess.run(tmux_command, check=True)
        if i == 0:
            subprocess.run(["tmux", "set-option", "-t", manifest["session"],
                            "remain-on-exit", "on"], check=True)
    command = shlex.join([manifest["python"], "-u", str(Path(__file__).resolve()),
                          "monitor", "--run-dir", str(args.run_dir)])
    subprocess.run(["tmux", "new-window", "-t", manifest["session"] + ":", "-n", "monitor",
                    "-c", str(REPO), command], check=True)
    print("Started:", manifest["session"], flush=True)


def completed_results(manifest, method, model):
    root = Path(manifest["results_dir"]) / model
    rows = []
    for task in manifest["tasks"]:
        for path in sorted((root / task / method).glob("*.json")):
            result = read_json(path)
            if "results" in result:
                config = result["evaluation_config"]
                rows.append({"model": model, "method": method, "task": task,
                             "metrics": result["results"], "file": str(path),
                             "removed_layers": config["strategy"].get("config", {}).get("selected_layers"),
                             "task_version": config["task"]["version"],
                             "evaluation_max_samples": config["task"]["resolved_kwargs"]["max_samples"]})
    return rows


def worker(args):
    manifest = read_json(args.run_dir / "manifest.json")
    queue = manifest["queues"][args.method]
    env = {**os.environ, **manifest["environment"], "CUDA_VISIBLE_DEVICES": str(queue["gpu"])}
    state_path = args.run_dir / f"state_{args.method}.json"
    state = {"method": args.method, "gpu": queue["gpu"], "status": "starting",
             "worker_pid": os.getpid(), "worker_start": process_start(os.getpid()),
             "started_at": now(), "completed_jobs": []}
    write_json(state_path, state)
    try:
        for job in queue["jobs"]:
            if (args.run_dir / "NEEDS_CONFIRMATION.json").exists():
                state.update(status="paused", updated_at=now())
                write_json(state_path, state)
                return
            state.update(status="running", job=job["id"], model=job["model"],
                         log=job["log"], updated_at=now())
            with Path(job["log"]).open("a", buffering=1) as logfile:
                logfile.write(f"{now()} GPU={queue['gpu']} {shlex.join(job['command'])}\n")
                process = subprocess.Popen(job["command"], cwd=REPO, env=env,
                                           stdout=logfile, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                state.update(pid=process.pid, process_start=process_start(process.pid))
                write_json(state_path, state)
                returncode = process.wait()
            state.update(returncode=returncode, updated_at=now())
            if returncode != 0:
                state["status"] = "failed"
                write_json(state_path, state)
                issue(args.run_dir, args.method, "Experiment exited; confirmation required.",
                      job=job["id"], returncode=returncode, log=job["log"])
                return
            rows = completed_results(manifest, args.method, job["model"])
            if {r["task"] for r in rows} != set(manifest["tasks"]):
                raise RuntimeError(f"{job['id']} exited successfully but task results are incomplete")
            if any(r["evaluation_max_samples"] is not None for r in rows):
                raise RuntimeError(f"{job['id']} used a capped final evaluation")
            state["completed_jobs"].append(job["id"])
            state.pop("pid", None)
            state.pop("process_start", None)
            write_json(state_path, state)
        state.update(status="completed", updated_at=now())
        write_json(state_path, state)
    except Exception as exc:
        state.update(status="failed", error=repr(exc), updated_at=now())
        write_json(state_path, state)
        issue(args.run_dir, args.method, repr(exc), log=state.get("log"))


def snapshot(run_dir):
    manifest = read_json(run_dir / "manifest.json")
    states, traces, results, missing = [], [], [], []
    for method in METHODS:
        path = run_dir / f"state_{method}.json"
        state = read_json(path) if path.exists() else {"method": method, "status": "pending"}
        if state["status"] in ("running", "starting"):
            state["worker_alive"] = alive(state.get("worker_pid"), state.get("worker_start"))
            if state.get("pid"):
                state["process_alive"] = alive(state["pid"], state.get("process_start"))
            if state.get("log"):
                path = Path(state["log"])
                state["log_bytes"] = path.stat().st_size if path.exists() else 0
                if path.exists():
                    with path.open("rb") as handle:
                        handle.seek(max(0, path.stat().st_size - 1600))
                        state["log_tail"] = handle.read().decode(errors="replace").splitlines()[-5:]
        states.append(state)
        for model in MODELS:
            rows = completed_results(manifest, method, model)
            results.extend(rows)
            present = {row["task"] for row in rows}
            missing.extend(f"{model}/{task}/{method}" for task in TASKS if task not in present)
    root = Path(manifest["results_dir"])
    for path in sorted(root.glob("*/pruning/*/*/*.json")):
        envelope = read_json(path)
        trace = envelope["algorithm_trace"]
        progress = trace.get("in_progress_round") or {}
        rounds = trace.get("rounds", [])
        # TALE stores candidates on the current round, SLEB in in_progress_round.
        if rounds and not rounds[-1].get("accepted", "selected_layer" in rounds[-1]):
            progress = rounds[-1]
        candidates = progress.get("candidates", [])
        traces.append({"file": str(path), "model": Path(envelope["model"]).name,
                       "method": envelope["method"], "scope": envelope["scope"],
                       "complete": trace.get("complete", trace.get("completed", False)),
                       "rounds_completed": sum("selected_layer" in r for r in rounds),
                       "current_candidates": len(candidates),
                       "candidates_evaluated": sum(len(r.get("candidates", [])) for r in rounds)
                                               + len((trace.get("in_progress_round") or {}).get("candidates", [])),
                       "removed_layers": trace.get("selected_layers", trace.get("removal_order")),
                       "baseline": trace.get("baseline"),
                       "search_samples": envelope["search_config"].get("search_num_samples")})
    samples = [{"file": str(path), "bytes": path.stat().st_size}
               for path in sorted(root.glob("*/*/*/*.jsonl"))]
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         timeout=10)
    return {"updated_at": now(), "session": manifest["session"], "states": states,
            "traces": traces, "results": results, "missing_results": missing,
            "sample_files": samples, "gpus": gpu.stdout.strip().splitlines(),
            "needs_confirmation": (run_dir / "NEEDS_CONFIRMATION.json").exists()}


def monitor(args):
    print("Monitoring every 30 seconds; failures pause queued work. No automatic retries.", flush=True)
    missing_workers = {}
    while True:
        try:
            value = snapshot(args.run_dir)
            for state in value["states"]:
                method = state["method"]
                if state.get("worker_alive") is False:
                    missing_workers[method] = missing_workers.get(method, 0) + 1
                    if missing_workers[method] == 2:
                        issue(args.run_dir, method, "Worker disappeared; confirmation required.", state=state)
                else:
                    missing_workers[method] = 0
            value["needs_confirmation"] = (args.run_dir / "NEEDS_CONFIRMATION.json").exists()
            write_json(args.run_dir / "status.json", value)
            brief = {"time": value["updated_at"], "queues": {s["method"]: s["status"] for s in value["states"]},
                     "results": len(value["results"]), "expected_results": 32,
                     "needs_confirmation": value["needs_confirmation"]}
            print(json.dumps(brief), flush=True)
            with (args.run_dir / "monitor.jsonl").open("a") as handle:
                handle.write(json.dumps(brief) + "\n")
            if all(s["status"] == "completed" for s in value["states"]):
                write_json(args.run_dir / "COMPLETED.json", value)
                return
            if any(s["status"] == "failed" for s in value["states"]):
                print("NEEDS_CONFIRMATION: inspect issue files and experiment logs.", flush=True)
        except Exception as exc:
            issue(args.run_dir, "monitor", repr(exc))
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "worker", "monitor", "status"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--tale-search-samples", type=int, default=128)
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    args.run_dir = args.run_dir.resolve()
    if len(set(args.gpus)) != 4 or args.tale_search_samples <= 0:
        parser.error("Four distinct GPU IDs and a positive TALE sample count are required.")
    if args.action == "worker" and args.method is None:
        parser.error("worker requires --method")
    if args.action == "status":
        print(json.dumps(snapshot(args.run_dir), indent=2, ensure_ascii=False))
    else:
        globals()[args.action](args)


if __name__ == "__main__":
    main()
