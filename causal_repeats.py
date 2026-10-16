#!/usr/bin/env python3
"""Opt-in repeated C1--C4 causal policy workflow. See README_CAUSAL_REPEATS.md."""
from __future__ import annotations
import argparse
import copy
import csv
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import signal
import subprocess
import sys

VERSION = "1.0.0"
MODES = ("ground_truth", "partial_observation")
SPLITS = ("train", "validation", "test")
FACTORS = ("resistant_transmission", "resistant_admission_weights", "turnover")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@contextmanager
def lock(path, wait=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+b")
    try:
        if os.name == "nt":
            import msvcrt
            f.seek(0)
            if not f.read(1):
                f.write(b"0")
                f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK if wait else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | (0 if wait else fcntl.LOCK_NB))
    except OSError:
        f.close()
        raise RuntimeError(f"Another process is using {path.parent}")
    try:
        yield
    finally:
        f.close()


def run_command(command, repo, log, seed, threads, deterministic, context=None):
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("DT_CAUSAL_REPEAT_CONTEXT", "DT_PT_OUT_DIR", "DT_STATE_MODE", "DT_CONVERT_EXTRA_ARGS", "DT_SIM_EXTRA_ARGS"):
        env.pop(key, None)
    env.update(PYTHONHASHSEED=str(seed), MPLBACKEND="Agg", OMP_NUM_THREADS=str(threads),
               MKL_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
               VECLIB_MAXIMUM_THREADS=str(threads))
    if deterministic:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if context is not None:
        env["DT_CAUSAL_REPEAT_CONTEXT"] = str(context)
    command = [str(x) for x in command]
    print(f"[run] {shlex.join(command)}\n      log: {log}", flush=True)
    with log.open("a", encoding="utf-8") as f:
        f.write("\nCOMMAND " + shlex.join(command) + "\n")
        f.flush()
        proc = subprocess.Popen(command, cwd=repo, env=env, stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=(os.name != "nt"))
        try:
            while True:
                try:
                    code = proc.wait(timeout=60)
                    break
                except subprocess.TimeoutExpired:
                    print(f"[running] {log.name}", flush=True)
        except BaseException:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    if code:
        raise RuntimeError(f"Stage failed (exit {code}); inspect {log}. Completed prior stages are retained.")


def int_spec(value):
    text = str(value).strip()
    if ":" in text:
        parts = list(map(int, text.split(":")))
        if len(parts) not in (2, 3):
            raise ValueError(f"Invalid range: {value}")
        lo, hi = parts[:2]
        step = parts[2] if len(parts) == 3 else 1
        if step <= 0 or hi < lo:
            raise ValueError(f"Invalid range: {value}")
        return list(range(lo, hi + 1, step))
    return [int(x) for x in text.split(",") if x.strip()]


def validate_config(cfg, driver):
    required = {"simulation_seed_offsets", "model_seeds", "parameter_seed", "parameter_multipliers",
                "parameter_rationale", "tracks", "shared_noise_seed_offset", "bootstrap_replicates",
                "bootstrap_seed", "deterministic", "threads", "dataset_jobs", "conversion_workers",
                "run_milp", "keep_raw_graphml", "keep_action_full_pt", "train_overrides", "milp_overrides"}
    if set(cfg) - required - {"description", "draws"} or required - set(cfg):
        raise ValueError(f"Unknown/missing configuration keys: extra={set(cfg)-required-{'description','draws'}}, missing={required-set(cfg)}")
    n = len(cfg["model_seeds"])
    if n < 2 or len(cfg["simulation_seed_offsets"]) != n:
        raise ValueError("Require at least two repeats, with equally sized seed lists")
    for key in ("simulation_seed_offsets", "model_seeds"):
        if len(set(cfg[key])) != n or any(type(v) is not int or v < 0 or v + 1000000 >= 2**32 for v in cfg[key]):
            raise ValueError(f"{key} must contain distinct nonnegative seeds/offsets within uint32 range")
    if not cfg["tracks"] or len(set(cfg["tracks"])) != len(cfg["tracks"]) or set(cfg["tracks"]) - set(MODES):
        raise ValueError("Select each requested observation track once")
    for key in ("deterministic", "run_milp", "keep_raw_graphml", "keep_action_full_pt"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be a JSON boolean")
    for key in ("threads", "dataset_jobs", "bootstrap_replicates"):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("conversion_workers", "parameter_seed", "bootstrap_seed", "shared_noise_seed_offset"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    if not str(cfg["parameter_rationale"]).strip() or set(cfg["parameter_multipliers"]) != set(FACTORS):
        raise ValueError("Specify all parameter ranges and their scientific source or sensitivity rationale")
    for key, bounds in cfg["parameter_multipliers"].items():
        if len(bounds) != 2 or not all(math.isfinite(float(v)) for v in bounds) or not 0 < float(bounds[0]) < float(bounds[1]):
            raise ValueError(f"{key} requires finite positive [low, high], with low < high")
    if "draws" in cfg:
        if len(cfg["draws"]) != n or any(set(v) != set(FACTORS) for v in cfg["draws"]):
            raise ValueError("Explicit draws must specify all factors once per repeat")
        if any(not math.isfinite(float(x)) or float(x) <= 0 for row in cfg["draws"] for x in row.values()):
            raise ValueError("Explicit multipliers must be finite and positive")
    ds = driver.CONFIG["DATASET"]
    if ds.get("use_balanced_base_families"):
        raise ValueError("Repeat mode currently preserves the seed-based causal design; balanced-family experiments use the legacy driver")
    if ds["action_start_mode"] != "branch_at_decision_day" or ds["decision_applies_from"] != "next_day":
        raise ValueError("Repeat mode requires the next-day counterfactual branching design")
    if ds["window_T"] != 7 or int_spec(ds["horizons"]) != [7] or ds["sim_args"]["num_regions"] != 1 or not ds["include_baseline"]:
        raise ValueError("Repeat mode preserves T=7/H=7, one region and the baseline action")


def build_design(driver, cfg, repo):
    dataset = copy.deepcopy(driver.CONFIG["DATASET"])
    train = driver._load_best_train_config_override(driver.CONFIG)
    allowed = set(train) | {"use_cls", "split_seed", "early_stopping", "patience", "min_delta", "save_best_only",
                            "lr_scheduler_on_plateau", "lr_scheduler_factor", "lr_scheduler_patience", "lr_scheduler_min_lr"}
    if set(cfg["train_overrides"]) - allowed:
        raise ValueError("Unsupported train_overrides keys")
    train.update(cfg["train_overrides"])
    if train["task"] != "oracle_best_action_h7" or train["T"] != 7 or train["sliding_step"] != 1 or not train["require_complete_action_set"]:
        raise ValueError("Repeat mode retains the original grouped policy task and complete action menu")
    train.update(cpu_threads=cfg["threads"], deterministic=cfg["deterministic"], metrics_only=True,
                 emit_translational_figures=False, fullgraph_attribution_pass=False, train_model=True)
    milp = copy.deepcopy(driver.CONFIG["MILP"])
    if set(cfg["milp_overrides"]) - set(milp):
        raise ValueError("Unsupported MILP override")
    milp.update(cfg["milp_overrides"])
    if str(milp.get("seed", "")).strip():
        raise ValueError("Repeat MILP evaluates every test trajectory; remove the single-seed MILP filter")
    candidate_path = driver._candidate_interventions_json(repo)
    baseline_path = driver._baseline_interventions_json(repo)
    actions = read_json(candidate_path)
    n_candidates = len(actions) if isinstance(actions, list) else len(actions.get("interventions", []))
    if n_candidates < 1:
        raise ValueError("Empty/unrecognized candidate intervention menu")
    original_seeds = {split: int_spec(dataset[key]) for split, key in
                      (("train", "train_seeds"), ("validation", "val_seeds"), ("test", "test_seeds"))}
    all_seed_values = [v for values in original_seeds.values() for v in values]
    if any(not v for v in original_seeds.values()) or len(set(all_seed_values)) != len(all_seed_values):
        raise ValueError("Train/validation/test simulator seed sets must be nonempty and disjoint")
    num_days = int(dataset["sim_args"]["num_days"])
    if str(dataset["decision_days"]).lower() in {"auto", "all", ""}:
        decision_days = list(range(7, num_days - 7 + 1, int(dataset["decision_stride"])))
    else:
        decision_days = [d for d in int_spec(dataset["decision_days"]) if 7 <= d <= num_days - 7]
    if not decision_days:
        raise ValueError("No complete decision windows/horizons")
    repeats, used = [], set()
    for r, (offset, seed) in enumerate(zip(cfg["simulation_seed_offsets"], cfg["model_seeds"]), 1):
        rng = random.Random(cfg["parameter_seed"] + r)
        draw = copy.deepcopy(cfg["draws"][r-1]) if "draws" in cfg else {
            key: rng.uniform(*map(float, cfg["parameter_multipliers"][key])) for key in FACTORS}
        ds = copy.deepcopy(dataset)
        ds["force"], ds["jobs"] = False, cfg["dataset_jobs"]
        seeds = {k: [v + offset for v in values] for k, values in original_seeds.items()}
        flattened = [v for values in seeds.values() for v in values]
        if used.intersection(flattened) or any(v + cfg["shared_noise_seed_offset"] >= 2**32 for v in flattened):
            raise ValueError("Repeat simulator seeds overlap or exceed the supported range")
        used.update(flattened)
        for split, key in (("train", "train_seeds"), ("validation", "val_seeds"), ("test", "test_seeds")):
            ds[key] = ",".join(map(str, seeds[split]))
        sim = ds["sim_args"]
        sim["beta_res_mult"] *= draw["resistant_transmission"]
        for key in ("p_admit_import_cr", "p_admit_import_ir"):
            sim[key] *= draw["resistant_admission_weights"]
        sim["daily_discharge_frac"] *= draw["turnover"]
        if not 0 <= sim["daily_discharge_frac"] <= 1:
            raise ValueError("Turnover draw outside [0, 1]")
        reps_train = copy.deepcopy(train)
        reps_train.update(model_seed=seed, eval_seed=seed + 1000000)
        repeats.append({"repeat": r, "model_seed": seed, "eval_seed": seed + 1000000,
                        "parameter_seed": cfg["parameter_seed"] + r, "parameter_multipliers": draw,
                        "simulator_seeds": seeds, "decision_days": decision_days, "dataset": ds,
                        "train": reps_train, "milp": milp,
                        "n_base_trajectories": len(flattened),
                        "n_counterfactual_arms": len(flattened) * len(decision_days) * n_candidates,
                        "n_actions": n_candidates + 1})
    return {"repeats": repeats, "candidate_interventions": actions,
            "candidate_path": str(candidate_path),
            "baseline_intervention": read_json(baseline_path) if baseline_path else None,
            "milp_config": read_json(driver._milp_config_json(repo))}


def task_complete(folder, fingerprint, files):
    marker = folder / "complete.json"
    if not marker.is_file():
        return False
    saved = read_json(marker)
    if saved.get("fingerprint") != fingerprint:
        raise RuntimeError(f"Completed stage inputs changed: {folder}; choose a new output directory")
    return all((folder / n).is_file() and file_hash(folder / n) == saved.get("files", {}).get(n) for n in files)


def mark_complete(folder, fingerprint, files):
    atomic_json(folder / "complete.json", {"fingerprint": fingerprint,
                "files": {name: file_hash(folder / name) for name in files}})


def verify_dataset(folder, rep):
    rows = read_csv(folder / "policy_manifest.csv")
    expected_states = {(split, seed, day) for split, seeds in rep["simulator_seeds"].items()
                       for seed in seeds for day in rep["decision_days"]}
    grouped = {}
    target_lookup = {}
    for row in rows:
        key = (row["split"], int(row["seed"]), int(row["decision_day"]))
        grouped.setdefault(key, []).append(row)
    if set(grouped) != expected_states:
        raise RuntimeError(f"Missing/extra decision states in {folder}")
    actions = {a["action_id"] for a in read_json(folder / "dataset_config.json")["actions"]}
    if len(actions) != rep["n_actions"]:
        raise RuntimeError("Action-menu size changed")
    for key, group in grouped.items():
        if len(group) != len(actions) or {r["action_id"] for r in group} != actions:
            raise RuntimeError(f"Incomplete/duplicate action menu for {key}")
        windows = [json.loads(r["window_pt_json"]) for r in group]
        if any(w != windows[0] for w in windows) or len(windows[0]) != 7:
            raise RuntimeError(f"Actions have different past windows for {key}")
        if any(int(r["action_start_day"]) != key[2]+1 or int(r["policy_valid"]) != 1 for r in group):
            raise RuntimeError(f"Invalid intervention timing for {key}")
        for row in group:
            needed = [*json.loads(row["window_pt_json"]), row["decision_pt_path"]]
            if not all(Path(p).is_file() for p in needed):
                raise RuntimeError(f"Missing PT input or target for {row['pair_id']}")
            target_lookup[(key, row["action_id"])] = {k: v for k, v in row.items()
                if k.startswith(("y_h", "oracle_best_action_", "oracle_tie_count_", "is_oracle_best_"))}
    # The verified arm marker records actual file integrity; paths in a manifest
    # alone are not proof of a complete reusable dataset.
    from causal_repeat_cache import validate_marker
    arm_dirs = {Path(r["baseline_pt_dir"]) for r in rows} | {Path(r["action_pt_dir"]) for r in rows}
    for arm in arm_dirs:
        if not validate_marker(arm, "arm_complete.json"):
            raise RuntimeError(f"Incomplete/corrupted arm: {arm}")
    return digest([{ "key": [*k[0], k[1]], "targets": v} for k, v in sorted(target_lookup.items())])


def run_repeat(rep, cfg, design, driver, repo, output, context):
    root = output / f"repeat_{rep['repeat']:03d}"
    original = copy.deepcopy(driver.CONFIG)
    target_identity = None
    try:
        for mode in cfg["tracks"]:
            track = root / f"TRACK_{mode}"
            work = track / "work"
            paths = {"dataset_root": work / rep["dataset"]["out_name"],
                     "train_root": work / rep["train"]["out_name"],
                     "eval_root": work / "policy_eval_test", "milp_root": work / rep["milp"]["out_name"]}
            driver.CONFIG["DATASET"], driver.CONFIG["TRAIN"], driver.CONFIG["MILP"] = rep["dataset"], rep["train"], rep["milp"]
            ds = paths["dataset_root"]
            ds_fp = digest({"dataset": rep["dataset"], "mode": mode, "actions": design["candidate_interventions"]})
            ds_files = ["policy_manifest.csv", "dataset_config.json", "dataset_summary.json"]
            ready = task_complete(ds, ds_fp, ds_files)
            if ready:
                try:
                    targets = verify_dataset(ds, rep)
                except (OSError, ValueError, RuntimeError):
                    ready = False
            if not ready:
                cmd = driver._build_dataset_cmd(repo, paths)
                cmd[cmd.index("--state_mode")+1] = mode
                pos = cmd.index("--")
                cmd[pos:pos] = ["--shared_noise_seed_offset", str(cfg["shared_noise_seed_offset"])]
                run_command(cmd, repo, output / "logs" / f"repeat_{rep['repeat']:03d}_{mode}_C1.log",
                            rep["model_seed"], cfg["threads"], cfg["deterministic"], context)
                targets = verify_dataset(ds, rep)
                mark_complete(ds, ds_fp, ds_files)
            if target_identity is not None and targets != target_identity:
                raise RuntimeError("Ground-truth/partial-observation action outcomes or state identities differ")
            target_identity = targets
            train = paths["train_root"]
            train_fp = digest({"manifest": file_hash(ds / "policy_manifest.csv"), "model": rep["train"], "dataset": ds_fp})
            train_files = ["trained_model.pt", "metrics_summary.json"]
            if not task_complete(train, train_fp, train_files):
                cmd = driver._build_train_cmd(paths, train_cfg_override=rep["train"])
                run_command(cmd, repo, output / "logs" / f"repeat_{rep['repeat']:03d}_{mode}_C2.log",
                            rep["model_seed"], cfg["threads"], cfg["deterministic"])
                mark_complete(train, train_fp, train_files)
            for split in SPLITS:
                ev = work / f"policy_eval_{split}"
                ev_fp = digest({"train": train_fp, "checkpoint": file_hash(train / "trained_model.pt"),
                                "eval": driver.CONFIG["EVAL"], "split": split, "eval_seed": rep["eval_seed"]})
                ev_files = ["policy_selection_summary.json", "policy_selection_per_state_requested_split.csv",
                            "policy_action_scores_requested_split.csv"]
                if not task_complete(ev, ev_fp, ev_files):
                    cmd = [sys.executable, "evaluate_policy_selector.py", "--data_folder", str(ds),
                           "--trained_dir", str(train), "--split", split, "--out_dir", str(ev),
                           "--oracle_metric", str(driver.CONFIG["EVAL"]["oracle_metric"]),
                           "--selection_direction", str(driver.CONFIG["EVAL"]["selection_direction"]),
                           "--candidate_interventions_json", design["candidate_path"],
                           "--require_complete_action_set", "true", "--emit_action_scores_csv", "true",
                           "--eval_seed", str(rep["eval_seed"]), "--cpu_threads", str(cfg["threads"]),
                           "--deterministic", str(cfg["deterministic"]).lower(), "--metrics_only", "true"]
                    run_command(cmd, repo, output / "logs" / f"repeat_{rep['repeat']:03d}_{mode}_C3_{split}.log",
                                rep["eval_seed"], cfg["threads"], cfg["deterministic"])
                    summary = read_json(ev / "policy_selection_summary.json")
                    n_states = len(rep["simulator_seeds"][split]) * len(rep["decision_days"])
                    if summary["n_states"] != n_states or summary["dropped_states"]:
                        raise RuntimeError(f"Evaluator omitted states: {ev}")
                    mark_complete(ev, ev_fp, ev_files)
            if cfg["run_milp"]:
                milp = paths["milp_root"]
                fp = digest({"scores": file_hash(paths["eval_root"] / "policy_action_scores_requested_split.csv"),
                             "settings": rep["milp"], "constraints": design["milp_config"]})
                files = ["milp_policy_plan.csv", "milp_policy_summary.json"]
                if not task_complete(milp, fp, files):
                    cmd = driver._build_milp_cmd(repo, paths)
                    run_command(cmd, repo, output / "logs" / f"repeat_{rep['repeat']:03d}_{mode}_C4.log",
                                rep["model_seed"], cfg["threads"], cfg["deterministic"])
                    mark_complete(milp, fp, files)
        atomic_json(root / "repeat_complete.json", {"repeat": rep["repeat"], "target_identity": target_identity})
    finally:
        driver.CONFIG.clear()
        driver.CONFIG.update(original)


def dependency_report(cfg, design):
    required = ["numpy", "scipy", "sklearn", "torch", "torch_geometric", "networkx", "matplotlib"]
    if cfg["run_milp"] and design["repeats"][0]["milp"]["solver"] != "scipy":
        required.append("pyomo")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    notes = []
    if cfg["run_milp"]:
        milp = design["repeats"][0]["milp"]
        if milp["solver"] == "cplex":
            executable = Path(str(milp["cplex_executable"])).expanduser()
            if not executable.is_file():
                notes.append(f"Configured CPLEX executable is unavailable here: {executable}")
    return missing, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat_config", default=str(Path(__file__).with_name("causal_repeats.json")))
    parser.add_argument("--repeat_output", default="causal_repeat_results")
    parser.add_argument("--reuse_root", action="append", default=[], help="Compatible previous causal repeat result/cache roots, read-only")
    parser.add_argument("--repeat_ids", help="Comma-separated 1-based repeat IDs; omit to run/resume all")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--summarize_only", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parent
    output = Path(args.repeat_output).expanduser().resolve()
    cfg = read_json(Path(args.repeat_config).expanduser())
    driver = load_module(repo / "experiments_causal.py", "_causal_repeat_driver")
    validate_config(cfg, driver)
    design = build_design(driver, cfg, repo)
    sources = {p.name: file_hash(p) for p in sorted(repo.glob("*.py"))}
    for name in ("best_config.json", "candidate_interventions.json", "milp_policy_config.json"):
        sources[name] = file_hash(repo / name)
    identity = {"version": VERSION, "config": cfg, "driver_config": driver.CONFIG, "design": design, "sources": sources}
    identity_sha = digest(identity)
    manifest = output / "run_manifest.json"
    if manifest.is_file() and read_json(manifest)["identity_sha256"] != identity_sha:
        raise RuntimeError("Output belongs to a different source/configuration. Choose a new --repeat_output.")
    selected = set(range(1, len(design["repeats"]) + 1))
    if args.repeat_ids:
        selected = set(map(int, args.repeat_ids.split(",")))
        if not selected or not selected <= set(range(1, len(design["repeats"]) + 1)):
            raise ValueError("Unknown repeat_ids")
    roots = [Path(p).expanduser().resolve() for p in args.reuse_root]
    for root in roots:
        if not root.is_dir() or output == root or output in root.parents or root in output.parents:
            raise ValueError("Reuse roots must exist and be separate from the output directory")
    if args.summarize_only:
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        from summarize_causal_repeats import summarize
        with lock(output / "run.lock"):
            summarize(output)
        return 0
    missing, notes = dependency_report(cfg, design)
    reps = [r for r in design["repeats"] if r["repeat"] in selected]
    from causal_repeat_cache import plan_cache_status
    plan = {"repeats": len(reps), "tracks": cfg["tracks"],
            "base_trajectories": sum(r["n_base_trajectories"] for r in reps),
            "counterfactual_arms": sum(r["n_counterfactual_arms"] for r in reps),
            "gnn_fits": len(reps) * len(cfg["tracks"]),
            "milp_solves": len(reps) * len(cfg["tracks"]) if cfg["run_milp"] else 0,
            "missing_python_modules": missing, "solver_notes": notes,
            "cache": plan_cache_status(output, roots),
            "note": "Counts are the full selected design before exact cache matching. New simulator seeds/draws generally need new rollouts. Raw rollouts are shared across observation tracks."}
    print(json.dumps(plan, indent=2), flush=True)
    if args.preflight:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "preflight.json", {"plan": plan, "identity": identity})
        return 0
    if missing or notes:
        raise RuntimeError(f"Resolve dependencies/solver configuration before starting: {missing}; {notes}")
    with lock(output / "run.lock"):
        import importlib.metadata as metadata
        import torch
        versions = {}
        for name in ("numpy", "scipy", "scikit-learn", "torch", "torch-geometric", "networkx", "pyomo"):
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                pass
        runtime = {"python": sys.version, "packages": versions, "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}
        runtime_path = output / "runtime.json"
        if runtime_path.is_file() and read_json(runtime_path) != runtime:
            raise RuntimeError("Runtime/software/device changed. Use a new output directory.")
        for root in roots:
            previous_runtime = root / "runtime.json"
            if not previous_runtime.is_file() and root.name == "arm_cache":
                previous_runtime = root.parent / "runtime.json"
            if not previous_runtime.is_file() or read_json(previous_runtime) != runtime:
                raise RuntimeError(f"Reuse root has a missing/different runtime record: {root}. Remove this --reuse_root to generate new data.")
        atomic_json(runtime_path, runtime)
        atomic_json(manifest, {"identity_sha256": identity_sha, "identity": identity})
        snapshot = output / "source_snapshot"
        snapshot.mkdir(exist_ok=True)
        for name in sources:
            shutil.copy2(repo / name, snapshot / name)
        atomic_json(snapshot / "causal_repeats.json", cfg)
        context = output / "cache_context.json"
        atomic_json(context, {"cache_root": str(output / "arm_cache"), "reuse_roots": [str(r) for r in roots],
                             "tracks": cfg["tracks"], "keep_raw_graphml": cfg["keep_raw_graphml"],
                             "keep_action_full_pt": cfg["keep_action_full_pt"],
                             "conversion_workers": cfg["conversion_workers"],
                             "baseline_intervention": design["baseline_intervention"],
                             "sources": sources})
        for rep in reps:
            print(f"[repeat {rep['repeat']}/{len(design['repeats'])}] {rep['parameter_multipliers']}", flush=True)
            run_repeat(rep, cfg, design, driver, repo, output, context)
            from summarize_causal_repeats import summarize
            summarize(output)
    print(f"Complete. Results: {output / 'summary'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
