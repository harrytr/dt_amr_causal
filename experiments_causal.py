#!/usr/bin/env python3
from __future__ import annotations

"""
experiments_causal.py

Driver for the intervention-conditioned causal policy experiment.

This script orchestrates five stages:
  STEP C1: build the causal policy dataset
  STEP C2: train the action-conditioned temporal GNN
  STEP C3: evaluate policy selection on a held-out split
  STEP C4: optimize an intervention plan with a MILP over scored candidate actions
  STEP C5: quantitatively evaluate cross-state score calibration and constrained allocation

It also supports dual-track execution (ground_truth / partial_observation) and
an Overleaf export similar in spirit to the pbc7 driver.
"""

import argparse
import datetime as _dt
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import signal
_ACTIVE_CHILD_PROCS: List[subprocess.Popen[Any]] = []


def _load_best_train_config_override(config: Dict[str, Any]) -> Dict[str, Any]:
    train_cfg = dict(config.get("TRAIN", {}))

    best_cfg_path = Path(__file__).resolve().parent / "best_config.json"
    if not best_cfg_path.exists():
        print("No best_config.json found beside experiments_causal.py; using CONFIG['TRAIN'] as-is.", flush=True)
        return train_cfg

    with open(best_cfg_path, "r", encoding="utf-8") as f:
        best_cfg = json.load(f)

    if not isinstance(best_cfg, dict):
        raise ValueError(f"best_config.json must contain a JSON object, got: {type(best_cfg)}")

    overridden_keys = []
    ignored_keys = []

    for k, v in best_cfg.items():
        if k in train_cfg:
            train_cfg[k] = v
            overridden_keys.append(str(k))
        else:
            ignored_keys.append(str(k))

    print(f"Found best_config.json and overriding TRAIN params from: {best_cfg_path}", flush=True)
    if overridden_keys:
        print("Overridden TRAIN keys: " + ", ".join(sorted(overridden_keys)), flush=True)
    if ignored_keys:
        print("Ignored keys not present in CONFIG['TRAIN']: " + ", ".join(sorted(ignored_keys)), flush=True)

    return train_cfg
  
  

# =============================================================================
# CONFIG (EDIT HERE)
# =============================================================================
CONFIG: Dict[str, Any] = {
    "DATASET": {
        "out_name": "causal_policy_monthly_shock_superspreader",
        "candidate_interventions_json": "candidate_interventions.json",
        "baseline_intervention_json": "",
        "include_baseline": True,

        "use_balanced_base_families": False,

        "train_seeds": "4000:4019",
        "val_seeds": "5000:5003",
        "test_seeds": "6000:6009",

        "window_T": 7,
        "horizons": "7",
        "decision_days": "7:60:7",
        "decision_stride": 1,
        "action_start_mode": "branch_at_decision_day",
        "decision_applies_from": "next_day",
        "force": True,
        "require_two_class_splits": False,
        "jobs": 4,

        "sim_args": {
            "num_days": 60,
            "num_regions": 1,
            "num_wards": 5,
            "num_patients": 50,
            "num_staff": 100,
            "staff_wards_per_staff": 2,
            "export_yaml": False,
            "export_gif": False,

            "isolation_mult": 0.8,
            "isolation_days": 3,

            # Day-0 seeding is explicit so the driver, rather than simulator
            # defaults, defines the initial CS/CR/IS/IR burden.
            "p_import_cs": 0.3,
            "p_import_cr": 0.1,
            "p_import_is": 0.3,
            "p_import_ir": 0.1,

            "p_admit_import_cs": 0.3,
            "p_admit_import_cr": 0.4,
            "p_admit_import_is": 0.3,
            "p_admit_import_ir": 0.4,
            
            "daily_discharge_frac": 0.30,
            "daily_discharge_min_per_ward": 3,

            "screen_every_k_days": 7,
            "screen_on_admission": 0,
            "screen_result_delay_days": 3,
            "persist_observations": 1,
            "beta_res_mult": 1.0,
            "beta_state_mult_cs": 0.5,
            "beta_state_mult_cr": 0.5,
            "beta_state_mult_is": 1.0,
            "beta_state_mult_ir": 1.0,
            "crossward_hand_hygiene_enabled": 1,
            "crossward_hand_hygiene_mult": 0.55,
            "passive_screen_capacity_per_week": 0,
            "active_symptom_screening": 1,
            "admission_screen_symptomatic_only": 1,
            "isolation_trigger_policy": "resistant_only",

            "admit_import_seasonality": "shock",
            "admit_import_shock_min_days": 20,
            "admit_import_shock_max_days": 30,
            "admit_import_shock_mult_min": 2.0,
            "admit_import_shock_mult_max": 3.0,
            "admit_import_pmax_cs": 1.0,
            "admit_import_pmax_cr": 1.0,
            "admit_import_pmax_is": 1.0,
            "admit_import_pmax_ir": 1.0,

            "superspreader_staff": "s1",
            "superspreader_state": "CR",
            "superspreader_start_day": 14,
            "superspreader_end_day": 21,
            "superspreader_patient_frac_mult": 2,
            "superspreader_patient_min_add": 1,
            "superspreader_staff_contacts": 30,
            "superspreader_edge_weight_mult": 2,
        },
    },

    "TRAIN": {
        "require_complete_action_set": True,
        "task": "oracle_best_action_h7",
        "T": 7,
        "sliding_step": 1,
        "use_action_conditioning": True,
        "hidden": 128,
        "heads": 2,
        "dropout": 0.2,
        "transformer_layers": 2,
        "sage_layers": 4,
        "batch_size": 8,
        "epochs": 100,
        "lr": 1e-4,
        "neighbor_sampling": False,
        "num_neighbors": "20,10",
        "seed_count": 256,
        "seed_strategy": "random",
        "seed_batch_size": 64,
        "max_sub_batches": 4,
        "max_neighbors": 0,
        "emit_translational_figures": False,
        "out_name": "policy_training_large_ns",

        "oracle_best_action_loss": True,
        "oracle_best_action_loss_weight": 1,

        "aux_policy_loss": False,
        "aux_policy_target_name": "y_h7_trans_import_res_gain",
        "aux_policy_loss_weight": 2,

        "pairwise_policy_ranking_loss": True,
        "pairwise_policy_ranking_weight": 1,
        "pairwise_policy_margin": 0.02,
        "pairwise_policy_min_target_gap": 0.01,

        "action_hidden_dim": 256,
        "action_interaction_hidden_dim": 256,
        "action_interaction_dropout": 0.1,
    },

    "EVAL": {
        "split": "test",
        "oracle_metric": "y_h7_trans_import_res_gain",
        "selection_direction": "auto",
        "emit_action_scores_csv": True,
        "out_name": "policy_selector_eval_large",
    },

    "MILP": {
        "config_json": "milp_policy_config.json",
        "solver": "cplex",
        "cplex_executable": "~/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/cplex",
        "split": "test",
        "score_transform": "baseline_delta",
        "seed": "",
        "budget_total": "",
        "time_limit_sec": 300,
        "mip_gap": 0.0,
        "out_name": "policy_milp_large",
    },

    # C5 is a post-hoc analysis only. It consumes C1--C4 outputs and does not
    # retrain or alter the primary causal policy experiment.
    "C5": {
        "out_name": "policy_milp_quantitative_eval",
        "calibration_split": "validation",
        "evaluation_split": "test",
        "budget_fractions": "0.25,0.50,1.00",
        "max_nonbaseline_fraction": 0.50,
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 260928,
        "calibration_bins": 8,
        "time_limit_sec": 300,
        "mip_gap": 0.0,
        "force_validation_rescore": False,
    },
}

DEFAULT_RESULTS_PARENT = Path("experiments_causal_results")

# =============================================================================
# Utilities
# =============================================================================
def _safe_tag(s: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in ".-_") else "_" for ch in str(s))
    while "__" in out:
        out = out.replace("__", "_")
    out = out.strip("_")
    return out or "tag"


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _state_mode_label() -> str:
    state_mode = str(os.environ.get("DT_STATE_MODE", "ground_truth")).strip()
    return state_mode if state_mode in {"ground_truth", "partial_observation"} else "ground_truth"


def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Missing required file: {path.resolve()}")


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _track_root(results_parent: Path) -> Path:
    return results_parent / f"TRACK_{_state_mode_label()}"


def _overleaf_dir(results_parent: Path, override: Optional[str]) -> Path:
    if override:
        return Path(override)
    return results_parent / "overleaf_package"


def _normalize_python_script_cmd_for_cwd(cmd: List[str], cwd: Optional[Path]) -> List[str]:
    if not cmd or cwd is None:
        return list(cmd)
    out = list(cmd)
    if len(out) >= 2 and str(out[1]).endswith(".py"):
        p = Path(out[1])
        if p.is_absolute():
            try:
                out[1] = str(p.relative_to(cwd))
            except Exception:
                pass
    return out

def _register_child_proc(proc: subprocess.Popen[Any]) -> None:
    _ACTIVE_CHILD_PROCS.append(proc)


def _unregister_child_proc(proc: subprocess.Popen[Any]) -> None:
    try:
        _ACTIVE_CHILD_PROCS.remove(proc)
    except ValueError:
        pass


def _terminate_active_children() -> None:
    for proc in list(_ACTIVE_CHILD_PROCS):
        try:
            if proc.poll() is not None:
                continue
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass

    deadline = time.time() + 5.0
    while time.time() < deadline:
        alive = [p for p in _ACTIVE_CHILD_PROCS if p.poll() is None]
        if not alive:
            return
        time.sleep(0.2)

    for proc in list(_ACTIVE_CHILD_PROCS):
        try:
            if proc.poll() is not None:
                continue
            if os.name == "nt":
                proc.kill()
            else:
                os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass


def _signal_handler(signum: int, frame: Any) -> None:
    _terminate_active_children()
    raise KeyboardInterrupt
# =============================================================================
# Report / progress
# =============================================================================
class Report:
    def __init__(self, path: Path, dry: bool):
        self.path = path
        self.dry = bool(dry)
        self._enabled = not self.dry
        self._console_enabled = os.environ.get("DT_DISABLE_CONSOLE_PROGRESS", "0") != "1"
        self._active_kind: Optional[str] = None
        self._console_last_render_len: int = 0
        self._step_title: Optional[str] = None
        self._step_current: int = 0
        self._step_total: int = 1
        self._step_extra: str = ""
        self._step_started_at: Optional[float] = None
        if self._enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            existed = self.path.exists() and self.path.stat().st_size > 0
            with self.path.open("a", encoding="utf-8") as f:
                if existed:
                    f.write("\n" + "#" * 80 + "\n")
                f.write(f"SESSION START UTC: {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}\n")

    def write(self, line: str = "") -> None:
        if not self._enabled:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(str(line) + "\n")

    def section(self, title: str, total: int = 1) -> None:
        self.write("")
        self.write("=" * 80)
        self.write(str(title))
        self.write("=" * 80)
        self.start_step_console(str(title), total=total)

    def kv(self, key: str, value: Any) -> None:
        self.write(f"{key}: {value}")

    def _progress_bar(self, current: int, total: int, width: int = 28) -> str:
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        filled = int(round(width * current / total))
        return "[" + "#" * filled + "-" * (width - filled) + f"] {current}/{total}"

    def _fmt_elapsed(self, started_at: Optional[float]) -> str:
        if started_at is None:
            return "00:00:00"
        elapsed = max(0, int(round(time.time() - started_at)))
        h = elapsed // 3600
        m = (elapsed % 3600) // 60
        s = elapsed % 60
        return f"{h:02d}:{m:02d}:{s:02d}"

    def start_step_console(self, title: str, total: int = 1) -> None:
        self._step_title = str(title)
        self._step_current = 0
        self._step_total = max(1, int(total))
        self._step_extra = ""
        self._step_started_at = time.time()
        self._active_kind = "step"
        self._render_console()

    def update_step_console(self, current: Optional[int] = None, total: Optional[int] = None, extra: str = "") -> None:
        if self._active_kind != "step":
            return
        if current is not None:
            self._step_current = int(current)
        if total is not None:
            self._step_total = max(1, int(total))
        self._step_extra = str(extra)
        self._render_console()

    def finish_step_console(self, force_complete: bool = True) -> None:
        if self._active_kind != "step":
            return
        if force_complete:
            self._step_current = self._step_total
        self._render_console(final=True)
        self._active_kind = None
        self._console_last_render_len = 0

    def _render_console(self, final: bool = False) -> None:
        if not self._console_enabled or self._step_title is None:
            return
        bar = self._progress_bar(self._step_current, self._step_total)
        extra = f" | {self._step_extra}" if self._step_extra else ""
        elapsed = self._fmt_elapsed(self._step_started_at)
        line = f"{self._step_title} {bar} | {elapsed}{extra}"
        if sys.stdout.isatty():
            pad = max(0, self._console_last_render_len - len(line))
            sys.stdout.write("\r" + line + (" " * pad))
            sys.stdout.flush()
            self._console_last_render_len = len(line)
            if final:
                sys.stdout.write("\n")
                sys.stdout.flush()
        else:
            print(line, flush=True)


def _write_track_progress_state(progress_path: Path, payload: Dict[str, Any]) -> None:
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_track_progress_state(progress_path: Path) -> Optional[Dict[str, Any]]:
    return _read_json_if_exists(progress_path)


def _render_parallel_tracks_console(states: List[Dict[str, Any]], initialized: bool) -> bool:
    if not sys.stdout.isatty() or str(os.environ.get("TERM", "")).strip().lower() == "dumb":
        if not initialized:
            print(f"TOTAL causal policy experiment | tracks={len(states)}", flush=True)
            initialized = True
        for st in states:
            track_name = str(st.get("track_name", "TRACK_unknown"))
            completed = int(st.get("completed_steps", 0))
            total = max(1, int(st.get("total_steps", 1)))
            status = str(st.get("status", "pending"))
            active = str(st.get("active_step", "starting"))
            elapsed = float(st.get("elapsed_sec", 0.0))
            hh = int(elapsed) // 3600
            mm = (int(elapsed) % 3600) // 60
            ss = int(elapsed) % 60
            bar = "[" + "#" * int(round(28 * completed / total)) + "-" * (28 - int(round(28 * completed / total))) + f"] {completed}/{total}"
            print(f"{track_name:<24} {bar} | stage={status} | {hh:02d}:{mm:02d}:{ss:02d} | {active}", flush=True)
        return initialized

    if not initialized:
        sys.stdout.write("\033[?25l")
        initialized = True
    sys.stdout.write("\033[H\033[J")
    sys.stdout.write(f"TOTAL causal policy experiment | tracks={len(states)}\n")
    for st in states:
        track_name = str(st.get("track_name", "TRACK_unknown"))
        completed = int(st.get("completed_steps", 0))
        total = max(1, int(st.get("total_steps", 1)))
        status = str(st.get("status", "pending"))
        active = str(st.get("active_step", "starting"))
        elapsed = float(st.get("elapsed_sec", 0.0))
        hh = int(elapsed) // 3600
        mm = (int(elapsed) % 3600) // 60
        ss = int(elapsed) % 60
        width = 28
        filled = int(round(width * completed / total))
        bar = "[" + "#" * filled + "-" * (width - filled) + f"] {completed}/{total}"
        sys.stdout.write(f"{track_name:<24} {bar} | stage={status} | {hh:02d}:{mm:02d}:{ss:02d} | {active}\n")
    sys.stdout.flush()
    return initialized


# =============================================================================
# Command runner
# =============================================================================
def run_cmd(
    cmd: List[str],
    *,
    cwd: Optional[Path],
    dry: bool,
    report: Optional[Report],
    stream_output: bool = False,
) -> None:
    cwd_path = cwd.resolve() if cwd is not None else None
    cmd_norm = _normalize_python_script_cmd_for_cwd(cmd, cwd_path)
    if report is not None:
        report.write(f"CMD: {' '.join(shlex.quote(str(x)) for x in cmd_norm)}")
        if cwd_path is not None:
            report.write(f"CWD: {cwd_path}")
    if dry:
        return

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(cwd_path) if cwd_path is not None else None,
        "text": True,
    }
    if stream_output:
        popen_kwargs["stdout"] = None
        popen_kwargs["stderr"] = None
    else:
        popen_kwargs["stdout"] = subprocess.PIPE
        popen_kwargs["stderr"] = subprocess.PIPE

    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd_norm, **popen_kwargs)
    _register_child_proc(proc)

    try:
        stdout_text, stderr_text = proc.communicate()
    except KeyboardInterrupt:
        if report is not None:
            report.write("KeyboardInterrupt received; terminating active subprocess tree.")
        try:
            if os.name == "nt":
                proc.terminate()
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except Exception:
            pass
        try:
            proc.wait(timeout=5)
        except Exception:
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
            try:
                proc.wait(timeout=5)
            except Exception:
                pass
        raise
    finally:
        _unregister_child_proc(proc)

    stdout_text = "" if stream_output else (stdout_text or "")
    stderr_text = "" if stream_output else (stderr_text or "")

    if report is not None and not stream_output:
        if stdout_text.strip():
            report.write("--- stdout begin ---")
            for line in stdout_text.splitlines():
                report.write(line)
            report.write("--- stdout end ---")
        if stderr_text.strip():
            report.write("--- stderr begin ---")
            for line in stderr_text.splitlines():
                report.write(line)
            report.write("--- stderr end ---")

    if proc.returncode != 0:
        err_lines = [
            f"Command failed with exit code {proc.returncode}",
            f"CMD: {' '.join(str(x) for x in cmd_norm)}",
            f"CWD: {cwd_path if cwd_path is not None else Path.cwd()}",
        ]
        if stdout_text.strip():
            err_lines += ["--- stdout (tail) ---", stdout_text.strip()[-4000:]]
        if stderr_text.strip():
            err_lines += ["--- stderr (tail) ---", stderr_text.strip()[-4000:]]
        raise subprocess.CalledProcessError(
            proc.returncode,
            cmd_norm,
            output=stdout_text,
            stderr="\n".join(err_lines),
        )
# =============================================================================
# Path helpers
# =============================================================================
def _paths(results_parent: Path) -> Dict[str, Path]:
    track_root = _track_root(results_parent)
    work_root = track_root / "work"
    dataset_root = work_root / _safe_tag(str(CONFIG["DATASET"]["out_name"]))
    train_root = work_root / _safe_tag(str(CONFIG["TRAIN"]["out_name"]))
    eval_root = work_root / _safe_tag(str(CONFIG["EVAL"]["out_name"]))
    milp_root = work_root / _safe_tag(str(CONFIG["MILP"]["out_name"]))
    c5_root = work_root / _safe_tag(str(CONFIG["C5"]["out_name"]))
    return {
        "track_root": track_root,
        "work_root": work_root,
        "dataset_root": dataset_root,
        "train_root": train_root,
        "eval_root": eval_root,
        "milp_root": milp_root,
        "c5_root": c5_root,
        "progress_json": track_root / "causal_progress.json",
        "summary_json": work_root / "causal_policy_experiment_summary.json",
    }


def _candidate_interventions_json(repo_root: Path) -> Path:
    p = str(CONFIG["DATASET"].get("candidate_interventions_json", "")).strip()
    if p == "":
        raise ValueError("CONFIG['DATASET']['candidate_interventions_json'] must be set")
    path = Path(p)
    if not path.is_absolute():
        path = repo_root / path
    return path


def _baseline_interventions_json(repo_root: Path) -> Optional[Path]:
    p = str(CONFIG["DATASET"].get("baseline_intervention_json", "")).strip()
    if p == "":
        return None
    path = Path(p)
    if not path.is_absolute():
        path = repo_root / path
    return path


# =============================================================================
# Build commands
# =============================================================================
def _baseline_balance_label_attr_from_task(task_name: str) -> str:
    task = str(task_name).strip().lower()
    m = __import__("re").search(r"_h(\d+)$", task)
    if not m:
        return ""
    h = int(m.group(1))
    if task.startswith("endogenous_importation_majority_h"):
        return f"y_h{h}_endog_majority"
    if task.startswith("endogenous_transmission_majority_h"):
        return f"y_h{h}_trans_majority"
    if task.startswith("early_outbreak_warning_h"):
        return f"y_h{h}_resistant_frac_cls"
    return ""


def _build_dataset_cmd(repo_root: Path, paths: Dict[str, Path]) -> List[str]:
    cfg = CONFIG["DATASET"]
    cmd: List[str] = [
        sys.executable,
        "build_causal_policy_dataset.py",
        "--out_root", str(paths["dataset_root"]),
        "--state_mode", _state_mode_label(),
        "--candidate_interventions_json", str(_candidate_interventions_json(repo_root)),
        "--window_T", str(int(cfg["window_T"])),
        "--horizons", str(cfg["horizons"]),
        "--decision_days", str(cfg["decision_days"]),
        "--decision_stride", str(int(cfg["decision_stride"])),
        "--action_start_mode", str(cfg["action_start_mode"]),
        "--decision_applies_from", str(cfg["decision_applies_from"]),
        "--jobs", str(int(cfg.get("jobs", 1))),
    ]
    if bool(cfg.get("include_baseline", False)):
        cmd.append("--include_baseline")
    baseline_json = _baseline_interventions_json(repo_root)
    if baseline_json is not None:
        cmd.extend(["--baseline_intervention_json", str(baseline_json)])
    if bool(cfg.get("force", False)):
        cmd.append("--force")

    if bool(cfg.get("use_balanced_base_families", False)):
        cmd.extend([
            "--use_balanced_base_families",
            "--family_profile", str(cfg.get("family_profile", "mechanism_split")),
            "--n_sims_per_family", str(int(cfg.get("n_sims_per_family", 10))),
            "--val_sims_per_family", str(int(cfg.get("val_sims_per_family", 2))),
            "--train_endog_seed_base", str(int(cfg.get("train_endog_seed_base", 4100))),
            "--train_import_seed_base", str(int(cfg.get("train_import_seed_base", 5100))),
            "--test_endog_seed_base", str(int(cfg.get("test_endog_seed_base", 6100))),
            "--test_import_seed_base", str(int(cfg.get("test_import_seed_base", 7100))),
        ])
        balance_attr = _baseline_balance_label_attr_from_task(str(CONFIG["TRAIN"].get("task", "")))
        if balance_attr:
            cmd.extend(["--balance_label_attr", balance_attr])
        if bool(cfg.get("require_two_class_splits", False)):
            cmd.append("--require_two_class_splits")
    else:
        cmd.extend(["--train_seeds", str(cfg["train_seeds"]), "--val_seeds", str(cfg["val_seeds"]), "--test_seeds", str(cfg["test_seeds"])])

    cmd.append("--")
    sim_args = dict(cfg.get("sim_args", {}) or {})
    for key, value in sim_args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if key == "export_gif":
                cmd.append("--export_gif" if value else "--no_export_gif")
            elif value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def _build_train_cmd(paths: Dict[str, Path], train_cfg_override: Optional[Dict[str, Any]] = None) -> List[str]:
    cfg = dict(train_cfg_override) if train_cfg_override is not None else _load_best_train_config_override(CONFIG)

    cmd: List[str] = [
        sys.executable,
        "train_amr_dygformer.py",
        "--data_folder", str(paths["dataset_root"]),
        "--task", str(cfg["task"]),
        "--T", str(int(cfg["T"])),
        "--sliding_step", str(int(cfg["sliding_step"])),
        "--hidden", str(int(cfg["hidden"])),
        "--heads", str(int(cfg["heads"])),
        "--dropout", str(float(cfg["dropout"])),
        "--transformer_layers", str(int(cfg["transformer_layers"])),
        "--sage_layers", str(int(cfg["sage_layers"])),
        "--batch_size", str(int(cfg["batch_size"])),
        "--epochs", str(int(cfg["epochs"])),
        "--lr", str(float(cfg["lr"])),
        "--out_dir", str(paths["train_root"]),
        "--neighbor_sampling", "true" if bool(cfg.get("neighbor_sampling", False)) else "false",
        "--emit_translational_figures", "true" if bool(cfg.get("emit_translational_figures", False)) else "false",
        "--use_action_conditioning", "true" if bool(cfg.get("use_action_conditioning", False)) else "false",
        "--require_complete_action_set", "true" if bool(cfg.get("require_complete_action_set", True)) else "false",
    ]
    
    if bool(cfg.get("use_cls", False)):
        cmd.append("--use_cls")

    # Neighbor sampling / legacy edge thinning
    if bool(cfg.get("neighbor_sampling", False)):
        cmd.extend([
            "--num_neighbors", str(cfg.get("num_neighbors", "15,10")),
            "--seed_count", str(int(cfg.get("seed_count", 256))),
            "--seed_strategy", str(cfg.get("seed_strategy", "random")),
            "--seed_batch_size", str(int(cfg.get("seed_batch_size", 64))),
            "--max_sub_batches", str(int(cfg.get("max_sub_batches", 4))),
        ])
    else:
        cmd.extend([
            "--max_neighbors", str(int(cfg.get("max_neighbors", 20))),
        ])

    # Action-conditioning / policy-learning controls
    cmd.extend([
        "--action_hidden_dim", str(int(cfg.get("action_hidden_dim", 32))),
        "--action_interaction_hidden_dim", str(int(cfg.get("action_interaction_hidden_dim", 128))),
        "--action_interaction_dropout", str(float(cfg.get("action_interaction_dropout", 0.1))),
        "--aux_policy_loss", "true" if bool(cfg.get("aux_policy_loss", True)) else "false",
        "--aux_policy_loss_weight", str(float(cfg.get("aux_policy_loss_weight", 0.25))),
        "--aux_policy_target_name", str(cfg.get("aux_policy_target_name", "")),
        "--pairwise_policy_ranking_loss", "true" if bool(cfg.get("pairwise_policy_ranking_loss", True)) else "false",
        "--pairwise_policy_ranking_weight", str(float(cfg.get("pairwise_policy_ranking_weight", 2.0))),
        "--pairwise_policy_margin", str(float(cfg.get("pairwise_policy_margin", 0.05))),
        "--pairwise_policy_min_target_gap", str(float(cfg.get("pairwise_policy_min_target_gap", 0.01))),
        "--oracle_best_action_loss", "true" if bool(cfg.get("oracle_best_action_loss", True)) else "false",
        "--oracle_best_action_loss_weight", str(float(cfg.get("oracle_best_action_loss_weight", 1.0))),
    ])

    # Attribution / heatmap controls
    cmd.extend([
        "--fullgraph_attribution_pass", "true" if bool(cfg.get("fullgraph_attribution_pass", True)) else "false",
        "--translational_top_k", str(int(cfg.get("translational_top_k", 20))),
        "--attn_top_k", str(int(cfg.get("attn_top_k", 50))),
        "--attn_rank_by", str(cfg.get("attn_rank_by", "abs_diff")),
    ])

    # Reproducibility / metadata / execution controls
    cmd.extend([
        "--cpu_threads", str(int(cfg.get("cpu_threads", max(1, (os.cpu_count() or 1))))),
        "--split_seed", str(int(cfg.get("split_seed", 0))),
        "--train_model", "true" if bool(cfg.get("train_model", True)) else "false",
        "--require_pt_metadata", "true" if bool(cfg.get("require_pt_metadata", True)) else "false",
        "--fail_on_noncontiguous", "true" if bool(cfg.get("fail_on_noncontiguous", True)) else "false",
    ])

    # Repeat controls are opt-in; ordinary pipeline defaults are unchanged.
    for key in ("model_seed", "eval_seed"):
        if cfg.get(key) is not None:
            cmd.extend([f"--{key}", str(int(cfg[key]))])
    for key in ("deterministic", "metrics_only"):
        if key in cfg:
            cmd.extend([f"--{key}", "true" if cfg[key] else "false"])

    # Optional task-level hyperparameter override
    if bool(cfg.get("use_task_hparams", False)):
        cmd.append("--use_task_hparams")

    # Early stopping / scheduler
    cmd.extend([
        "--early_stopping", "true" if bool(cfg.get("early_stopping", True)) else "false",
        "--patience", str(int(cfg.get("patience", 7))),
        "--min_delta", str(float(cfg.get("min_delta", 1e-4))),
        "--save_best_only", "true" if bool(cfg.get("save_best_only", True)) else "false",
        "--lr_scheduler_on_plateau", "true" if bool(cfg.get("lr_scheduler_on_plateau", False)) else "false",
        "--lr_scheduler_factor", str(float(cfg.get("lr_scheduler_factor", 0.5))),
        "--lr_scheduler_patience", str(int(cfg.get("lr_scheduler_patience", 3))),
        "--lr_scheduler_min_lr", str(float(cfg.get("lr_scheduler_min_lr", 1e-6))),
    ])

    return cmd

def _build_eval_cmd(paths: Dict[str, Path]) -> List[str]:
    cfg = CONFIG["EVAL"]
    cmd: List[str] = [
        sys.executable,
        "evaluate_policy_selector.py",
        "--data_folder",
        str(paths["dataset_root"]),
        "--trained_dir",
        str(paths["train_root"]),
        "--split",
        str(cfg.get("split", "test")),
        "--oracle_metric",
        str(cfg.get("oracle_metric", "auto")),
        "--selection_direction",
        str(cfg.get("selection_direction", "auto")),
        "--emit_action_scores_csv",
        "true" if bool(cfg.get("emit_action_scores_csv", True)) else "false",
        "--out_dir",
        str(paths["eval_root"]),
        "--candidate_interventions_json",
        str(_candidate_interventions_json(Path(__file__).resolve().parent)),
    ]
    train_cfg = _load_best_train_config_override(CONFIG)
    if bool(train_cfg.get("use_action_conditioning", False)):
        cmd.extend(["--action_hidden_dim", str(int(train_cfg.get("action_hidden_dim", 32)))])
        cmd.extend(["--action_interaction_hidden_dim", str(int(train_cfg.get("action_interaction_hidden_dim", 128)))])
        cmd.extend(["--action_interaction_dropout", str(float(train_cfg.get("action_interaction_dropout", 0.1)))])
    return cmd


def _milp_config_json(repo_root: Path) -> Path:
    p = str(CONFIG["MILP"].get("config_json", "")).strip()
    if p == "":
        raise ValueError("CONFIG['MILP']['config_json'] must be set")
    return (repo_root / p).resolve()


def _build_milp_cmd(repo_root: Path, paths: Dict[str, Path]) -> List[str]:
    cfg = CONFIG["MILP"]
    cmd: List[str] = [
        sys.executable,
        "optimize_policy_milp.py",
        "--eval_dir",
        str(paths["eval_root"]),
        "--config_json",
        str(_milp_config_json(repo_root)),
        "--split",
        str(cfg.get("split", CONFIG["EVAL"].get("split", "test"))),
        "--score_transform",
        str(cfg.get("score_transform", "baseline_delta")),
        "--solver",
        str(cfg.get("solver", "cplex")),
        "--time_limit_sec",
        str(int(cfg.get("time_limit_sec", 300))),
        "--mip_gap",
        str(float(cfg.get("mip_gap", 0.0))),
        "--out_dir",
        str(paths["milp_root"]),
    ]
    seed_value = str(cfg.get("seed", "")).strip()
    if seed_value != "":
        cmd.extend(["--seed", seed_value])
    budget_value = str(cfg.get("budget_total", "")).strip()
    if budget_value != "":
        cmd.extend(["--budget_total", budget_value])
    cplex_executable = str(cfg.get("cplex_executable", "")).strip()
    if cplex_executable != "":
        cmd.extend(["--cplex_executable", cplex_executable])
    return cmd


def _build_c5_cmd(repo_root: Path, paths: Dict[str, Path]) -> List[str]:
    cfg = CONFIG["C5"]
    return [
        sys.executable,
        "evaluate_policy_milp_c5.py",
        "--data_folder", str(paths["dataset_root"]),
        "--trained_dir", str(paths["train_root"]),
        "--eval_dir", str(paths["eval_root"]),
        "--c4_milp_dir", str(paths["milp_root"]),
        "--milp_config_json", str(_milp_config_json(repo_root)),
        "--candidate_interventions_json", str(_candidate_interventions_json(repo_root)),
        "--evaluator_script", str(repo_root / "evaluate_policy_selector.py"),
        "--out_dir", str(paths["c5_root"]),
        "--calibration_split", str(cfg.get("calibration_split", "validation")),
        "--evaluation_split", str(cfg.get("evaluation_split", "test")),
        "--budget_fractions", str(cfg.get("budget_fractions", "0.25,0.50,1.00")),
        "--max_nonbaseline_fraction", str(float(cfg.get("max_nonbaseline_fraction", 0.50))),
        "--bootstrap_replicates", str(int(cfg.get("bootstrap_replicates", 2000))),
        "--bootstrap_seed", str(int(cfg.get("bootstrap_seed", 260928))),
        "--calibration_bins", str(int(cfg.get("calibration_bins", 8))),
        "--time_limit_sec", str(float(cfg.get("time_limit_sec", 300))),
        "--mip_gap", str(float(cfg.get("mip_gap", 0.0))),
        "--force_validation_rescore", "true" if bool(cfg.get("force_validation_rescore", False)) else "false",
    ]


# =============================================================================
# Summaries / Overleaf export
# =============================================================================
def _collect_summary(paths: Dict[str, Path]) -> Dict[str, Any]:
    dataset_manifest = _read_json_if_exists(paths["dataset_root"] / "dataset_summary.json")
    train_metrics = _read_json_if_exists(paths["train_root"] / "metrics_summary.json")
    eval_summary = _read_json_if_exists(paths["eval_root"] / "policy_selection_summary.json")
    milp_summary = _read_json_if_exists(paths["milp_root"] / "milp_policy_summary.json")
    c5_summary = _read_json_if_exists(paths["c5_root"] / "c5_summary.json")
    return {
        "state_mode": _state_mode_label(),
        "dataset_root": str(paths["dataset_root"]),
        "train_root": str(paths["train_root"]),
        "eval_root": str(paths["eval_root"]),
        "milp_root": str(paths["milp_root"]),
        "c5_root": str(paths["c5_root"]),
        "dataset_summary": dataset_manifest,
        "train_metrics": train_metrics,
        "policy_eval_summary": eval_summary,
        "milp_summary": milp_summary,
        "c5_summary": c5_summary,
    }


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _export_track_figures(track_root: Path, figures_dir: Path) -> Tuple[List[str], List[str]]:
    work_root = track_root / "work"
    train_root = work_root / _safe_tag(str(CONFIG["TRAIN"]["out_name"]))
    eval_root = work_root / _safe_tag(str(CONFIG["EVAL"]["out_name"]))
    milp_root = work_root / _safe_tag(str(CONFIG["MILP"]["out_name"]))
    copied_train: List[str] = []
    copied_eval: List[str] = []
    copied_milp: List[str] = []

    train_figs = [
        "loss_curves.png",
        "confusion_matrix.png",
        "roc_curve.png",
        "confusion_matrix_test.png",
        "roc_curve_test.png",
        "attention_heatmap.png",
        "attention_heatmap_test.png",
    ]
    eval_figs = [
        "policy_selection_confusion_matrix.png",
        "policy_regret_histogram.png",
        "policy_gain_over_baseline.png",
        "policy_action_selection_counts.png",
        "policy_predicted_vs_oracle_scores.png",
        "policy_evaluation_dashboard.png",
    ]
    milp_figs = [
        "milp_selected_actions_timeline.png",
        "milp_cumulative_predicted_gain.png",
    ]

    track_tag = _safe_tag(track_root.name)
    for name in train_figs:
        src = train_root / name
        if _copy_if_exists(src, figures_dir / f"{track_tag}__train__{name}"):
            copied_train.append(f"{track_tag}__train__{name}")
    for name in eval_figs:
        src = eval_root / name
        if _copy_if_exists(src, figures_dir / f"{track_tag}__eval__{name}"):
            copied_eval.append(f"{track_tag}__eval__{name}")
    for name in milp_figs:
        src = milp_root / name
        if _copy_if_exists(src, figures_dir / f"{track_tag}__milp__{name}"):
            copied_milp.append(f"{track_tag}__milp__{name}")
    return copied_train, copied_eval, copied_milp


def export_overleaf_package(results_parent: Path, overleaf_dir: Path, report: Optional[Report]) -> None:
    figures_dir = overleaf_dir / "figures"
    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    latex_lines: List[str] = [
        "% Auto-generated by experiments_causal.py",
        f"% UTC: {_dt.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}",
        "",
        "\\section{Causal policy experiment}",
        "",
        "\\graphicspath{{overleaf_package/}{overleaf_package/figures/}}",
        "",
    ]
    copied = 0
    for track_root in sorted([p for p in results_parent.iterdir() if p.is_dir() and p.name.startswith("TRACK_")]):
        train_figs, eval_figs, milp_figs = _export_track_figures(track_root, figures_dir)
        copied += len(train_figs) + len(eval_figs) + len(milp_figs)
        summary_json = track_root / "work" / "causal_policy_experiment_summary.json"
        summary = _read_json_if_exists(summary_json) or {}
        train_metrics = (summary.get("train_metrics") if isinstance(summary, dict) else None) or {}
        policy_eval = (summary.get("policy_eval_summary") if isinstance(summary, dict) else None) or {}
        milp_summary = (summary.get("milp_summary") if isinstance(summary, dict) else None) or {}
        latex_lines.extend([
            f"\\subsection{{{_latex_escape(track_root.name)}}}",
            "",
            f"\\paragraph{{Training task.}} {_latex_escape(str(CONFIG['TRAIN']['task']))}",
            "",
        ])
        if isinstance(policy_eval, dict) and policy_eval:
            metrics = policy_eval.get("summary", {}) if isinstance(policy_eval.get("summary", {}), dict) else policy_eval
            if isinstance(metrics, dict):
                txt = ", ".join(f"{k}={metrics[k]}" for k in sorted(metrics.keys()) if isinstance(metrics[k], (int, float, str)))
                if txt:
                    latex_lines.extend([f"\\paragraph{{Policy evaluation summary.}} {_latex_escape(txt)}", ""])
        for idx, fig_name in enumerate(train_figs, start=1):
            latex_lines.extend([
                "\\begin{figure}[p]",
                "\\centering",
                f"\\includegraphics[width=0.95\\textwidth]{{figures/{fig_name}}}",
                f"\\caption{{Training figure {idx} for {_latex_escape(track_root.name)} in the causal policy experiment.}}",
                f"\\label{{fig:{_safe_tag(track_root.name)}_train_{idx}}}",
                "\\end{figure}",
                "",
            ])
        for idx, fig_name in enumerate(eval_figs, start=1):
            latex_lines.extend([
                "\\begin{figure}[p]",
                "\\centering",
                f"\\includegraphics[width=0.95\\textwidth]{{figures/{fig_name}}}",
                f"\\caption{{Policy-evaluation figure {idx} for {_latex_escape(track_root.name)} in the causal policy experiment.}}",
                f"\\label{{fig:{_safe_tag(track_root.name)}_eval_{idx}}}",
                "\\end{figure}",
                "",
            ])
    latex_path = overleaf_dir / "latex.txt"
    latex_path.write_text("\n".join(latex_lines).rstrip() + "\n", encoding="utf-8")
    manifest = {
        "overleaf_dir": str(overleaf_dir),
        "figures_dir": str(figures_dir),
        "copied_figure_count": int(copied),
        "latex_file": str(latex_path),
    }
    (overleaf_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if report is not None:
        report.kv("overleaf_dir", overleaf_dir)
        report.kv("overleaf_figures", copied)
        report.kv("overleaf_latex", latex_path)


# =============================================================================
# Main execution
# =============================================================================
def _execute_single_track(results_parent: Path, dry: bool, start_step: int, stop_step: int) -> int:
    repo_root = _repo_root()
    require_file(repo_root / "build_causal_policy_dataset.py")
    require_file(repo_root / "train_amr_dygformer.py")
    require_file(repo_root / "evaluate_policy_selector.py")
    require_file(repo_root / "optimize_policy_milp.py")
    require_file(repo_root / "evaluate_policy_milp_c5.py")
    require_file(repo_root / str(CONFIG["DATASET"]["candidate_interventions_json"]))
    require_file(_milp_config_json(repo_root))

    paths = _paths(results_parent)
    report = Report(paths["track_root"] / f"run_report_{paths['track_root'].name}.txt", dry=dry)
    started = time.time()
    total_steps = 5

    def write_progress(status: str, completed_steps: int, active_step: str) -> None:
        _write_track_progress_state(
            paths["progress_json"],
            {
                "track_name": paths["track_root"].name,
                "status": status,
                "completed_steps": completed_steps,
                "total_steps": total_steps,
                "active_step": active_step,
                "elapsed_sec": float(time.time() - started),
            },
        )

    if not dry:
        paths["track_root"].mkdir(parents=True, exist_ok=True)
        paths["work_root"].mkdir(parents=True, exist_ok=True)

    report.section("STEP C1 | build causal policy dataset", total=1)
    report.kv("track", paths["track_root"].name)
    report.kv("state_mode", _state_mode_label())
    report.kv("results_parent", results_parent)
    report.kv("dataset_root", paths["dataset_root"])
    if start_step <= 1 <= stop_step:
        write_progress("running", 0, "STEP C1 dataset build")
        run_cmd(_build_dataset_cmd(repo_root, paths), cwd=repo_root, dry=dry, report=report, stream_output=False)
    else:
        report.write("STEP C1 skipped by start/stop range")
    report.update_step_console(current=1, total=1, extra="done")
    report.finish_step_console(force_complete=True)

    report.section("STEP C2 | train action-conditioned predictor", total=1)
    report.kv("train_root", paths["train_root"])
    if start_step <= 2 <= stop_step:
        write_progress("running", 1, "STEP C2 model training")
        run_cmd(_build_train_cmd(paths), cwd=repo_root, dry=dry, report=report, stream_output=False)
    else:
        report.write("STEP C2 skipped by start/stop range")
    report.update_step_console(current=1, total=1, extra="done")
    report.finish_step_console(force_complete=True)

    report.section("STEP C3 | evaluate policy selector", total=1)
    report.kv("eval_root", paths["eval_root"])
    if start_step <= 3 <= stop_step:
        write_progress("running", 2, "STEP C3 policy evaluation")
        run_cmd(_build_eval_cmd(paths), cwd=repo_root, dry=dry, report=report, stream_output=False)
    else:
        report.write("STEP C3 skipped by start/stop range")
    report.update_step_console(current=1, total=1, extra="done")
    report.finish_step_console(force_complete=True)

    report.section("STEP C4 | optimize MILP decision layer", total=1)
    report.kv("milp_root", paths["milp_root"])
    if start_step <= 4 <= stop_step:
        write_progress("running", 3, "STEP C4 MILP optimization")
        run_cmd(_build_milp_cmd(repo_root, paths), cwd=repo_root, dry=dry, report=report, stream_output=False)
    else:
        report.write("STEP C4 skipped by start/stop range")
    report.update_step_console(current=1, total=1, extra="done")
    report.finish_step_console(force_complete=True)

    report.section("STEP C5 | quantitative MILP evaluation", total=1)
    report.kv("c5_root", paths["c5_root"])
    if start_step <= 5 <= stop_step:
        write_progress("running", 4, "STEP C5 quantitative MILP evaluation")
        run_cmd(_build_c5_cmd(repo_root, paths), cwd=repo_root, dry=dry, report=report, stream_output=False)
    else:
        report.write("STEP C5 skipped by start/stop range")
    report.update_step_console(current=1, total=1, extra="done")
    report.finish_step_console(force_complete=True)

    summary = _collect_summary(paths)
    if not dry:
        paths["summary_json"].write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    report.kv("summary_json", paths["summary_json"])
    write_progress("done", total_steps, "complete")
    return 0


def _launch_parallel_tracks(args: argparse.Namespace, results_parent: Path) -> int:
    repo_root = _repo_root()
    parent_report = Report(results_parent / "run_report.txt", dry=bool(args.dry_run))
    parent_report.section("TOTAL | causal policy experiment", total=1)
    parent_report.kv("results_parent", results_parent)
    parent_report.kv("run_both_state_modes", True)

    track_modes = ["ground_truth", "partial_observation"]
    procs: List[Tuple[str, subprocess.Popen[str], Path]] = []
    initialized = False

    for state_mode in track_modes:
        track_root = results_parent / f"TRACK_{state_mode}"
        progress_path = track_root / "causal_progress.json"
        if progress_path.exists() and not bool(args.dry_run):
            try:
                progress_path.unlink()
            except FileNotFoundError:
                pass

        child_cmd = [
            sys.executable,
            "experiments_causal.py",
            "--results_parent",
            str(results_parent),
            "--start",
            str(args.start),
            "--stop",
            str(args.stop),
        ]
        if bool(args.dry_run):
            child_cmd.append("--dry_run")

        env = os.environ.copy()
        env["DT_STATE_MODE"] = state_mode
        env["DT_DISABLE_CONSOLE_PROGRESS"] = "1"

        popen_kwargs: Dict[str, Any] = {
            "cwd": str(repo_root),
            "env": env,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "text": True,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            popen_kwargs["start_new_session"] = True

        proc = subprocess.Popen(
            _normalize_python_script_cmd_for_cwd(child_cmd, repo_root),
            **popen_kwargs,
        )
        procs.append((state_mode, proc, progress_path))

    interrupted = False
    try:
        while True:
            states: List[Dict[str, Any]] = []
            done_count = 0
            for state_mode, proc, progress_path in procs:
                state = _read_track_progress_state(progress_path)
                if not state:
                    state = {
                        "track_name": f"TRACK_{state_mode}",
                        "status": "pending" if proc.poll() is None else ("done" if proc.returncode == 0 else "failed"),
                        "completed_steps": 0,
                        "total_steps": 5,
                        "active_step": "starting" if proc.poll() is None else "complete",
                        "elapsed_sec": 0.0,
                    }
                if proc.poll() is not None:
                    if proc.returncode == 0:
                        state["status"] = "done"
                        state["completed_steps"] = state.get("total_steps", 5)
                        state["active_step"] = "complete"
                    else:
                        state["status"] = "failed"
                    done_count += 1
                states.append(state)

            initialized = _render_parallel_tracks_console(states, initialized=initialized)
            if done_count == len(procs):
                break
            time.sleep(0.5)

    except KeyboardInterrupt:
        interrupted = True
        parent_report.write("KeyboardInterrupt received; terminating all child track processes.")
        for state_mode, proc, _ in procs:
            if proc.poll() is not None:
                continue
            try:
                if os.name == "nt":
                    proc.terminate()
                else:
                    os.killpg(proc.pid, signal.SIGTERM)
            except Exception as exc:
                parent_report.write(f"Track {state_mode}: SIGTERM/terminate failed: {exc}")

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if all(proc.poll() is not None for _, proc, _ in procs):
                break
            time.sleep(0.2)

        for state_mode, proc, _ in procs:
            if proc.poll() is not None:
                continue
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(proc.pid, signal.SIGKILL)
            except Exception as exc:
                parent_report.write(f"Track {state_mode}: SIGKILL/kill failed: {exc}")

        raise

    finally:
        if sys.stdout.isatty() and str(os.environ.get("TERM", "")).strip().lower() != "dumb":
            sys.stdout.write("\n\033[?25h")
            sys.stdout.flush()

    failures: List[str] = []
    for state_mode, proc, _ in procs:
        rc = proc.wait()
        _unregister_child_proc(proc)
        parent_report.kv(f"track_{state_mode}_returncode", rc)
        if rc != 0:
            failures.append(state_mode)

    if interrupted:
        return 1

    if not failures and not bool(args.dry_run) and int(args.start) <= 5 <= int(args.stop):
        parent_report.write("Collating C5 outputs across ground-truth and partial-observation tracks.")
        run_cmd(
            [
                sys.executable,
                "evaluate_policy_milp_c5.py",
                "--collate_parent", str(results_parent),
                "--c5_dir_name", _safe_tag(str(CONFIG["C5"]["out_name"])),
                "--out_dir", str(results_parent / "C5_MANUSCRIPT_BUNDLE"),
            ],
            cwd=repo_root,
            dry=False,
            report=parent_report,
            stream_output=False,
        )

    parent_report.update_step_console(
        current=1 if not failures else 0,
        total=1,
        extra="complete" if not failures else "failed",
    )
    parent_report.finish_step_console(force_complete=not failures)
    return 1 if failures else 0
# =============================================================================
# CLI
# =============================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--run_both_state_modes", action="store_true")
    parser.add_argument("--results_parent", type=str, default=str(DEFAULT_RESULTS_PARENT))
    parser.add_argument("--emit_latex", action="store_true")
    parser.add_argument("--emit_latex_only", action="store_true")
    parser.add_argument("--overleaf_dir", type=str, default="")
    parser.add_argument("--start", type=int, default=1, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--stop", type=int, default=5, choices=[1, 2, 3, 4, 5])
    return parser


def main() -> int:
    if "--causal_repeats" in sys.argv[1:]:
        from causal_repeats import main as repeat_main
        return repeat_main([a for a in sys.argv[1:] if a != "--causal_repeats"])
    args = build_parser().parse_args()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    if int(args.start) > int(args.stop):
        raise ValueError("--start must be <= --stop")
    results_parent = Path(args.results_parent)
    overleaf_dir = _overleaf_dir(results_parent, args.overleaf_dir)

    if args.emit_latex_only:
        export_overleaf_package(results_parent, overleaf_dir, report=None)
        return 0

    if args.run_both_state_modes:
        rc = _launch_parallel_tracks(args, results_parent)
        if rc == 0 and args.emit_latex and not args.dry_run:
            export_overleaf_package(results_parent, overleaf_dir, report=None)
        return rc

    rc = _execute_single_track(results_parent, dry=bool(args.dry_run), start_step=int(args.start), stop_step=int(args.stop))
    if rc == 0 and args.emit_latex and not args.dry_run:
        export_overleaf_package(results_parent, overleaf_dir, report=None)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
