#!/usr/bin/env python3
"""
tune_policy_trainer.py


python tune_policy_trainer.py --repo_root /Users/harrytriantafyllidis/AMR_MILP --config_py /Users/harrytriantafyllidis/AMR_MILP/experiments_causal.py --data_folder /Users/harrytriantafyllidis/AMR_MILP/experiments_causal_results/TRACK_ground_truth/work/causal_policy_monthly_shock_superspreader --num_trials 30 --run_name gt_policy_tuning

Validation-only hyperparameter search helper for the causal policy trainer.

What it does
- Loads CONFIG from experiments_causal.py (or another compatible config file).
- Uses CONFIG["TRAIN"] as the base trainer settings.
- Samples hyperparameter trials from a discrete search space.
- Launches train_amr_dygformer.py for each trial.
- Launches evaluate_policy_selector.py on the VALIDATION split only for ranking.
- Selects the best trial using validation oracle-following metrics only.
- Runs the best trial once on the TEST split at the end.

Selection principle
- Test is never used for tuning.
- Trials are ranked primarily by validation policy accuracy, then validation top-2
  accuracy, then lowest validation mean regret.

Outputs
- tuning_runs/<run_name>/leaderboard.csv
- tuning_runs/<run_name>/leaderboard.json
- tuning_runs/<run_name>/best_config.json
- tuning_runs/<run_name>/best_trial.txt
- tuning_runs/<run_name>/best_validation_summary.json
- tuning_runs/<run_name>/best_test_summary.json (if test split exists)
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SEARCH_SPACE: Dict[str, List[Any]] = {
    "lr": [1e-5, 1e-4],
    "dropout": [0.1, 0.2, 0.3],
    "hidden": [32, 64, 128],
    "heads": [2, 4],
    "transformer_layers": [1,2, 3],
    "sage_layers": [1,2, 3],
    "batch_size": [8,16,32],
    "action_hidden_dim": [32, 64, 128],
    "action_interaction_hidden_dim": [64, 128, 256],
    "action_interaction_dropout": [0.0, 0.1, 0.2],
    "aux_policy_loss": [False, True],
    "aux_policy_loss_weight": [0.05, 0.1, 0.2],
    "pairwise_policy_ranking_loss": [False, True],
    "pairwise_policy_ranking_weight": [0.5, 1.0, 1.5, 2, 5, 10],
    "pairwise_policy_margin": [0.05, 0.1, 0.2, 0.5],
    "pairwise_policy_min_target_gap": [0.0, 0.05, 0.1, 0.2, 0.5],
    "oracle_best_action_loss": [False, True],
    "oracle_best_action_loss_weight": [0.5, 1.0, 1.5, 2, 5, 10],
}


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def _load_python_config(path: Path) -> Mapping[str, Any]:
    spec = importlib.util.spec_from_file_location("tuning_config_module", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import config module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = getattr(module, "CONFIG", None)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Expected CONFIG dict in {path}, but none was found.")
    return cfg


def _json_load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return obj


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _read_search_space(path: Optional[Path]) -> Dict[str, List[Any]]:
    if path is None:
        return copy.deepcopy(DEFAULT_SEARCH_SPACE)
    obj = _json_load(path)
    out: Dict[str, List[Any]] = {}
    for k, v in obj.items():
        if not isinstance(v, list) or len(v) == 0:
            raise ValueError(f"Search-space entry '{k}' must be a non-empty list.")
        out[str(k)] = list(v)
    return out


def _sample_trial_config(
    *,
    base_train: Mapping[str, Any],
    search_space: Mapping[str, Sequence[Any]],
    rng: random.Random,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(dict(base_train))
    for key, candidates in search_space.items():
        cfg[key] = copy.deepcopy(rng.choice(list(candidates)))
    return cfg


def _maybe_add_bool(cmd: List[str], flag: str, value: bool) -> None:
    cmd.extend([flag, "true" if bool(value) else "false"])


def _build_train_cmd(
    *,
    repo_root: Path,
    data_folder: Path,
    out_dir: Path,
    train_cfg: Mapping[str, Any],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(repo_root / "train_amr_dygformer.py"),
        "--data_folder",
        str(data_folder),
        "--task",
        str(train_cfg["task"]),
        "--T",
        str(int(train_cfg["T"])),
        "--sliding_step",
        str(int(train_cfg["sliding_step"])),
        "--hidden",
        str(int(train_cfg["hidden"])),
        "--heads",
        str(int(train_cfg["heads"])),
        "--dropout",
        str(float(train_cfg["dropout"])),
        "--transformer_layers",
        str(int(train_cfg["transformer_layers"])),
        "--sage_layers",
        str(int(train_cfg["sage_layers"])),
        "--batch_size",
        str(int(train_cfg["batch_size"])),
        "--epochs",
        str(int(train_cfg["epochs"])),
        "--lr",
        str(float(train_cfg["lr"])),
        "--out_dir",
        str(out_dir),
    ]

    _maybe_add_bool(cmd, "--use_action_conditioning", bool(train_cfg.get("use_action_conditioning", False)))
    if bool(train_cfg.get("use_action_conditioning", False)):
        cmd.extend(["--action_hidden_dim", str(int(train_cfg.get("action_hidden_dim", 32)))])
        cmd.extend(["--action_interaction_hidden_dim", str(int(train_cfg.get("action_interaction_hidden_dim", 128)))])
        cmd.extend(["--action_interaction_dropout", str(float(train_cfg.get("action_interaction_dropout", 0.1)))])
        cmd.extend(["--aux_policy_target_name", str(train_cfg.get("aux_policy_target_name", ""))])
        _maybe_add_bool(cmd, "--aux_policy_loss", bool(train_cfg.get("aux_policy_loss", True)))
        cmd.extend(["--aux_policy_loss_weight", str(float(train_cfg.get("aux_policy_loss_weight", 0.25)))])
        _maybe_add_bool(cmd, "--pairwise_policy_ranking_loss", bool(train_cfg.get("pairwise_policy_ranking_loss", True)))
        cmd.extend(["--pairwise_policy_ranking_weight", str(float(train_cfg.get("pairwise_policy_ranking_weight", 2.0)))])
        cmd.extend(["--pairwise_policy_margin", str(float(train_cfg.get("pairwise_policy_margin", 0.05)))])
        cmd.extend(["--pairwise_policy_min_target_gap", str(float(train_cfg.get("pairwise_policy_min_target_gap", 0.01)))])
        _maybe_add_bool(cmd, "--oracle_best_action_loss", bool(train_cfg.get("oracle_best_action_loss", True)))
        cmd.extend(["--oracle_best_action_loss_weight", str(float(train_cfg.get("oracle_best_action_loss_weight", 1.0)))])

    if bool(train_cfg.get("neighbor_sampling", False)):
        _maybe_add_bool(cmd, "--neighbor_sampling", True)
        cmd.extend(["--num_neighbors", str(train_cfg.get("num_neighbors", "15,10"))])
        cmd.extend(["--seed_count", str(int(train_cfg.get("seed_count", 256)))])
        cmd.extend(["--seed_strategy", str(train_cfg.get("seed_strategy", "random"))])
        cmd.extend(["--seed_batch_size", str(int(train_cfg.get("seed_batch_size", 64)))])
        cmd.extend(["--max_sub_batches", str(int(train_cfg.get("max_sub_batches", 4)))])
    else:
        _maybe_add_bool(cmd, "--neighbor_sampling", False)
        cmd.extend(["--max_neighbors", str(int(train_cfg.get("max_neighbors", 20)))])

    _maybe_add_bool(cmd, "--emit_translational_figures", bool(train_cfg.get("emit_translational_figures", False)))
    return cmd


def _build_eval_cmd(
    *,
    repo_root: Path,
    data_folder: Path,
    trained_dir: Path,
    out_dir: Path,
    eval_cfg: Mapping[str, Any],
    train_cfg: Mapping[str, Any],
    split_name: str,
    candidate_interventions_json: Optional[Path],
) -> List[str]:
    cmd: List[str] = [
        sys.executable,
        str(repo_root / "evaluate_policy_selector.py"),
        "--data_folder",
        str(data_folder),
        "--trained_dir",
        str(trained_dir),
        "--split",
        str(split_name),
        "--oracle_metric",
        str(eval_cfg.get("oracle_metric", "auto")),
        "--selection_direction",
        str(eval_cfg.get("selection_direction", "auto")),
        "--emit_action_scores_csv",
        "true" if bool(eval_cfg.get("emit_action_scores_csv", True)) else "false",
        "--out_dir",
        str(out_dir),
    ]
    if candidate_interventions_json is not None:
        cmd.extend(["--candidate_interventions_json", str(candidate_interventions_json)])

    if bool(train_cfg.get("use_action_conditioning", False)):
        cmd.extend(["--use_action_conditioning", "true"])
        cmd.extend(["--action_hidden_dim", str(int(train_cfg.get("action_hidden_dim", 32)))])
        cmd.extend(["--action_interaction_hidden_dim", str(int(train_cfg.get("action_interaction_hidden_dim", 128)))])
        cmd.extend(["--action_interaction_dropout", str(float(train_cfg.get("action_interaction_dropout", 0.1)))])
    return cmd


def _run_cmd(cmd: Sequence[str], cwd: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("CMD:\n")
        logf.write(" ".join(str(x) for x in cmd) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=logf,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc.wait()


def _trial_score(summary: Mapping[str, Any]) -> Tuple[float, float, float]:
    acc = float(summary.get("policy_accuracy", 0.0))
    top2 = float(summary.get("top2_accuracy", 0.0))
    regret_obj = summary.get("regret", {})
    regret_mean = float(regret_obj.get("mean", 1e18)) if isinstance(regret_obj, Mapping) else 1e18
    score = (1000.0 * acc) + (100.0 * top2) - regret_mean
    return score, acc, regret_mean


def _leaderboard_sort_key(row: Mapping[str, Any]) -> Tuple[float, float, float, int]:
    return (
        float(row.get("validation_policy_accuracy", -1.0)),
        float(row.get("validation_top2_accuracy", -1.0)),
        -float(row.get("validation_regret_mean", 1e18)),
        -int(row.get("trial_index", -1)),
    )


def _write_leaderboard_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description="Validation-only tuner for train_amr_dygformer.py")
    parser.add_argument("--repo_root", type=str, required=True, help="Repository root containing train_amr_dygformer.py and evaluate_policy_selector.py")
    parser.add_argument("--config_py", type=str, default="experiments_causal.py", help="Path to Python config file containing CONFIG")
    parser.add_argument("--data_folder", type=str, required=True, help="Policy-manifest dataset folder used by the trainer")
    parser.add_argument("--candidate_interventions_json", type=str, default="", help="Optional candidate_interventions.json path for evaluation labels")
    parser.add_argument("--search_space_json", type=str, default="", help="Optional JSON file overriding the discrete search space")
    parser.add_argument("--num_trials", type=int, default=30, help="Number of random trials, excluding the base config")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for trial sampling")
    parser.add_argument("--run_name", type=str, default="policy_tuning", help="Subdirectory name under tuning_runs")
    parser.add_argument("--keep_all_trial_dirs", type=str2bool, default=True, help="If false, keep only best trial artifacts")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    config_py = (repo_root / args.config_py).resolve() if not Path(args.config_py).is_absolute() else Path(args.config_py).resolve()
    data_folder = Path(args.data_folder).resolve()
    candidate_json = None
    if str(args.candidate_interventions_json).strip() != "":
        candidate_json = Path(args.candidate_interventions_json).resolve()
    else:
        candidate_path = repo_root / "candidate_interventions.json"
        if candidate_path.exists():
            candidate_json = candidate_path.resolve()

    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo_root not found: {repo_root}")
    if not config_py.is_file():
        raise FileNotFoundError(f"config_py not found: {config_py}")
    if not data_folder.is_dir():
        raise FileNotFoundError(f"data_folder not found: {data_folder}")

    cfg = _load_python_config(config_py)
    if "TRAIN" not in cfg or "EVAL" not in cfg:
        raise ValueError("CONFIG must contain at least TRAIN and EVAL sections.")

    base_train = copy.deepcopy(dict(cfg["TRAIN"]))
    base_eval = copy.deepcopy(dict(cfg["EVAL"]))
    search_space = _read_search_space(Path(args.search_space_json).resolve() if str(args.search_space_json).strip() != "" else None)

    tuning_root = repo_root / "tuning_runs" / str(args.run_name)
    shutil.rmtree(tuning_root, ignore_errors=True)
    tuning_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))
    trials: List[Dict[str, Any]] = []
    trials.append(copy.deepcopy(base_train))
    for _ in range(int(args.num_trials)):
        trials.append(_sample_trial_config(base_train=base_train, search_space=search_space, rng=rng))

    leaderboard: List[Dict[str, Any]] = []
    best_row: Optional[Dict[str, Any]] = None
    best_train_cfg: Optional[Dict[str, Any]] = None
    best_trial_dir: Optional[Path] = None

    for trial_index, train_cfg in enumerate(trials):
        trial_name = f"trial_{trial_index:03d}"
        trial_dir = tuning_root / trial_name
        train_dir = trial_dir / "train"
        val_eval_dir = trial_dir / "eval_validation"
        trial_dir.mkdir(parents=True, exist_ok=True)

        train_cmd = _build_train_cmd(
            repo_root=repo_root,
            data_folder=data_folder,
            out_dir=train_dir,
            train_cfg=train_cfg,
        )
        train_rc = _run_cmd(train_cmd, cwd=repo_root, log_path=trial_dir / "train.log")

        row: Dict[str, Any] = {
            "trial_index": int(trial_index),
            "trial_name": trial_name,
            "train_return_code": int(train_rc),
        }
        row.update({f"cfg__{k}": v for k, v in train_cfg.items()})

        if train_rc != 0:
            row.update(
                {
                    "validation_policy_accuracy": -1.0,
                    "validation_top2_accuracy": -1.0,
                    "validation_regret_mean": 1e18,
                    "validation_score": -1e18,
                    "status": "train_failed",
                }
            )
            leaderboard.append(row)
            continue

        val_eval_cmd = _build_eval_cmd(
            repo_root=repo_root,
            data_folder=data_folder,
            trained_dir=train_dir,
            out_dir=val_eval_dir,
            eval_cfg=base_eval,
            train_cfg=train_cfg,
            split_name="validation",
            candidate_interventions_json=candidate_json,
        )
        val_rc = _run_cmd(val_eval_cmd, cwd=repo_root, log_path=trial_dir / "eval_validation.log")
        row["validation_return_code"] = int(val_rc)

        if val_rc != 0:
            row.update(
                {
                    "validation_policy_accuracy": -1.0,
                    "validation_top2_accuracy": -1.0,
                    "validation_regret_mean": 1e18,
                    "validation_score": -1e18,
                    "status": "validation_eval_failed",
                }
            )
            leaderboard.append(row)
            continue

        summary_path = val_eval_dir / "policy_selection_summary.json"
        if not summary_path.is_file():
            row.update(
                {
                    "validation_policy_accuracy": -1.0,
                    "validation_top2_accuracy": -1.0,
                    "validation_regret_mean": 1e18,
                    "validation_score": -1e18,
                    "status": "validation_summary_missing",
                }
            )
            leaderboard.append(row)
            continue

        val_summary = _json_load(summary_path)
        val_score, val_acc, val_regret = _trial_score(val_summary)
        val_top2 = float(val_summary.get("top2_accuracy", 0.0))
        row.update(
            {
                "validation_policy_accuracy": val_acc,
                "validation_top2_accuracy": val_top2,
                "validation_regret_mean": val_regret,
                "validation_score": float(val_score),
                "validation_n_states": int(val_summary.get("n_states", 0)),
                "status": "ok",
            }
        )
        leaderboard.append(row)

        if best_row is None or _leaderboard_sort_key(row) > _leaderboard_sort_key(best_row):
            best_row = copy.deepcopy(row)
            best_train_cfg = copy.deepcopy(train_cfg)
            best_trial_dir = trial_dir

    leaderboard_sorted = sorted(leaderboard, key=_leaderboard_sort_key, reverse=True)
    _write_leaderboard_csv(tuning_root / "leaderboard.csv", leaderboard_sorted)
    _json_dump(tuning_root / "leaderboard.json", leaderboard_sorted)

    if best_row is None or best_train_cfg is None or best_trial_dir is None:
        raise RuntimeError("No successful tuning trial completed.")

    (tuning_root / "best_trial.txt").write_text(best_row["trial_name"] + "\n", encoding="utf-8")
    _json_dump(tuning_root / "best_config.json", best_train_cfg)

    best_val_summary_path = best_trial_dir / "eval_validation" / "policy_selection_summary.json"
    if best_val_summary_path.is_file():
        shutil.copy2(best_val_summary_path, tuning_root / "best_validation_summary.json")

    best_test_dir = tuning_root / "best_test_eval"
    best_test_cmd = _build_eval_cmd(
        repo_root=repo_root,
        data_folder=data_folder,
        trained_dir=best_trial_dir / "train",
        out_dir=best_test_dir,
        eval_cfg=base_eval,
        train_cfg=best_train_cfg,
        split_name="test",
        candidate_interventions_json=candidate_json,
    )
    test_rc = _run_cmd(best_test_cmd, cwd=repo_root, log_path=tuning_root / "best_test_eval.log")
    if test_rc == 0:
        best_test_summary_path = best_test_dir / "policy_selection_summary.json"
        if best_test_summary_path.is_file():
            shutil.copy2(best_test_summary_path, tuning_root / "best_test_summary.json")

    if not bool(args.keep_all_trial_dirs):
        keep_dirs = {best_trial_dir.name, "best_test_eval"}
        for child in tuning_root.iterdir():
            if child.is_dir() and child.name not in keep_dirs:
                shutil.rmtree(child, ignore_errors=True)

    final_summary = {
        "run_name": str(args.run_name),
        "repo_root": str(repo_root),
        "config_py": str(config_py),
        "data_folder": str(data_folder),
        "num_trials_total": int(len(trials)),
        "best_trial": best_row,
        "best_config": best_train_cfg,
        "completed_at_unix": time.time(),
        "validation_selection_only": True,
        "test_used_for_tuning": False,
    }
    _json_dump(tuning_root / "tuning_summary.json", final_summary)

    print(f"BEST_TRIAL {best_row['trial_name']}")
    print(f"BEST_VALIDATION_POLICY_ACCURACY {best_row.get('validation_policy_accuracy', -1.0):.6f}")
    print(f"BEST_VALIDATION_TOP2_ACCURACY {best_row.get('validation_top2_accuracy', -1.0):.6f}")
    print(f"BEST_VALIDATION_REGRET_MEAN {best_row.get('validation_regret_mean', 1e18):.6f}")
    print(f"TUNING_ROOT {tuning_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
