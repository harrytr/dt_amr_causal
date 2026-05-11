#!/usr/bin/env python3
from __future__ import annotations

"""
build_causal_policy_dataset.py
==============================

Paper-grade dataset builder for treatment-conditional temporal prediction in the
causal AMR branch.

What this script builds
-----------------------
It produces a manifest of supervised samples of the form:

    (pre_intervention_window, action_id) -> horizon outcome under that action

where:
- the input window is a contiguous sequence of daily .pt graphs of length T;
- the action is a discrete candidate intervention (baseline included);
- the target is read from the action-conditioned trajectory at the decision day,
  using the horizon labels already written by convert_to_pt.py.

Two action-start modes are supported:

1. branch_at_decision_day   [paper-correct]
   - the baseline trajectory is generated once per seed;
   - for each decision day and action, the counterfactual trajectory is generated
     with the action activated starting at that decision day (or the next day);
   - all actions share the same pre-intervention history up to the decision day.

2. from_day1_regime_conditioning   [fallback only]
   - each action trajectory is generated once from day 1;
   - windows are taken from the same action trajectory;
   - this is fully runnable in the current branch, but it is NOT the intended
     policy-response dataset because the pre-intervention history is already
     action-conditioned.

New in this patched version
---------------------------
It can also generate the causal dataset from pbc7-style balanced base families.
That means the base trajectories are created using the same canonical
endogenous-dominant and importation-dominant simulator settings used in the
baseline predictive benchmark, then interventions branch from those cached base
states. This preserves the same family logic while still producing the manifest-
based state-action-outcome dataset required by the causal branch.

This version also adds optional parallelization across independent base plans
via --jobs, and removes forced YAML export during bulk causal dataset generation
to reduce I/O overhead.
"""

import argparse
import math
import csv
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from causal_interventions import (
    CausalInterventionSpec,
    InterventionValidationError,
    describe_intervention,
    load_interventions_json,
    validate_intervention_dict,
)


REPO_DIR = Path(__file__).resolve().parent
GENERATE_SCRIPT = REPO_DIR / "generate_amr_data.py"
CONVERT_SCRIPT = REPO_DIR / "convert_to_pt.py"


class DatasetBuildError(RuntimeError):
    """Raised when dataset generation fails."""


@dataclass(frozen=True)
class ActionSpec:
    action_id: str
    spec: Optional[CausalInterventionSpec]
    is_baseline: bool

    @property
    def action_name(self) -> str:
        if self.is_baseline or self.spec is None:
            return "baseline"
        display_name = self.spec.params.get("display_name") if isinstance(self.spec.params, dict) else None
        if display_name is not None and str(display_name).strip():
            return str(display_name).strip()
        return str(self.spec.name)

    @property
    def description(self) -> str:
        if self.is_baseline or self.spec is None:
            return "No intervention / baseline policy."
        return describe_intervention(self.spec)


@dataclass(frozen=True)
class BaseTrajectoryPlan:
    split: str
    family_name: str
    regime: str
    sim_index: int
    seed: int
    shared_noise_seed: int
    sim_args: Tuple[str, ...]


PBC7_MECHANISM_SPLIT_FAMILIES: Dict[str, Dict[str, Any]] = {
    "endog_high_train": {
        "seed_base": 4100,
        "regime": "endogenous",
        "group": "train_pool",
        "p_admit_import_cs": 0.005,
        "p_admit_import_cr": 0.005,
        "p_admit_import_is": 0.005,
        "p_admit_import_ir": 0.005,
        "daily_discharge_frac": 0.02,
        "daily_discharge_min_per_ward": 0,
        "extra_sim_args": [],
    },
    "import_high_train": {
        "seed_base": 5100,
        "regime": "importation",
        "group": "train_pool",
        "p_admit_import_cs": 0.60,
        "p_admit_import_cr": 0.60,
        "p_admit_import_is": 0.60,
        "p_admit_import_ir": 0.60,
        "daily_discharge_frac": 0.25,
        "daily_discharge_min_per_ward": 1,
        "extra_sim_args": [],
    },
    "endog_high_test": {
        "seed_base": 6100,
        "regime": "endogenous",
        "group": "test_pool",
        "p_admit_import_cs": 0.005,
        "p_admit_import_cr": 0.005,
        "p_admit_import_is": 0.005,
        "p_admit_import_ir": 0.005,
        "daily_discharge_frac": 0.02,
        "daily_discharge_min_per_ward": 0,
        "extra_sim_args": [],
    },
    "import_high_test": {
        "seed_base": 7100,
        "regime": "importation",
        "group": "test_pool",
        "p_admit_import_cs": 0.60,
        "p_admit_import_cr": 0.60,
        "p_admit_import_is": 0.60,
        "p_admit_import_ir": 0.60,
        "daily_discharge_frac": 0.25,
        "daily_discharge_min_per_ward": 1,
        "extra_sim_args": [],
    },
}


TARGET_SUFFIXES = (
    "endog_share",
    "endog_majority",
    "import_share",
    "trans_share",
    "trans_majority",
    "resistant_frac",
    "any_res_emergence",
    "cr_acq",
    "ir_inf",
    "trans_res",
    "trans_res_baseline",
    "trans_res_gain",
    "trans_import_res",
    "trans_import_res_baseline",
    "trans_import_res_gain",
    "select_res",
    "import_res",
    "endog_res",
    "transmissions",
    "total_inf",
)

_PT_NAME_RE = re.compile(r"^(?P<run>.+?)__.+?_t(?P<day>\d+)\.pt$")
_GML_NAME_RE = re.compile(r"^.+?_t(?P<day>\d+)(?:_L\d+)?\.graphml$")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _safe_mkdir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _natural_key(text: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", str(text))]


def _parse_int_list(value: str) -> List[int]:
    s = str(value or "").strip()
    if s == "":
        return []

    out: List[int] = []
    seen = set()

    for tok in s.split(","):
        tok = tok.strip()
        if tok == "":
            continue

        if ":" in tok:
            parts = [p.strip() for p in tok.split(":")]
            if len(parts) != 2:
                raise DatasetBuildError(
                    f"Invalid integer range token '{tok}'. Use start:end."
                )

            lo_s, hi_s = parts
            if not re.fullmatch(r"-?\d+", lo_s) or not re.fullmatch(r"-?\d+", hi_s):
                raise DatasetBuildError(
                    f"Invalid integer range token '{tok}'. Use start:end."
                )

            lo = int(lo_s)
            hi = int(hi_s)
            step = 1 if hi >= lo else -1

            for x in range(lo, hi + step, step):
                if x not in seen:
                    out.append(x)
                    seen.add(x)
            continue

        if not re.fullmatch(r"-?\d+", tok):
            raise DatasetBuildError(f"Invalid integer token '{tok}'.")

        x = int(tok)
        if x not in seen:
            out.append(x)
            seen.add(x)

    return out


def _parse_day_spec(value: str, *, min_day: int, max_day: int, stride: int) -> List[int]:
    s = str(value or "").strip().lower()
    if min_day > max_day:
        return []
    stride = max(1, int(stride))
    if s in {"", "auto", "all"}:
        return list(range(int(min_day), int(max_day) + 1, stride))
    if ":" in s:
        parts = s.split(":")
        if len(parts) not in {2, 3}:
            raise DatasetBuildError(f"Invalid day range spec '{value}'. Use start:end[:step].")
        lo = int(parts[0])
        hi = int(parts[1])
        st = int(parts[2]) if len(parts) == 3 else stride
        st = max(1, st)
        lo = max(int(min_day), lo)
        hi = min(int(max_day), hi)
        if lo > hi:
            return []
        return list(range(lo, hi + 1, st))
    days = sorted(set(_parse_int_list(s)))
    return [d for d in days if int(min_day) <= int(d) <= int(max_day)]


def _run_cmd(cmd: Sequence[str], *, cwd: Optional[Path] = None) -> None:
    env = dict(**__import__("os").environ)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    proc = subprocess.run(
        list(cmd),
        cwd=str(cwd) if cwd is not None else None,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise DatasetBuildError(
            f"Command failed ({proc.returncode}): {' '.join(str(x) for x in cmd)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}"
        )


def _load_action_specs(
    *,
    baseline_intervention_json: str,
    candidate_interventions_json: str,
    include_baseline: bool,
) -> List[ActionSpec]:
    actions: List[ActionSpec] = []
    baseline_spec: Optional[CausalInterventionSpec] = None
    baseline_path = str(baseline_intervention_json or "").strip()
    if baseline_path != "":
        payload = _read_json(Path(baseline_path))
        baseline_spec = validate_intervention_dict(payload)
    if include_baseline:
        actions.append(ActionSpec(action_id="baseline", spec=baseline_spec, is_baseline=True))
    candidate_path = str(candidate_interventions_json or "").strip()
    if candidate_path == "":
        raise DatasetBuildError("--candidate_interventions_json is required.")
    candidate_specs = load_interventions_json(candidate_path)
    seen_ids = {"baseline"}
    for idx, spec in enumerate(candidate_specs, start=1):
        raw = str(spec.name).strip().lower()
        suffix = raw
        if spec.target_type:
            suffix += f"__{str(spec.target_type).strip().lower()}"
        if spec.target_id:
            safe_target = re.sub(r"[^A-Za-z0-9._-]+", "-", str(spec.target_id).strip())
            suffix += f"__{safe_target}"
        aid = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-") or f"action_{idx:02d}"
        base_aid = aid
        bump = 2
        while aid in seen_ids:
            aid = f"{base_aid}_{bump}"
            bump += 1
        seen_ids.add(aid)
        actions.append(ActionSpec(action_id=aid, spec=spec, is_baseline=False))
    if not actions:
        raise DatasetBuildError("No actions were loaded.")
    return actions


def _serialize_intervention_for_generate(
    spec: Optional[CausalInterventionSpec],
    *,
    start_day: Optional[int],
    end_day: Optional[int],
) -> str:
    if spec is None:
        return ""
    payload = spec.to_dict()
    if start_day is not None:
        payload["start_day"] = int(start_day)
    if end_day is not None:
        payload["end_day"] = int(end_day)
    return json.dumps(payload, sort_keys=True)


def _supports_delayed_action_start(generate_script: Path) -> bool:
    text = generate_script.read_text(encoding="utf-8", errors="ignore")
    has_field = ("start_day:" in text and "end_day:" in text)
    has_from_dict = ('payload.get("start_day")' in text and 'payload.get("end_day")' in text)
    has_runtime_gate = ("start_day" in text and "end_day" in text and "intervention" in text)
    return bool(has_field and has_from_dict and has_runtime_gate)


def _build_generate_cmd(
    *,
    python_bin: str,
    output_dir: Path,
    seed: int,
    pair_id: str,
    role: str,
    shared_noise_seed: int,
    intervention_json: str,
    sim_args: Sequence[str],
) -> List[str]:
    cmd = [
        python_bin,
        str(GENERATE_SCRIPT),
        "--output_dir",
        str(output_dir),
        "--seed",
        str(int(seed)),
        "--causal_mode",
        "1",
        "--causal_pair_id",
        str(pair_id),
        "--causal_role",
        str(role),
        "--causal_shared_noise_seed",
        str(int(shared_noise_seed)),
    ]
    if str(intervention_json).strip() != "":
        cmd.extend(["--causal_intervention_json", str(intervention_json)])
    cmd.extend([str(x) for x in sim_args])
    return cmd


def _build_convert_cmd(*, python_bin: str, graphml_dir: Path, horizons: str, state_mode: str) -> List[str]:
    return [
        python_bin,
        str(CONVERT_SCRIPT),
        "--graphml_dir",
        str(graphml_dir),
        "--horizons",
        str(horizons),
        "--state_mode",
        str(state_mode),
    ]


def _index_pt_days(folder: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for path in sorted(folder.glob("*.pt"), key=lambda p: _natural_key(p.name)):
        m = _PT_NAME_RE.match(path.name)
        if m is None:
            continue
        out[int(m.group("day"))] = path.resolve()
    return out


def _index_graphml_days(folder: Path) -> Dict[int, Path]:
    out: Dict[int, Path] = {}
    for path in sorted(folder.glob("*.graphml"), key=lambda p: _natural_key(p.name)):
        m = _GML_NAME_RE.match(path.name)
        if m is None:
            continue
        out[int(m.group("day"))] = path.resolve()
    return out


def _extract_targets_from_pt(pt_path: Path, horizons: Sequence[int]) -> Dict[str, Any]:
    data = torch.load(str(pt_path), weights_only=False)
    out: Dict[str, Any] = {}
    for h in horizons:
        for suffix in TARGET_SUFFIXES:
            attr = f"y_h{int(h)}_{suffix}"
            if hasattr(data, attr):
                value = getattr(data, attr)
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        out[attr] = value.item()
                    else:
                        out[attr] = [float(x) for x in value.detach().cpu().view(-1).tolist()]
                else:
                    out[attr] = value
    for attr in (
        "cf_pair_id",
        "cf_role",
        "cf_intervention_name",
        "cf_intervention_target_type",
        "cf_intervention_target_id",
        "sim_id",
        "day",
        "filename",
    ):
        if hasattr(data, attr):
            value = getattr(data, attr)
            out[attr] = value.item() if torch.is_tensor(value) and value.numel() == 1 else value
    return out


def _infer_num_days_from_graphs(folder: Path) -> int:
    indexed_pt = _index_pt_days(folder)
    if indexed_pt:
        return max(indexed_pt)
    indexed_gml = _index_graphml_days(folder)
    return max(indexed_gml) if indexed_gml else 0
  

def _prepare_arm(
    *,
    python_bin: str,
    out_dir: Path,
    seed: int,
    pair_id: str,
    role: str,
    shared_noise_seed: int,
    intervention_json: str,
    sim_args: Sequence[str],
    horizons: str,
    state_mode: str,
    force: bool,
) -> None:
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    _safe_mkdir(out_dir)
    if not any(out_dir.glob("*.graphml")):
        _run_cmd(
            _build_generate_cmd(
                python_bin=python_bin,
                output_dir=out_dir,
                seed=seed,
                pair_id=pair_id,
                role=role,
                shared_noise_seed=shared_noise_seed,
                intervention_json=intervention_json,
                sim_args=sim_args,
            ),
            cwd=REPO_DIR,
        )
    if not any(out_dir.glob("*.pt")):
        _run_cmd(
            _build_convert_cmd(
                python_bin=python_bin,
                graphml_dir=out_dir,
                horizons=horizons,
                state_mode=state_mode,
            ),
            cwd=REPO_DIR,
        )


def _sim_arg_list_from_mapping(sim_args: Mapping[str, Any]) -> List[str]:
    out: List[str] = []
    for key, value in sim_args.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if key == "export_gif":
                out.append("--export_gif" if value else "--no_export_gif")
            elif value:
                out.append(flag)
        else:
            out.extend([flag, str(value)])
    return out


def _canonical_family_profiles(profile_name: str) -> Dict[str, Dict[str, Any]]:
    profile = str(profile_name or "mechanism_split").strip().lower()
    if profile != "mechanism_split":
        raise DatasetBuildError(f"Unsupported family profile '{profile_name}'.")
    return PBC7_MECHANISM_SPLIT_FAMILIES


def _build_balanced_base_plans(args: argparse.Namespace, user_sim_args: Sequence[str]) -> List[BaseTrajectoryPlan]:
    if int(args.n_sims_per_family) <= 0:
        raise DatasetBuildError(
            "--n_sims_per_family must be positive when --use_balanced_base_families is enabled."
        )
    if int(args.val_sims_per_family) < 0:
        raise DatasetBuildError("--val_sims_per_family must be >= 0.")
    if int(args.val_sims_per_family) >= int(args.n_sims_per_family):
        raise DatasetBuildError("--val_sims_per_family must be smaller than --n_sims_per_family.")

    profile = _canonical_family_profiles(args.family_profile)
    seed_overrides = {
        "endog_high_train": int(args.train_endog_seed_base),
        "import_high_train": int(args.train_import_seed_base),
        "endog_high_test": int(args.test_endog_seed_base),
        "import_high_test": int(args.test_import_seed_base),
    }

    plans: List[BaseTrajectoryPlan] = []
    for family_name in ("endog_high_train", "import_high_train", "endog_high_test", "import_high_test"):
        fam = dict(profile[family_name])
        fam["seed_base"] = int(seed_overrides[family_name])
        regime = str(fam.get("regime", "unknown"))
        group = str(fam.get("group", "train_pool"))
        for sim_index in range(int(args.n_sims_per_family)):
            if group == "test_pool":
                split = "test"
            else:
                split = "validation" if sim_index < int(args.val_sims_per_family) else "train"
            seed = int(fam["seed_base"]) + sim_index
            shared_noise_seed = int(args.shared_noise_seed_offset) + seed
            family_args = [
                "--p_admit_import_cs",
                str(float(fam["p_admit_import_cs"])),
                "--p_admit_import_cr",
                str(float(fam["p_admit_import_cr"])),
                "--p_admit_import_is",
                str(float(fam.get("p_admit_import_is", fam["p_admit_import_cs"]))),
                "--p_admit_import_ir",
                str(float(fam.get("p_admit_import_ir", fam["p_admit_import_cr"]))),
                "--daily_discharge_frac",
                str(float(fam["daily_discharge_frac"])),
                "--daily_discharge_min_per_ward",
                str(int(fam["daily_discharge_min_per_ward"])),
            ]
            family_args.extend([str(x) for x in fam.get("extra_sim_args", [])])
            full_sim_args = tuple(list(user_sim_args) + family_args)
            plans.append(
                BaseTrajectoryPlan(
                    split=split,
                    family_name=family_name,
                    regime=regime,
                    sim_index=sim_index,
                    seed=seed,
                    shared_noise_seed=shared_noise_seed,
                    sim_args=full_sim_args,
                )
            )
    return plans


def _build_seed_based_plans(args: argparse.Namespace, user_sim_args: Sequence[str]) -> List[BaseTrajectoryPlan]:
    splits: Dict[str, List[int]] = {
        "train": _parse_int_list(args.train_seeds),
        "validation": _parse_int_list(args.val_seeds),
        "test": _parse_int_list(args.test_seeds),
    }
    splits = {k: v for k, v in splits.items() if v}
    if not splits:
        raise DatasetBuildError("At least one of --train_seeds / --val_seeds / --test_seeds must be provided.")
    plans: List[BaseTrajectoryPlan] = []
    for split_name, seeds in splits.items():
        for sim_index, seed in enumerate(seeds):
            seed = int(seed)
            plans.append(
                BaseTrajectoryPlan(
                    split=split_name,
                    family_name=f"manual_{split_name}",
                    regime="manual",
                    sim_index=sim_index,
                    seed=seed,
                    shared_noise_seed=int(args.shared_noise_seed_offset) + seed,
                    sim_args=tuple(user_sim_args),
                )
            )
    return plans


def _manifest_header(horizons: Sequence[int]) -> List[str]:
    cols = [
        "split",
        "seed",
        "shared_noise_seed",
        "base_family",
        "base_regime",
        "sim_index",
        "state_id",
        "pair_id",
        "decision_day",
        "action_start_day",
        "window_T",
        "action_id",
        "action_name",
        "action_index",
        "is_baseline",
        "policy_valid",
        "action_start_mode",
        "baseline_graphml_dir",
        "baseline_pt_dir",
        "action_graphml_dir",
        "action_pt_dir",
        "window_pt_json",
        "window_graphml_json",
        "decision_pt_path",
        "decision_graphml_path",
        "baseline_intervention_json",
        "action_intervention_json",
        "action_description",
    ]
    for h in horizons:
        h_int = int(h)
        for suffix in TARGET_SUFFIXES:
            cols.append(f"y_h{h_int}_{suffix}")
        cols.extend([
            f"oracle_best_action_index_h{h_int}",
            f"oracle_best_action_id_h{h_int}",
            f"oracle_best_action_name_h{h_int}",
            f"oracle_tie_count_h{h_int}",
            f"is_oracle_best_h{h_int}",
        ])
    return cols


def _write_manifest_csv(path: Path, rows: Sequence[Mapping[str, Any]], horizons: Sequence[int]) -> None:
    header = _manifest_header(horizons)
    _safe_mkdir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            flat = {k: row.get(k, "") for k in header}
            for key in (
                "window_pt_json",
                "window_graphml_json",
                "baseline_intervention_json",
                "action_intervention_json",
            ):
                if isinstance(flat.get(key), (dict, list)):
                    flat[key] = json.dumps(flat[key], sort_keys=True)
            writer.writerow(flat)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _safe_mkdir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")


def _policy_valid_for_mode(action_start_mode: str) -> int:
    return 1 if str(action_start_mode) == "branch_at_decision_day" else 0


def _attach_baseline_relative_gain_columns(rows: List[Dict[str, Any]], horizons: Sequence[int]) -> None:
    """
    For each decision state, attach baseline oracle values and action gains.
    Gain is defined as: baseline burden - action burden.
    Positive gain therefore means the action improves over baseline.

    We compute this both for the legacy transmission-only burden and for the
    transmission+importation burden so downstream scripts can switch tasks
    without breaking older runs.
    """
    by_state: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_state.setdefault(str(row.get("state_id", "")), []).append(row)

    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    for state_id, state_rows in by_state.items():
        baseline_row = None
        for row in state_rows:
            if int(row.get("is_baseline", 0)) == 1:
                baseline_row = row
                break
        if baseline_row is None:
            raise DatasetBuildError(f"Missing baseline action for state_id='{state_id}'.")

        for h in horizons:
            h_int = int(h)

            for row in state_rows:
                trans_val = _safe_float(row.get(f"y_h{h_int}_trans_res", float("nan")))
                import_val = _safe_float(row.get(f"y_h{h_int}_import_res", float("nan")))
                row[f"y_h{h_int}_trans_import_res"] = float(trans_val + import_val)

            base_trans_val = _safe_float(baseline_row.get(f"y_h{h_int}_trans_res", float("nan")))
            base_trans_baseline_col = f"y_h{h_int}_trans_res_baseline"
            base_trans_gain_col = f"y_h{h_int}_trans_res_gain"

            base_trans_import_val = _safe_float(baseline_row.get(f"y_h{h_int}_trans_import_res", float("nan")))
            base_trans_import_baseline_col = f"y_h{h_int}_trans_import_res_baseline"
            base_trans_import_gain_col = f"y_h{h_int}_trans_import_res_gain"

            for row in state_rows:
                action_trans_val = _safe_float(row.get(f"y_h{h_int}_trans_res", float("nan")))
                row[base_trans_baseline_col] = float(base_trans_val)
                row[base_trans_gain_col] = float(base_trans_val - action_trans_val)

                action_trans_import_val = _safe_float(row.get(f"y_h{h_int}_trans_import_res", float("nan")))
                row[base_trans_import_baseline_col] = float(base_trans_import_val)
                row[base_trans_import_gain_col] = float(base_trans_import_val - action_trans_import_val)




def _attach_oracle_best_action_columns(
    rows: List[Dict[str, Any]],
    horizons: Sequence[int],
    action_order: Mapping[str, int],
) -> None:
    """
    Attach stable multiclass policy labels per decision state.

    The oracle is defined by the same baseline-relative gain columns used by the
    policy evaluator: larger gain is better. We also attach a stable per-action
    index derived from the configured action menu so the trainer can learn a
    consistent class mapping across splits and reruns.
    """
    by_state: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        state_id = str(row.get("state_id", "")).strip()
        by_state.setdefault(state_id, []).append(row)

    def _safe_float(value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return float("nan")

    for state_id, state_rows in by_state.items():
        if not state_rows:
            continue

        for row in state_rows:
            action_id = str(row.get("action_id", "")).strip()
            if action_id not in action_order:
                raise DatasetBuildError(
                    f"Action id '{action_id}' in state_id='{state_id}' is missing from the configured action order."
                )
            row["action_index"] = int(action_order[action_id])

        for h in horizons:
            h_int = int(h)
            gain_col = f"y_h{h_int}_trans_import_res_gain"
            oracle_index_col = f"oracle_best_action_index_h{h_int}"
            oracle_id_col = f"oracle_best_action_id_h{h_int}"
            oracle_name_col = f"oracle_best_action_name_h{h_int}"
            oracle_ties_col = f"oracle_tie_count_h{h_int}"
            oracle_flag_col = f"is_oracle_best_h{h_int}"

            gain_vals = [_safe_float(r.get(gain_col, float("nan"))) for r in state_rows]
            finite_pairs = [
                (idx, val)
                for idx, val in enumerate(gain_vals)
                if math.isfinite(val)
            ]
            if not finite_pairs:
                raise DatasetBuildError(
                    f"State_id='{state_id}' is missing finite oracle gain values in column '{gain_col}'."
                )

            best_val = max(val for _, val in finite_pairs)
            best_rows = [idx for idx, val in finite_pairs if math.isclose(val, best_val, rel_tol=1e-12, abs_tol=1e-12)]
            best_row_idx = min(
                best_rows,
                key=lambda idx: (
                    int(state_rows[idx].get("action_index", 10**9)),
                    str(state_rows[idx].get("action_id", "")),
                ),
            )
            oracle_row = state_rows[best_row_idx]
            oracle_action_index = int(oracle_row["action_index"])
            oracle_action_id = str(oracle_row.get("action_id", "")).strip()
            oracle_action_name = str(oracle_row.get("action_name", oracle_action_id)).strip()
            tie_count = int(len(best_rows))

            for idx, row in enumerate(state_rows):
                row[oracle_index_col] = int(oracle_action_index)
                row[oracle_id_col] = oracle_action_id
                row[oracle_name_col] = oracle_action_name
                row[oracle_ties_col] = int(tie_count)
                row[oracle_flag_col] = 1 if idx == best_row_idx else 0


def _binary_balance_summary(rows: Sequence[Mapping[str, Any]], label_attr: str) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split_name in sorted({str(r.get("split", "")) for r in rows}):
        split_rows = [
            r
            for r in rows
            if str(r.get("split", "")) == split_name and int(r.get("is_baseline", 0)) == 1
        ]
        counts = {"0": 0, "1": 0, "other": 0, "n": 0}
        for row in split_rows:
            val = row.get(label_attr)
            counts["n"] += 1
            try:
                iv = int(val)
            except Exception:
                counts["other"] += 1
                continue
            if iv == 0:
                counts["0"] += 1
            elif iv == 1:
                counts["1"] += 1
            else:
                counts["other"] += 1
        out[split_name] = counts
    return out


def _assert_two_class_splits(rows: Sequence[Mapping[str, Any]], label_attr: str) -> None:
    balance = _binary_balance_summary(rows, label_attr)
    bad: List[str] = []
    for split_name, counts in balance.items():
        if counts["n"] <= 0:
            bad.append(f"{split_name}: empty")
        elif counts["0"] <= 0 or counts["1"] <= 0:
            bad.append(f"{split_name}: 0={counts['0']} 1={counts['1']}")
    if bad:
        raise DatasetBuildError(
            f"Requested two-class split check failed for baseline label '{label_attr}'. "
            + "; ".join(bad)
        )


def _emit_rows_for_plan(
    *,
    plan: BaseTrajectoryPlan,
    actions: Sequence[ActionSpec],
    python_bin: str,
    traj_root: Path,
    horizons: Sequence[int],
    args: argparse.Namespace,
    baseline_json: str,
    manifest_rows: List[Dict[str, Any]],
) -> None:
    baseline_root = traj_root / "baseline" / "graphml"
    _safe_mkdir(traj_root)
    _prepare_arm(
        python_bin=python_bin,
        out_dir=baseline_root,
        seed=plan.seed,
        pair_id=f"{plan.family_name}__sim{plan.sim_index:02d}__baseline",
        role="factual",
        shared_noise_seed=plan.shared_noise_seed,
        intervention_json=baseline_json,
        sim_args=plan.sim_args,
        horizons=args.horizons,
        state_mode=args.state_mode,
        force=bool(args.force),
    )
    baseline_pt_by_day = _index_pt_days(baseline_root)
    baseline_graphml_by_day = _index_graphml_days(baseline_root)
    num_days = _infer_num_days_from_graphs(baseline_root)
    if num_days <= 0:
        raise DatasetBuildError(f"No day files found in baseline folder: {baseline_root}")

    max_h = max(sorted({int(x) for x in _parse_int_list(args.horizons)}))
    min_decision_day = int(args.window_T)
    max_decision_day = int(num_days) - int(max_h)
    decision_days = _parse_day_spec(
        args.decision_days,
        min_day=min_decision_day,
        max_day=max_decision_day,
        stride=int(args.decision_stride),
    )
    if not decision_days:
        raise DatasetBuildError(
            f"No valid decision days for seed={plan.seed}. "
            f"window_T={args.window_T}, max_h={max_h}, num_days={num_days}."
        )

    if str(args.action_start_mode) == "from_day1_regime_conditioning":
        arm_info: Dict[str, Dict[str, Any]] = {}
        for action in actions:
            if action.is_baseline:
                arm_info[action.action_id] = {
                    "graphml_dir": baseline_root,
                    "pt_dir": baseline_root,
                    "pt_by_day": baseline_pt_by_day,
                    "graphml_by_day": baseline_graphml_by_day,
                    "intervention_json": baseline_json,
                }
                continue
            action_root = traj_root / "actions_from_day1" / action.action_id / "graphml"
            action_json = _serialize_intervention_for_generate(action.spec, start_day=None, end_day=None)
            _prepare_arm(
                python_bin=python_bin,
                out_dir=action_root,
                seed=plan.seed,
                pair_id=f"{plan.family_name}__sim{plan.sim_index:02d}__{action.action_id}__fromday1",
                role="counterfactual",
                shared_noise_seed=plan.shared_noise_seed,
                intervention_json=action_json,
                sim_args=plan.sim_args,
                horizons=args.horizons,
                state_mode=args.state_mode,
                force=bool(args.force),
            )
            arm_info[action.action_id] = {
                "graphml_dir": action_root,
                "pt_dir": action_root,
                "pt_by_day": _index_pt_days(action_root),
                "graphml_by_day": _index_graphml_days(action_root),
                "intervention_json": action_json,
            }

        for decision_day in decision_days:
            window_days = list(range(decision_day - int(args.window_T) + 1, decision_day + 1))
            for action in actions:
                info = arm_info[action.action_id]
                pt_by_day = info["pt_by_day"]
                if not all(d in pt_by_day for d in window_days):
                    continue
                if decision_day not in pt_by_day:
                    continue
                row = {
                    "split": plan.split,
                    "seed": plan.seed,
                    "shared_noise_seed": plan.shared_noise_seed,
                    "base_family": plan.family_name,
                    "base_regime": plan.regime,
                    "sim_index": plan.sim_index,
                    "state_id": f"{plan.split}__{plan.family_name}__sim{plan.sim_index:02d}__day{decision_day}",
                    "pair_id": f"{plan.split}__{plan.family_name}__sim{plan.sim_index:02d}__day{decision_day}__{action.action_id}",
                    "decision_day": int(decision_day),
                    "action_start_day": 1,
                    "window_T": int(args.window_T),
                    "action_id": action.action_id,
                    "action_name": action.action_name,
                    "is_baseline": 1 if action.is_baseline else 0,
                    "policy_valid": _policy_valid_for_mode(args.action_start_mode),
                    "action_start_mode": str(args.action_start_mode),
                    "baseline_graphml_dir": "",
                    "baseline_pt_dir": str(baseline_root),
                    "action_graphml_dir": "",
                    "action_pt_dir": str(info["pt_dir"]),
                    "window_pt_json": [str(pt_by_day[d]) for d in window_days],
                    "window_graphml_json": [],
                    "decision_pt_path": str(pt_by_day[decision_day]),
                    "decision_graphml_path": "",
                    "baseline_intervention_json": json.loads(baseline_json) if baseline_json else {},
                    "action_intervention_json": json.loads(info["intervention_json"])
                    if info["intervention_json"]
                    else {},
                    "action_description": action.description,
                }
                row.update(_extract_targets_from_pt(Path(pt_by_day[decision_day]), horizons))
                manifest_rows.append(row)
        return

    for decision_day in decision_days:
        action_start_day = int(decision_day) if str(args.decision_applies_from) == "same_day" else int(decision_day) + 1
        if action_start_day > num_days:
            continue
        window_days = list(range(decision_day - int(args.window_T) + 1, decision_day + 1))
        if not all(d in baseline_pt_by_day for d in window_days):
            continue
        window_pt = [str(baseline_pt_by_day[d]) for d in window_days]
        window_graphml: List[str] = []
        for action in actions:
            if action.is_baseline:
                info = {
                    "graphml_dir": baseline_root,
                    "pt_dir": baseline_root,
                    "pt_by_day": baseline_pt_by_day,
                    "graphml_by_day": baseline_graphml_by_day,
                    "intervention_json": baseline_json,
                }
            else:
                action_root = traj_root / f"decision_day_{decision_day}" / action.action_id / "graphml"
                action_json = _serialize_intervention_for_generate(
                    action.spec,
                    start_day=action_start_day,
                    end_day=None,
                )
                pair_id = (
                    f"{plan.split}__{plan.family_name}__sim{plan.sim_index:02d}"
                    f"__day{decision_day}__{action.action_id}"
                )
                _prepare_arm(
                    python_bin=python_bin,
                    out_dir=action_root,
                    seed=plan.seed,
                    pair_id=pair_id,
                    role="counterfactual",
                    shared_noise_seed=plan.shared_noise_seed,
                    intervention_json=action_json,
                    sim_args=plan.sim_args,
                    horizons=args.horizons,
                    state_mode=args.state_mode,
                    force=bool(args.force),
                )
                info = {
                    "graphml_dir": action_root,
                    "pt_dir": action_root,
                    "pt_by_day": _index_pt_days(action_root),
                    "graphml_by_day": _index_graphml_days(action_root),
                    "intervention_json": action_json,
                }
            pt_by_day = info["pt_by_day"]
            if decision_day not in pt_by_day:
                continue
            row = {
                "split": plan.split,
                "seed": plan.seed,
                "shared_noise_seed": plan.shared_noise_seed,
                "base_family": plan.family_name,
                "base_regime": plan.regime,
                "sim_index": plan.sim_index,
                "state_id": f"{plan.split}__{plan.family_name}__sim{plan.sim_index:02d}__day{decision_day}",
                "pair_id": f"{plan.split}__{plan.family_name}__sim{plan.sim_index:02d}__day{decision_day}__{action.action_id}",
                "decision_day": int(decision_day),
                "action_start_day": int(action_start_day),
                "window_T": int(args.window_T),
                "action_id": action.action_id,
                "action_name": action.action_name,
                "is_baseline": 1 if action.is_baseline else 0,
                "policy_valid": _policy_valid_for_mode(args.action_start_mode),
                "action_start_mode": str(args.action_start_mode),
                "baseline_graphml_dir": "",
                "baseline_pt_dir": str(baseline_root),
                "action_graphml_dir": "",
                "action_pt_dir": str(info["pt_dir"]),
                "window_pt_json": window_pt,
                "window_graphml_json": window_graphml,
                "decision_pt_path": str(pt_by_day[decision_day]),
                "decision_graphml_path": "",
                "baseline_intervention_json": json.loads(baseline_json) if baseline_json else {},
                "action_intervention_json": json.loads(info["intervention_json"]) if info["intervention_json"] else {},
                "action_description": action.description,
            }
            row.update(_extract_targets_from_pt(Path(pt_by_day[decision_day]), horizons))
            manifest_rows.append(row)


def _build_rows_for_plan_worker(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    plan = BaseTrajectoryPlan(**payload["plan"])
    actions_payload = payload["actions"]
    actions = [
        ActionSpec(
            action_id=str(a["action_id"]),
            spec=(None if a["spec"] is None else validate_intervention_dict(a["spec"])),
            is_baseline=bool(a["is_baseline"]),
        )
        for a in actions_payload
    ]

    class _Args:
        pass

    args_obj = _Args()
    for k, v in payload["args"].items():
        setattr(args_obj, k, v)

    rows: List[Dict[str, Any]] = []
    _emit_rows_for_plan(
        plan=plan,
        actions=actions,
        python_bin=str(payload["python_bin"]),
        traj_root=Path(payload["traj_root"]),
        horizons=list(payload["horizons"]),
        args=args_obj,
        baseline_json=str(payload["baseline_json"]),
        manifest_rows=rows,
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Build intervention-conditioned causal policy datasets.")
    parser.add_argument("--out_root", required=True, type=str)
    parser.add_argument(
        "--state_mode",
        type=str,
        default=__import__("os").environ.get("DT_STATE_MODE", "ground_truth"),
        choices=["ground_truth", "partial_observation"],
    )
    parser.add_argument("--candidate_interventions_json", required=True, type=str)
    parser.add_argument("--baseline_intervention_json", type=str, default="")
    parser.add_argument("--include_baseline", action="store_true")
    parser.add_argument("--train_seeds", type=str, default="")
    parser.add_argument("--val_seeds", type=str, default="")
    parser.add_argument("--test_seeds", type=str, default="")
    parser.add_argument("--window_T", type=int, default=7)
    parser.add_argument("--horizons", type=str, default="7")
    parser.add_argument("--decision_days", type=str, default="auto")
    parser.add_argument("--decision_stride", type=int, default=1)
    parser.add_argument(
        "--action_start_mode",
        type=str,
        default="branch_at_decision_day",
        choices=["branch_at_decision_day", "from_day1_regime_conditioning"],
    )
    parser.add_argument(
        "--decision_applies_from",
        type=str,
        default="next_day",
        choices=["same_day", "next_day"],
    )
    parser.add_argument("--shared_noise_seed_offset", type=int, default=910000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--use_balanced_base_families", action="store_true")
    parser.add_argument("--family_profile", type=str, default="mechanism_split")
    parser.add_argument("--n_sims_per_family", type=int, default=0)
    parser.add_argument("--val_sims_per_family", type=int, default=0)
    parser.add_argument("--train_endog_seed_base", type=int, default=4100)
    parser.add_argument("--train_import_seed_base", type=int, default=5100)
    parser.add_argument("--test_endog_seed_base", type=int, default=6100)
    parser.add_argument("--test_import_seed_base", type=int, default=7100)
    parser.add_argument("--balance_label_attr", type=str, default="")
    parser.add_argument("--require_two_class_splits", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="Number of parallel plan workers.")
    parser.add_argument(
        "sim_args",
        nargs=argparse.REMAINDER,
        help="Arguments passed through to generate_amr_data.py after '--'.",
    )
    args = parser.parse_args()

    python_bin = sys.executable
    out_root = Path(args.out_root).resolve()
    _safe_mkdir(out_root)

    sim_args = list(args.sim_args)
    if sim_args and sim_args[0] == "--":
        sim_args = sim_args[1:]

    horizons = sorted({int(x) for x in _parse_int_list(args.horizons)})
    if not horizons:
        raise DatasetBuildError("--horizons must contain at least one positive integer.")
    if min(horizons) <= 0:
        raise DatasetBuildError("Horizons must be positive.")
    if int(args.window_T) <= 0:
        raise DatasetBuildError("--window_T must be positive.")

    actions = _load_action_specs(
        baseline_intervention_json=args.baseline_intervention_json,
        candidate_interventions_json=args.candidate_interventions_json,
        include_baseline=bool(args.include_baseline),
    )

    if str(args.action_start_mode) == "branch_at_decision_day" and not _supports_delayed_action_start(GENERATE_SCRIPT):
        raise DatasetBuildError(
            "branch_at_decision_day was requested, but the current generate_amr_data.py "
            "does not appear to honour delayed start_day / end_day fields in causal "
            "intervention payloads. Use --action_start_mode from_day1_regime_conditioning "
            "only as a temporary fallback."
        )

    if bool(args.use_balanced_base_families):
        plans = _build_balanced_base_plans(args, sim_args)
    else:
        plans = _build_seed_based_plans(args, sim_args)

    summary: Dict[str, Any] = {
        "state_mode": str(args.state_mode),
        "window_T": int(args.window_T),
        "horizons": horizons,
        "decision_days": str(args.decision_days),
        "decision_stride": int(args.decision_stride),
        "action_start_mode": str(args.action_start_mode),
        "decision_applies_from": str(args.decision_applies_from),
        "sim_args": list(sim_args),
        "use_balanced_base_families": bool(args.use_balanced_base_families),
        "family_profile": str(args.family_profile),
        "n_sims_per_family": int(args.n_sims_per_family),
        "val_sims_per_family": int(args.val_sims_per_family),
        "actions": [
            {
                "action_id": a.action_id,
                "action_name": a.action_name,
                "action_index": int(i),
                "is_baseline": bool(a.is_baseline),
                "description": a.description,
                "payload": None if a.spec is None else a.spec.to_dict(),
            }
            for i, a in enumerate(actions)
        ],
    }
    _write_json(out_root / "dataset_config.json", summary)

    manifest_rows: List[Dict[str, Any]] = []
    baseline_action = next((a for a in actions if a.is_baseline), None)
    baseline_json = _serialize_intervention_for_generate(
        baseline_action.spec if baseline_action is not None else None,
        start_day=None,
        end_day=None,
    )

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for plan in plans:
            split_root = out_root / str(plan.split)
            traj_root = split_root / plan.family_name / f"sim_{int(plan.sim_index):02d}"
            _emit_rows_for_plan(
                plan=plan,
                actions=actions,
                python_bin=python_bin,
                traj_root=traj_root,
                horizons=horizons,
                args=args,
                baseline_json=baseline_json,
                manifest_rows=manifest_rows,
            )
    else:
        tasks: List[Dict[str, Any]] = []
        args_payload = {
            "state_mode": args.state_mode,
            "window_T": args.window_T,
            "horizons": args.horizons,
            "decision_days": args.decision_days,
            "decision_stride": args.decision_stride,
            "action_start_mode": args.action_start_mode,
            "decision_applies_from": args.decision_applies_from,
            "force": args.force,
        }

        for plan in plans:
            split_root = out_root / str(plan.split)
            traj_root = split_root / plan.family_name / f"sim_{int(plan.sim_index):02d}"
            tasks.append(
                {
                    "plan": {
                        "split": plan.split,
                        "family_name": plan.family_name,
                        "regime": plan.regime,
                        "sim_index": plan.sim_index,
                        "seed": plan.seed,
                        "shared_noise_seed": plan.shared_noise_seed,
                        "sim_args": tuple(plan.sim_args),
                    },
                    "actions": [
                        {
                            "action_id": a.action_id,
                            "spec": None if a.spec is None else a.spec.to_dict(),
                            "is_baseline": a.is_baseline,
                        }
                        for a in actions
                    ],
                    "python_bin": python_bin,
                    "traj_root": str(traj_root),
                    "horizons": list(horizons),
                    "args": args_payload,
                    "baseline_json": baseline_json,
                }
            )

        with ProcessPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(_build_rows_for_plan_worker, task) for task in tasks]
            for fut in as_completed(futures):
                manifest_rows.extend(fut.result())

    if not manifest_rows:
        raise DatasetBuildError("No manifest rows were created.")

    manifest_rows.sort(
        key=lambda r: (
            str(r.get("split", "")),
            str(r.get("base_family", "")),
            int(r.get("sim_index", 0)),
            int(r.get("decision_day", 0)),
            str(r.get("action_id", "")),
        )
    )
    _attach_baseline_relative_gain_columns(manifest_rows, horizons)
    action_order = {str(a.action_id): int(i) for i, a in enumerate(actions)}
    _attach_oracle_best_action_columns(manifest_rows, horizons, action_order)

    _write_manifest_csv(out_root / "policy_manifest.csv", manifest_rows, horizons)
    _write_jsonl(out_root / "policy_manifest.jsonl", manifest_rows)

    by_split: Dict[str, List[Dict[str, Any]]] = {}
    for row in manifest_rows:
        split_name = "validation" if str(row["split"]).strip().lower() == "val" else str(row["split"])
        by_split.setdefault(split_name, []).append(row)

    for split_name, rows in by_split.items():
        _write_manifest_csv(out_root / f"policy_manifest_{split_name}.csv", rows, horizons)
        _write_jsonl(out_root / f"policy_manifest_{split_name}.jsonl", rows)

    summary_out = {
        **summary,
        "n_rows": len(manifest_rows),
        "n_states": len({str(r["state_id"]) for r in manifest_rows}),
        "rows_per_split": {k: len(v) for k, v in by_split.items()},
        "states_per_split": {k: len({str(r["state_id"]) for r in v}) for k, v in by_split.items()},
        "rows_per_family": {
            fam: sum(1 for r in manifest_rows if str(r.get("base_family", "")) == fam)
            for fam in sorted({str(r.get("base_family", "")) for r in manifest_rows})
        },
        "policy_valid": _policy_valid_for_mode(args.action_start_mode),
    }

    label_attr = str(args.balance_label_attr or "").strip()
    if label_attr:
        summary_out["baseline_balance"] = _binary_balance_summary(manifest_rows, label_attr)
        if bool(args.require_two_class_splits):
            _assert_two_class_splits(manifest_rows, label_attr)

    _write_json(out_root / "dataset_summary.json", summary_out)

    print(
        f"CAUSAL_POLICY_DATASET_OK out_root={out_root} rows={len(manifest_rows)} "
        f"states={summary_out['n_states']} mode={args.action_start_mode} "
        f"balanced_mode={bool(args.use_balanced_base_families)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (DatasetBuildError, InterventionValidationError, FileNotFoundError, ValueError) as exc:
        print(f"CAUSAL_POLICY_DATASET_ERROR {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1)
