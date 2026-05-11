#!/usr/bin/env python3
from __future__ import annotations

"""
optimize_policy_milp.py
=======================

MILP decision layer for the causal AMR branch.

This script takes the action-level scores emitted by evaluate_policy_selector.py
(policy_action_scores.csv), builds a binary action-selection MILP over the
available decision states, and writes an optimized intervention plan.

Refactored structure
--------------------
This version uses a Pyomo AbstractModel so that the algebra is clearly separated
from the data:

Sets
- STATES: decision states
- ACTIONS: global action identifiers
- PAIRS: feasible (state, action) pairs
- MAX_USE_GROUPS: action families with explicit global max-use limits
- MAX_USE_PAIRS: feasible (group, state, action) tuples counted toward each max-use cap
- COOLDOWN_CONFLICTS: pairs of state-action assignments that cannot both be
  selected because of action cooldown rules within a trajectory

Parameters
- utility[s, a]: optimization score for choosing action a in state s
- cost[s, a]: action cost contribution
- budget_total: optional global budget
- max_uses[g]: optional global maximum usage count for action family g
- has_budget: 0/1 switch to activate the budget constraint

Variables
- x[s, a] in {0, 1}: whether action a is selected for state s

Objective
- maximize total utility

Constraints
- exactly one action selected per state
- optional total budget
- optional per-family global max-uses
- cooldown conflict constraints

Data can be supplied through:
1. The action-score CSV emitted by evaluate_policy_selector.py
2. A JSON config describing action costs / cooldown / max-use / budget
3. A generated JSON template if no config is supplied

The optimizer uses Pyomo as the primary backend so CPLEX or another MILP solver
can be used. A SciPy MILP fallback is retained for robustness.
"""

import argparse
import csv
import io
import json
import math
import sys
from collections import Counter, defaultdict
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TextIO, Tuple

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ActionRow:
    state_id: str
    split: str
    seed: int
    decision_day: int
    action_id: str
    action_name: str
    is_baseline: int
    pred_score: float
    oracle_value: float
    row_payload: Dict[str, Any]


@dataclass(frozen=True)
class StateRecord:
    state_id: str
    split: str
    seed: int
    decision_day: int
    episode_key: str
    actions: Tuple[ActionRow, ...]


class OptimizationError(RuntimeError):
    pass


class TeeWriter:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _log(message: str, log_handle: Optional[TextIO] = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    if log_handle is not None:
        log_handle.write(line + "\n")
        log_handle.flush()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _nature_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=10, width=0.8, length=3)
    ax.grid(False)


def _savefig600(path: Path) -> None:
    plt.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")


def _load_action_score_rows(csv_path: Path, split_filter: str = "", seed_filter: str = "") -> List[ActionRow]:
    if not csv_path.exists():
        raise OptimizationError(f"Missing action-score CSV: {csv_path}")

    rows: List[ActionRow] = []
    want_split = str(split_filter).strip().lower()
    want_seed = str(seed_filter).strip()

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            split = str(raw.get("split", "")).strip().lower()
            seed = _safe_int(raw.get("seed", 0), 0)
            if want_split and split != want_split:
                continue
            if want_seed and str(seed) != want_seed:
                continue
            rows.append(
                ActionRow(
                    state_id=str(raw.get("state_id", "")).strip(),
                    split=split,
                    seed=seed,
                    decision_day=_safe_int(raw.get("decision_day", 0), 0),
                    action_id=str(raw.get("action_id", "")).strip(),
                    action_name=str(raw.get("action_name", "")).strip(),
                    is_baseline=_safe_int(raw.get("is_baseline", 0), 0),
                    pred_score=_safe_float(raw.get("pred_score", float("nan"))),
                    oracle_value=_safe_float(raw.get("oracle_value", float("nan"))),
                    row_payload=dict(raw),
                )
            )

    if not rows:
        raise OptimizationError(
            f"No action-score rows found in {csv_path} for split='{split_filter or '*'}' seed='{seed_filter or '*'}'."
        )
    return rows


def _build_states(rows: Sequence[ActionRow]) -> List[StateRecord]:
    grouped: Dict[str, List[ActionRow]] = defaultdict(list)
    for row in rows:
        if not row.state_id:
            raise OptimizationError("Encountered a row with empty state_id in policy_action_scores.csv")
        grouped[row.state_id].append(row)

    states: List[StateRecord] = []
    for state_id, group in sorted(
        grouped.items(),
        key=lambda kv: (kv[1][0].split, kv[1][0].seed, kv[1][0].decision_day, kv[0]),
    ):
        first = group[0]
        episode_key = f"{first.split}__seed_{first.seed}"
        actions_sorted = tuple(sorted(group, key=lambda r: (r.action_id, r.action_name, r.is_baseline)))
        if len({a.action_id for a in actions_sorted}) != len(actions_sorted):
            raise OptimizationError(f"Duplicate action_id entries found within state '{state_id}'.")
        states.append(
            StateRecord(
                state_id=state_id,
                split=first.split,
                seed=first.seed,
                decision_day=first.decision_day,
                episode_key=episode_key,
                actions=actions_sorted,
            )
        )

    if not states:
        raise OptimizationError("No valid states were formed from the action-score CSV.")
    return states


def _infer_direction(config: Mapping[str, Any], eval_summary: Mapping[str, Any]) -> str:
    direction = str(config.get("selection_direction", "auto")).strip().lower()
    if direction in {"maximize", "minimize"}:
        return direction

    ev_dir = str(eval_summary.get("selection_direction", "")).strip().lower()
    if ev_dir in {"maximize", "minimize"}:
        return ev_dir

    return "minimize"


def _oriented_score(raw_score: float, direction: str) -> float:
    if direction == "maximize":
        return float(raw_score)
    return float(-raw_score)


def _resolve_action_rule(action_row: ActionRow, config: Mapping[str, Any]) -> Dict[str, Any]:
    default_cost = _safe_float(config.get("default_action_cost", 0.0), 0.0)
    default_cooldown = _safe_int(config.get("default_cooldown_days", 0), 0)
    default_max_uses = config.get("default_max_uses", None)

    rules_by_id = config.get("action_rules_by_id", {}) or {}
    rules_by_name = config.get("action_rules_by_name", {}) or {}

    has_id_rule = action_row.action_id in rules_by_id and isinstance(rules_by_id[action_row.action_id], Mapping)
    has_name_rule = action_row.action_name in rules_by_name and isinstance(rules_by_name[action_row.action_name], Mapping)

    rule: Dict[str, Any] = {}
    if has_id_rule:
        rule.update(dict(rules_by_id[action_row.action_id]))

    if has_name_rule:
        for k, v in dict(rules_by_name[action_row.action_name]).items():
            rule.setdefault(k, v)

    default_constraint_group = action_row.action_id
    if (not has_id_rule) and has_name_rule and str(action_row.action_name).strip() != "":
        default_constraint_group = str(action_row.action_name).strip()

    explicit_constraint_group = str(rule.get("constraint_group", rule.get("rule_group", ""))).strip()
    explicit_cooldown_group = str(rule.get("cooldown_group", explicit_constraint_group)).strip()
    explicit_max_uses_group = str(rule.get("max_uses_group", explicit_constraint_group)).strip()

    constraint_group = explicit_constraint_group or default_constraint_group
    cooldown_group = explicit_cooldown_group or constraint_group
    max_uses_group = explicit_max_uses_group or constraint_group

    out = {
        "cost": _safe_float(rule.get("cost", default_cost), default_cost),
        "cooldown_days": max(0, _safe_int(rule.get("cooldown_days", default_cooldown), default_cooldown)),
        "max_uses": rule.get("max_uses", default_max_uses),
        "constraint_group": constraint_group,
        "cooldown_group": cooldown_group,
        "max_uses_group": max_uses_group,
        "rule_source": "id" if has_id_rule else ("name" if has_name_rule else "default"),
    }
    if out["max_uses"] is not None:
        out["max_uses"] = max(0, _safe_int(out["max_uses"], 0))
    return out


def _normalize_solver_executable(path_value: str) -> str:
    path_clean = str(path_value).strip()
    if not path_clean:
        return ""
    return str(Path(path_clean).expanduser().resolve())


def _default_mac_cplex_executable() -> str:
    candidates = [
        Path("~/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/cplex").expanduser(),
        Path("~/Applications/CPLEX_Studio2212/cplex/bin/x86-64_osx/cplex").expanduser(),
    ]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return str(cand.resolve())
    return ""


def _build_template_config(states: Sequence[StateRecord], out_path: Path) -> None:
    seen: Dict[str, Dict[str, Any]] = {}
    for state in states:
        for action in state.actions:
            if action.action_id in seen:
                continue
            seen[action.action_id] = {
                "action_name": action.action_name,
                "cost": 0.0 if action.is_baseline else 1.0,
                "cooldown_days": 0,
                "max_uses": None,
            }

    template = {
        "selection_direction": "auto",
        "default_action_cost": 0.0,
        "default_cooldown_days": 0,
        "default_max_uses": None,
        "budget_total": None,
        "action_rules_by_id": seen,
        "action_rules_by_name": {},
    }
    _write_json(out_path, template)


def _load_or_create_config(config_path: Path, states: Sequence[StateRecord], out_dir: Path) -> Dict[str, Any]:
    if config_path.exists():
        payload = _read_json(config_path)
        if not isinstance(payload, Mapping):
            raise OptimizationError(f"MILP config must be a JSON object: {config_path}")
        return dict(payload)

    template_path = out_dir / "milp_policy_config_template.json"
    _build_template_config(states, template_path)
    return {
        "selection_direction": "auto",
        "default_action_cost": 0.0,
        "default_cooldown_days": 0,
        "default_max_uses": None,
        "budget_total": None,
        "action_rules_by_id": {},
        "action_rules_by_name": {},
        "template_written_to": str(template_path),
    }


def _collect_max_use_group_data(
    states: Sequence[StateRecord],
    pair_metrics: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> Tuple[Dict[str, int], List[Tuple[str, str, str]]]:
    group_to_pairs: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    group_to_limits: Dict[str, List[int]] = defaultdict(list)

    for state in states:
        for action in state.actions:
            pair = (state.state_id, action.action_id)
            metric = pair_metrics[pair]
            max_uses = metric.get("max_uses", None)
            if max_uses is None:
                continue
            group = str(metric.get("max_uses_group", action.action_id)).strip() or action.action_id
            group_to_pairs[group].append(pair)
            group_to_limits[group].append(int(max_uses))

    max_uses_by_group = {
        str(group): min(int(v) for v in vals)
        for group, vals in group_to_limits.items()
        if len(vals) > 0
    }
    max_use_pairs = [
        (str(group), str(state_id), str(action_id))
        for group in sorted(max_uses_by_group.keys())
        for state_id, action_id in sorted(group_to_pairs[group])
    ]
    return max_uses_by_group, max_use_pairs


def _collect_cooldown_conflicts(
    states: Sequence[StateRecord],
    pair_metrics: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Tuple[str, str, str, str]]:
    conflicts: set[Tuple[str, str, str, str]] = set()
    states_by_episode: Dict[str, List[StateRecord]] = defaultdict(list)
    for state in states:
        states_by_episode[state.episode_key].append(state)

    for _, episode_states in states_by_episode.items():
        ordered_states = sorted(episode_states, key=lambda s: (s.decision_day, s.state_id))
        cooldown_days_by_group: Dict[str, int] = {}
        pairs_by_state_group: Dict[Tuple[str, str], List[Tuple[str, str]]] = defaultdict(list)

        for state in ordered_states:
            for action in state.actions:
                pair = (state.state_id, action.action_id)
                metric = pair_metrics[pair]
                if int(metric.get("is_baseline", 0)) == 1:
                    continue
                cooldown_days = int(metric.get("cooldown_days", 0))
                if cooldown_days <= 0:
                    continue
                group = str(metric.get("cooldown_group", action.action_id)).strip() or action.action_id
                cooldown_days_by_group[group] = max(cooldown_days_by_group.get(group, 0), cooldown_days)
                pairs_by_state_group[(state.state_id, group)].append(pair)

        for group, cooldown_days in sorted(cooldown_days_by_group.items()):
            if cooldown_days <= 0:
                continue
            n_states = len(ordered_states)
            for i in range(n_states):
                s_i = ordered_states[i]
                pairs_i = pairs_by_state_group.get((s_i.state_id, group), [])
                if len(pairs_i) == 0:
                    continue
                for j in range(i + 1, n_states):
                    s_j = ordered_states[j]
                    if int(s_j.decision_day) - int(s_i.decision_day) > cooldown_days:
                        break
                    pairs_j = pairs_by_state_group.get((s_j.state_id, group), [])
                    if len(pairs_j) == 0:
                        continue
                    for p_i in pairs_i:
                        for p_j in pairs_j:
                            conflicts.add((p_i[0], p_i[1], p_j[0], p_j[1]))

    return sorted(conflicts)


def _load_evaluable_state_ids_from_per_state_csv(
    csv_path: Path,
    split_filter: str = "",
    seed_filter: str = "",
) -> List[str]:
    if not csv_path.exists():
        return []

    wanted_split = str(split_filter).strip().lower()
    wanted_seed = str(seed_filter).strip()
    state_ids: List[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            split = str(raw.get("split", "")).strip().lower()
            seed = _safe_int(raw.get("seed", 0), 0)
            if wanted_split and split != wanted_split:
                continue
            if wanted_seed and str(seed) != wanted_seed:
                continue
            state_id = str(raw.get("state_id", "")).strip()
            if state_id != "":
                state_ids.append(state_id)
    return sorted(set(state_ids))


def _resolve_action_score_source(
    eval_dir: Path,
    split_filter: str = "",
    seed_filter: str = "",
) -> Tuple[Path, List[str], str]:
    split_clean = str(split_filter).strip()
    action_candidates: List[Tuple[Path, str]] = []
    if split_clean != "":
        action_candidates.extend([
            (eval_dir / "policy_action_scores_requested_split.csv", "requested_split"),
            (eval_dir / "policy_action_scores.csv", "all_splits"),
            (eval_dir / "policy_action_scores_requested_split_all_rows.csv", "requested_split_all_rows"),
            (eval_dir / "policy_action_scores_all_rows.csv", "all_rows"),
        ])
        per_state_csv = eval_dir / "policy_selection_per_state_requested_split.csv"
    else:
        action_candidates.extend([
            (eval_dir / "policy_action_scores.csv", "all_splits"),
            (eval_dir / "policy_action_scores_all_rows.csv", "all_rows"),
        ])
        per_state_csv = eval_dir / "policy_selection_per_state.csv"

    action_csv: Optional[Path] = None
    source_tag = ""
    for cand, tag in action_candidates:
        if cand.exists():
            action_csv = cand
            source_tag = tag
            break

    if action_csv is None:
        raise OptimizationError(
            "Could not find an action-score CSV in the evaluation directory. "
            "Expected one of policy_action_scores.csv or policy_action_scores_requested_split.csv."
        )

    evaluable_state_ids = _load_evaluable_state_ids_from_per_state_csv(
        per_state_csv,
        split_filter=split_filter,
        seed_filter=seed_filter,
    )
    return action_csv, evaluable_state_ids, source_tag


def _compute_pair_metrics(
    states: Sequence[StateRecord],
    config: Mapping[str, Any],
    direction: str,
    score_transform: str,
) -> Tuple[Dict[Tuple[str, str], Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    state_info: Dict[str, Dict[str, Any]] = {}
    pair_metrics: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for state in states:
        baseline_action: Optional[ActionRow] = None
        for action in state.actions:
            if int(action.is_baseline) == 1:
                baseline_action = action
                break

        baseline_oriented = _oriented_score(baseline_action.pred_score, direction) if baseline_action is not None else 0.0
        state_info[state.state_id] = {
            "state_id": state.state_id,
            "split": state.split,
            "seed": state.seed,
            "decision_day": state.decision_day,
            "episode_key": state.episode_key,
            "baseline_action_id": "" if baseline_action is None else baseline_action.action_id,
            "baseline_action_name": "" if baseline_action is None else baseline_action.action_name,
            "baseline_pred_score": float("nan") if baseline_action is None else float(baseline_action.pred_score),
            "baseline_oriented_score": float(baseline_oriented),
        }

        for action in state.actions:
            oriented = _oriented_score(action.pred_score, direction)
            gain_vs_baseline = float(oriented - baseline_oriented)

            if score_transform == "raw":
                utility = float(oriented)
            elif score_transform == "nonnegative_baseline_delta":
                utility = float(max(0.0, gain_vs_baseline))
            else:
                utility = float(gain_vs_baseline)

            rule = _resolve_action_rule(action, config)
            pair_metrics[(state.state_id, action.action_id)] = {
                "utility": utility,
                "gain_vs_baseline": gain_vs_baseline,
                "pred_score": float(action.pred_score),
                "oriented_score": float(oriented),
                "action_name": action.action_name,
                "is_baseline": int(action.is_baseline),
                "cost": float(rule["cost"]),
                "cooldown_days": int(rule["cooldown_days"]),
                "max_uses": rule["max_uses"],
                "constraint_group": str(rule["constraint_group"]),
                "cooldown_group": str(rule["cooldown_group"]),
                "max_uses_group": str(rule["max_uses_group"]),
                "rule_source": str(rule["rule_source"]),
            }

    return pair_metrics, state_info


def _find_pyomo_solver(
    name: str,
    time_limit_sec: Optional[int],
    mip_gap: Optional[float],
    cplex_executable: str = "",
) -> Tuple[str, Any]:
    try:
        from pyomo.opt import SolverFactory
    except Exception as exc:
        raise OptimizationError(
            "Pyomo is required for optimize_policy_milp.py. Install pyomo and a MILP solver such as CPLEX."
        ) from exc

    exe_path = _normalize_solver_executable(cplex_executable)
    candidates = [str(name).strip()] if str(name).strip() else []
    if not candidates or candidates == ["auto"]:
        candidates = []
        if exe_path:
            candidates.extend(["cplex", "cplex_direct"])
        candidates.extend(["cplex_direct", "cplex", "appsi_highs", "highs", "glpk"])

    last_err: Optional[str] = None
    seen: set[str] = set()

    for cand in candidates:
        cand = str(cand).strip()
        if not cand or cand in seen:
            continue
        seen.add(cand)

        solver_kwargs: Dict[str, Any] = {}
        if cand == "cplex" and exe_path:
            solver_kwargs["executable"] = exe_path

        try:
            solver = SolverFactory(cand, **solver_kwargs)
            if solver is None:
                continue
            available = solver.available(exception_flag=False)
            if not available:
                continue

            if cand in {"cplex", "cplex_direct"}:
                try:
                    if time_limit_sec is not None:
                        solver.options["timelimit"] = int(time_limit_sec)
                    if mip_gap is not None:
                        solver.options["mipgap"] = float(mip_gap)
                except Exception:
                    pass
            elif cand in {"highs", "appsi_highs"}:
                try:
                    if time_limit_sec is not None:
                        solver.options["time_limit"] = int(time_limit_sec)
                    if mip_gap is not None:
                        solver.options["mip_rel_gap"] = float(mip_gap)
                except Exception:
                    pass

            return cand, solver
        except Exception as exc:
            last_err = str(exc)
            continue

    raise OptimizationError(
        "No suitable MILP solver was available through Pyomo. Tried: "
        + ", ".join(candidates)
        + (f". Last error: {last_err}" if last_err else "")
    )


def _build_abstract_model_data(
    states: Sequence[StateRecord],
    pair_metrics: Mapping[Tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    state_ids = [s.state_id for s in states]
    action_ids = sorted({a.action_id for s in states for a in s.actions})
    feasible_pairs = [(s.state_id, a.action_id) for s in states for a in s.actions]

    cost_param: Dict[Tuple[str, str], float] = {}
    utility_param: Dict[Tuple[str, str], float] = {}
    pair_is_feasible: Dict[Tuple[str, str], int] = {}

    for pair in feasible_pairs:
        cost_param[pair] = float(pair_metrics[pair]["cost"])
        utility_param[pair] = float(pair_metrics[pair]["utility"])
        pair_is_feasible[pair] = 1

    max_uses_by_group, max_use_pairs = _collect_max_use_group_data(states, pair_metrics)
    cooldown_conflicts = _collect_cooldown_conflicts(states, pair_metrics)

    budget_total = config.get("budget_total", None)
    has_budget = 0
    budget_value = 0.0
    if budget_total not in {None, "", "null"}:
        budget_value = _safe_float(budget_total, float("nan"))
        if not math.isfinite(budget_value):
            raise OptimizationError(f"Invalid budget_total in MILP config: {budget_total}")
        has_budget = 1

    data: Dict[str, Any] = {
        "STATES": state_ids,
        "ACTIONS": action_ids,
        "PAIRS": feasible_pairs,
        "MAX_USE_GROUPS": sorted(max_uses_by_group.keys()),
        "MAX_USE_PAIRS": max_use_pairs,
        "COOLDOWN_CONFLICTS": cooldown_conflicts,
        "utility": utility_param,
        "cost": cost_param,
        "pair_is_feasible": pair_is_feasible,
        "max_uses": max_uses_by_group,
        "has_budget": int(has_budget),
        "budget_total": float(budget_value),
    }
    return data


def _build_pyomo_abstract_model() -> Any:
    try:
        from pyomo.environ import (
            AbstractModel,
            Binary,
            Constraint,
            NonNegativeIntegers,
            Objective,
            Param,
            Set,
            Var,
            maximize,
        )
    except Exception as exc:
        raise OptimizationError("Pyomo is required to build the abstract MILP model.") from exc

    model = AbstractModel()

    model.STATES = Set(ordered=True)
    model.ACTIONS = Set(ordered=True)
    model.PAIRS = Set(dimen=2, ordered=True)
    model.MAX_USE_GROUPS = Set(ordered=True)
    model.MAX_USE_PAIRS = Set(dimen=3, ordered=True)
    model.COOLDOWN_CONFLICTS = Set(dimen=4, ordered=True)

    model.utility = Param(model.PAIRS)
    model.cost = Param(model.PAIRS)
    model.pair_is_feasible = Param(model.PAIRS, default=1)
    model.max_uses = Param(model.MAX_USE_GROUPS, within=NonNegativeIntegers)
    model.has_budget = Param()
    model.budget_total = Param()

    model.x = Var(model.PAIRS, within=Binary)

    def objective_rule(m: Any) -> Any:
        return sum(m.utility[s, a] * m.x[s, a] for (s, a) in m.PAIRS)

    model.objective = Objective(rule=objective_rule, sense=maximize)

    def one_action_per_state_rule(m: Any, state_id: str) -> Any:
        return sum(m.x[s, a] for (s, a) in m.PAIRS if s == state_id) == 1

    model.one_action_per_state = Constraint(model.STATES, rule=one_action_per_state_rule)

    def budget_rule(m: Any) -> Any:
        if int(m.has_budget) == 0:
            return Constraint.Skip
        return sum(m.cost[s, a] * m.x[s, a] for (s, a) in m.PAIRS) <= m.budget_total

    model.budget_constraint = Constraint(rule=budget_rule)

    def max_uses_rule(m: Any, group_id: str) -> Any:
        return sum(m.x[s, a] for (g, s, a) in m.MAX_USE_PAIRS if g == group_id) <= m.max_uses[group_id]

    model.max_uses_constraint = Constraint(model.MAX_USE_GROUPS, rule=max_uses_rule)

    def cooldown_rule(m: Any, s1: str, a1: str, s2: str, a2: str) -> Any:
        return m.x[s1, a1] + m.x[s2, a2] <= 1

    model.cooldown_constraint = Constraint(model.COOLDOWN_CONFLICTS, rule=cooldown_rule)

    return model


def _instantiate_abstract_model(model_data: Mapping[str, Any]) -> Any:
    model = _build_pyomo_abstract_model()
    pyomo_data = {
        None: {
            "STATES": {None: list(model_data["STATES"])},
            "ACTIONS": {None: list(model_data["ACTIONS"])},
            "PAIRS": {None: list(model_data["PAIRS"])},
            "MAX_USE_GROUPS": {None: list(model_data["MAX_USE_GROUPS"])},
            "MAX_USE_PAIRS": {None: list(model_data["MAX_USE_PAIRS"])},
            "COOLDOWN_CONFLICTS": {None: list(model_data["COOLDOWN_CONFLICTS"])},
            "utility": dict(model_data["utility"]),
            "cost": dict(model_data["cost"]),
            "pair_is_feasible": dict(model_data["pair_is_feasible"]),
            "max_uses": dict(model_data["max_uses"]),
            "has_budget": {None: int(model_data["has_budget"])},
            "budget_total": {None: float(model_data["budget_total"])},
        }
    }
    return model.create_instance(pyomo_data)


def _extract_plan_rows_from_binary_map(
    x_selected: Mapping[Tuple[str, str], float],
    states: Sequence[StateRecord],
    pair_metrics: Mapping[Tuple[str, str], Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    plan_rows: List[Dict[str, Any]] = []
    for state in sorted(states, key=lambda s: (s.split, s.seed, s.decision_day, s.state_id)):
        chosen_action_id: Optional[str] = None
        best_val = -1.0
        for action in state.actions:
            pair = (state.state_id, action.action_id)
            val = float(x_selected.get(pair, 0.0))
            if val > best_val:
                best_val = val
                chosen_action_id = action.action_id
        if chosen_action_id is None:
            raise OptimizationError(f"No action selected for state '{state.state_id}'.")

        metric = pair_metrics[(state.state_id, chosen_action_id)]
        plan_rows.append(
            {
                "state_id": state.state_id,
                "split": state.split,
                "seed": state.seed,
                "decision_day": state.decision_day,
                "episode_key": state.episode_key,
                "chosen_action_id": chosen_action_id,
                "chosen_action_name": str(metric["action_name"]),
                "pred_score": float(metric["pred_score"]),
                "utility": float(metric["utility"]),
                "gain_vs_baseline": float(metric["gain_vs_baseline"]),
                "cost": float(metric["cost"]),
                "is_baseline": int(metric["is_baseline"]),
                "constraint_group": str(metric.get("constraint_group", chosen_action_id)),
                "cooldown_group": str(metric.get("cooldown_group", chosen_action_id)),
                "max_uses_group": str(metric.get("max_uses_group", chosen_action_id)),
            }
        )
    return plan_rows


def _solve_milp_scipy(
    states: Sequence[StateRecord],
    pair_metrics: Mapping[Tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    log_handle: Optional[TextIO] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from scipy.optimize import Bounds, LinearConstraint, milp

    _log("Using SciPy MILP fallback.", log_handle=log_handle)

    state_ids = [s.state_id for s in states]
    action_pairs = [(s.state_id, a.action_id) for s in states for a in s.actions]
    pair_to_idx = {pair: idx for idx, pair in enumerate(action_pairs)}

    constraints_rows: List[np.ndarray] = []
    lb_list: List[float] = []
    ub_list: List[float] = []

    for state_id in state_ids:
        row = np.zeros(len(action_pairs), dtype=float)
        for pair in action_pairs:
            if pair[0] == state_id:
                row[pair_to_idx[pair]] = 1.0
        constraints_rows.append(row)
        lb_list.append(1.0)
        ub_list.append(1.0)

    budget_total = config.get("budget_total", None)
    if budget_total not in {None, "", "null"}:
        budget_value = _safe_float(budget_total, float("nan"))
        if not math.isfinite(budget_value):
            raise OptimizationError(f"Invalid budget_total in MILP config: {budget_total}")
        row = np.zeros(len(action_pairs), dtype=float)
        for idx, pair in enumerate(action_pairs):
            row[idx] = float(pair_metrics[pair]["cost"])
        constraints_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(float(budget_value))

    max_uses_by_group, max_use_pairs = _collect_max_use_group_data(states, pair_metrics)
    for group_id in sorted(max_uses_by_group.keys()):
        row = np.zeros(len(action_pairs), dtype=float)
        for g, state_id, action_id in max_use_pairs:
            if g != group_id:
                continue
            pair = (state_id, action_id)
            if pair in pair_to_idx:
                row[pair_to_idx[pair]] = 1.0
        constraints_rows.append(row)
        lb_list.append(-np.inf)
        ub_list.append(float(max_uses_by_group[group_id]))

    for s1, a1, s2, a2 in _collect_cooldown_conflicts(states, pair_metrics):
        p_i = (s1, a1)
        p_j = (s2, a2)
        if p_i in pair_to_idx and p_j in pair_to_idx:
            row = np.zeros(len(action_pairs), dtype=float)
            row[pair_to_idx[p_i]] = 1.0
            row[pair_to_idx[p_j]] = 1.0
            constraints_rows.append(row)
            lb_list.append(-np.inf)
            ub_list.append(1.0)

    c = np.asarray([-float(pair_metrics[pair]["utility"]) for pair in action_pairs], dtype=float)
    A = np.vstack(constraints_rows) if constraints_rows else np.zeros((0, len(action_pairs)), dtype=float)
    lb = np.asarray(lb_list, dtype=float) if lb_list else np.zeros(0, dtype=float)
    ub = np.asarray(ub_list, dtype=float) if ub_list else np.zeros(0, dtype=float)

    bounds = Bounds(lb=np.zeros(len(action_pairs), dtype=float), ub=np.ones(len(action_pairs), dtype=float))
    integrality = np.ones(len(action_pairs), dtype=int)
    constraints = LinearConstraint(A, lb, ub) if len(lb_list) > 0 else ()

    result = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
    if not bool(getattr(result, "success", False)):
        raise OptimizationError(
            f"SciPy MILP failed: status={getattr(result, 'status', 'unknown')} "
            f"message={getattr(result, 'message', '')}"
        )

    x = np.asarray(result.x, dtype=float)
    x = np.where(x >= 0.5, 1.0, 0.0)
    x_map = {pair: float(x[idx]) for idx, pair in enumerate(action_pairs)}

    plan_rows = _extract_plan_rows_from_binary_map(x_map, states, pair_metrics)
    summary = {
        "solver_name": "scipy_milp",
        "solver_status": str(getattr(result, "status", "")),
        "termination_condition": str(getattr(result, "message", "success")),
        "objective_value": float(sum(float(r["utility"]) for r in plan_rows)),
        "n_states": len(states),
        "n_action_pairs": len(action_pairs),
        "total_cost": float(sum(float(r["cost"]) for r in plan_rows)),
        "total_gain_vs_baseline": float(sum(float(r["gain_vs_baseline"]) for r in plan_rows)),
        "selected_action_counts": dict(sorted(Counter(str(r["chosen_action_id"]) for r in plan_rows).items())),
        "selected_constraint_group_counts": dict(sorted(Counter(str(r.get("constraint_group", r["chosen_action_id"])) for r in plan_rows).items())),
    }
    return plan_rows, summary


def _solve_with_pyomo_logging(
    solver: Any,
    instance: Any,
    solver_name: str,
    solver_log_path: Path,
    log_handle: Optional[TextIO] = None,
) -> Any:
    _log(f"Starting solver '{solver_name}'. Solver transcript -> {solver_log_path}", log_handle=log_handle)

    solve_kwargs: Dict[str, Any] = {"tee": True}

    try:
        solve_kwargs["logfile"] = str(solver_log_path)
    except Exception:
        pass

    tee_stdout = TeeWriter(sys.stdout, log_handle) if log_handle is not None else sys.stdout
    tee_stderr = TeeWriter(sys.stderr, log_handle) if log_handle is not None else sys.stderr

    with solver_log_path.open("a", encoding="utf-8") as solver_f:
        solver_f.write("\n" + "=" * 80 + "\n")
        solver_f.write(f"SOLVER RUN START: {datetime.now().isoformat()}\n")
        solver_f.write("=" * 80 + "\n")
        solver_f.flush()

        combined_stream = TeeWriter(tee_stdout, solver_f)
        combined_err_stream = TeeWriter(tee_stderr, solver_f)

        with redirect_stdout(combined_stream), redirect_stderr(combined_err_stream):
            results = solver.solve(instance, **solve_kwargs)

        solver_f.write("\n" + "=" * 80 + "\n")
        solver_f.write(f"SOLVER RUN END: {datetime.now().isoformat()}\n")
        solver_f.write("=" * 80 + "\n")
        solver_f.flush()

    _log(f"Finished solver '{solver_name}'.", log_handle=log_handle)
    return results


def _solve_milp(
    states: Sequence[StateRecord],
    pair_metrics: Mapping[Tuple[str, str], Mapping[str, Any]],
    config: Mapping[str, Any],
    solver_name: str,
    time_limit_sec: Optional[int],
    mip_gap: Optional[float],
    cplex_executable: str,
    solver_log_path: Path,
    log_handle: Optional[TextIO] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    solver_name_clean = str(solver_name).strip().lower()
    if solver_name_clean in {"scipy", "scipy_milp"}:
        return _solve_milp_scipy(
            states=states,
            pair_metrics=pair_metrics,
            config=config,
            log_handle=log_handle,
        )

    try:
        from pyomo.environ import value
        from pyomo.opt import TerminationCondition
    except Exception:
        _log("Pyomo import failed. Falling back to SciPy MILP.", log_handle=log_handle)
        return _solve_milp_scipy(
            states=states,
            pair_metrics=pair_metrics,
            config=config,
            log_handle=log_handle,
        )

    model_data = _build_abstract_model_data(states, pair_metrics, config)

    try:
        instance = _instantiate_abstract_model(model_data)
    except Exception as exc:
        _log(f"Pyomo AbstractModel instantiation failed ({exc}). Falling back to SciPy MILP.", log_handle=log_handle)
        return _solve_milp_scipy(
            states=states,
            pair_metrics=pair_metrics,
            config=config,
            log_handle=log_handle,
        )

    chosen_solver_name, solver = _find_pyomo_solver(
        solver_name,
        time_limit_sec=time_limit_sec,
        mip_gap=mip_gap,
        cplex_executable=cplex_executable,
    )

    results = _solve_with_pyomo_logging(
        solver=solver,
        instance=instance,
        solver_name=chosen_solver_name,
        solver_log_path=solver_log_path,
        log_handle=log_handle,
    )

    term = getattr(results.solver, "termination_condition", None)
    if term not in {"optimal", "feasible", "maxTimeLimit"}:
        try:
            from pyomo.opt import TerminationCondition

            valid_terms = {
                TerminationCondition.optimal,
                TerminationCondition.feasible,
                TerminationCondition.maxTimeLimit,
            }
            if term not in valid_terms:
                raise OptimizationError(f"MILP did not return a usable solution. termination_condition={term}")
        except Exception:
            raise OptimizationError(f"MILP did not return a usable solution. termination_condition={term}")

    x_map: Dict[Tuple[str, str], float] = {}
    for pair in model_data["PAIRS"]:
        x_map[pair] = float(value(instance.x[pair]))

    plan_rows = _extract_plan_rows_from_binary_map(x_map, states, pair_metrics)
    summary = {
        "solver_name": chosen_solver_name,
        "solver_status": str(getattr(results.solver, "status", "")),
        "termination_condition": str(term),
        "objective_value": float(value(instance.objective)),
        "n_states": len(states),
        "n_action_pairs": len(model_data["PAIRS"]),
        "total_cost": float(sum(float(r["cost"]) for r in plan_rows)),
        "total_gain_vs_baseline": float(sum(float(r["gain_vs_baseline"]) for r in plan_rows)),
        "selected_action_counts": dict(sorted(Counter(str(r["chosen_action_id"]) for r in plan_rows).items())),
        "selected_constraint_group_counts": dict(sorted(Counter(str(r.get("constraint_group", r["chosen_action_id"])) for r in plan_rows).items())),
    }
    return plan_rows, summary


def _plot_timeline(plan_rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    if not plan_rows:
        return

    ordered = sorted(plan_rows, key=lambda r: (str(r["episode_key"]), int(r["decision_day"]), str(r["state_id"])))
    episodes = list(dict.fromkeys(str(r["episode_key"]) for r in ordered))
    actions = list(dict.fromkeys(str(r["chosen_action_id"]) for r in ordered))

    ep_to_y = {ep: idx for idx, ep in enumerate(episodes)}
    act_to_color = {aid: idx for idx, aid in enumerate(actions)}

    fig_w = max(7.5, 0.8 * len({int(r["decision_day"]) for r in ordered}) + 3.0)
    fig_h = max(3.2, 0.65 * len(episodes) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    xs = [int(r["decision_day"]) for r in ordered]
    ys = [ep_to_y[str(r["episode_key"])] for r in ordered]
    cs = [act_to_color[str(r["chosen_action_id"])] for r in ordered]

    sc = ax.scatter(xs, ys, c=cs, s=80, alpha=0.90, edgecolors="white", linewidths=0.5)
    ax.set_yticks(list(ep_to_y.values()))
    ax.set_yticklabels(list(ep_to_y.keys()), fontsize=9)
    ax.set_xlabel("Decision day", fontsize=11)
    ax.set_ylabel("Trajectory", fontsize=11)
    ax.set_title("MILP-selected intervention schedule", fontsize=13, fontweight="bold")
    _nature_axes(ax)

    handles = []
    labels = []
    cmap = sc.cmap
    norm = sc.norm
    for aid, idx in act_to_color.items():
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=cmap(norm(idx)), markersize=8)
        )
        labels.append(aid)

    ax.legend(
        handles,
        labels,
        title="Action",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    fig.tight_layout()
    _savefig600(out_path)
    plt.close(fig)


def _plot_cumulative_gain(plan_rows: Sequence[Mapping[str, Any]], out_path: Path) -> None:
    if not plan_rows:
        return

    ordered = sorted(plan_rows, key=lambda r: (str(r["episode_key"]), int(r["decision_day"]), str(r["state_id"])))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    series: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    for row in ordered:
        series[str(row["episode_key"])].append((int(row["decision_day"]), float(row["gain_vs_baseline"])))

    for episode_key, vals in sorted(series.items()):
        days = [v[0] for v in vals]
        gains = np.cumsum([v[1] for v in vals])
        ax.plot(days, gains, linewidth=1.5, label=episode_key)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Decision day", fontsize=11)
    ax.set_ylabel("Cumulative predicted gain vs baseline", fontsize=11)
    ax.set_title("Accumulated predicted benefit of MILP plan", fontsize=13, fontweight="bold")
    _nature_axes(ax)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _savefig600(out_path)
    plt.close(fig)


def _write_plan_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise OptimizationError("No plan rows to write.")

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimize a minimal causal intervention plan with a MILP.")
    parser.add_argument(
        "--eval_dir",
        required=True,
        type=str,
        help="Directory from evaluate_policy_selector.py containing policy_action_scores.csv.",
    )
    parser.add_argument(
        "--config_json",
        type=str,
        default="",
        help="JSON config for action costs, cooldowns, and budget.",
    )
    parser.add_argument("--split", type=str, default="", help="Optional split filter, e.g. test.")
    parser.add_argument("--seed", type=str, default="", help="Optional single-seed optimization filter.")
    parser.add_argument(
        "--score_transform",
        type=str,
        default="baseline_delta",
        choices=["baseline_delta", "nonnegative_baseline_delta", "raw"],
        help="How to convert predicted action scores into MILP utility. For the combined CR task, baseline_delta uses the model's predicted gain vs baseline on transmission+imported resistant burden.",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="auto",
        help="Pyomo solver name, e.g. cplex, cplex_direct, appsi_highs, or scipy.",
    )
    parser.add_argument(
        "--cplex_executable",
        type=str,
        default="",
        help="Optional path to the external CPLEX executable, e.g. ~/Applications/CPLEX_Studio2212/cplex/bin/arm64_osx/cplex",
    )
    parser.add_argument("--budget_total", type=str, default="", help="Optional CLI override for total budget.")
    parser.add_argument("--time_limit_sec", type=int, default=300)
    parser.add_argument("--mip_gap", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str, default="policy_milp")
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="Optional full run log path. Defaults to <out_dir>/milp_run.log",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(args.log_file).resolve() if str(args.log_file).strip() else out_dir / "milp_run.log"
    solver_log_path = out_dir / "milp_solver.log"

    with log_path.open("a", encoding="utf-8") as log_handle:
        log_handle.write("\n" + "#" * 100 + "\n")
        log_handle.write(f"RUN START: {datetime.now().isoformat()}\n")
        log_handle.write("#" * 100 + "\n")
        log_handle.flush()

        try:
            _log(f"eval_dir={eval_dir}", log_handle=log_handle)
            _log(f"out_dir={out_dir}", log_handle=log_handle)
            _log(f"log_file={log_path}", log_handle=log_handle)
            _log(f"solver_log_file={solver_log_path}", log_handle=log_handle)

            action_csv, evaluable_state_ids, action_score_source = _resolve_action_score_source(
                eval_dir,
                split_filter=str(args.split),
                seed_filter=str(args.seed),
            )
            eval_summary_path = eval_dir / "policy_selection_summary.json"
            eval_summary = _read_json(eval_summary_path) if eval_summary_path.exists() else {}
            eval_task = str(eval_summary.get("task", "")).strip() if isinstance(eval_summary, Mapping) else ""
            eval_oracle_metric = str(eval_summary.get("oracle_metric", "")).strip() if isinstance(eval_summary, Mapping) else ""
            if eval_task or eval_oracle_metric:
                _log(
                    f"Evaluation objective: task={eval_task or '<unknown>'} oracle_metric={eval_oracle_metric or '<unknown>'}",
                    log_handle=log_handle,
                )
            
            rows = _load_action_score_rows(action_csv, split_filter=args.split, seed_filter=args.seed)
            _log(f"Loaded {len(rows)} action rows from {action_csv} (source={action_score_source})", log_handle=log_handle)

            if evaluable_state_ids:
                before_rows = len(rows)
                evaluable_state_set = set(evaluable_state_ids)
                rows = [row for row in rows if row.state_id in evaluable_state_set]
                _log(
                    f"Retained {len(rows)} action rows after intersecting with {len(evaluable_state_set)} evaluable policy states "
                    f"from the evaluation per-state table (dropped_rows={before_rows - len(rows)})",
                    log_handle=log_handle,
                )
                if not rows:
                    raise OptimizationError(
                        "No action rows remained after restricting the MILP input to evaluable policy states."
                    )

            states = _build_states(rows)
            _log(f"Constructed {len(states)} decision states", log_handle=log_handle)

            config_path = (
                Path(args.config_json).resolve()
                if str(args.config_json).strip()
                else out_dir / "milp_policy_config_template.json"
            )
            config = _load_or_create_config(config_path, states, out_dir)

            if str(args.budget_total).strip() != "":
                config = dict(config)
                config["budget_total"] = _safe_float(args.budget_total, float("nan"))

            direction = _infer_direction(config, eval_summary if isinstance(eval_summary, Mapping) else {})
            pair_metrics, state_info = _compute_pair_metrics(
                states,
                config,
                direction=direction,
                score_transform=str(args.score_transform).strip(),
            )
            _log(
                f"Computed pair metrics for {len(pair_metrics)} feasible state-action pairs "
                f"(selection_direction={direction}, score_transform={args.score_transform})",
                log_handle=log_handle,
            )

            cplex_executable = _normalize_solver_executable(str(args.cplex_executable))
            if not cplex_executable and str(args.solver).strip().lower() in {"auto", "cplex"}:
                cplex_executable = _default_mac_cplex_executable()

            if cplex_executable:
                _log(f"Using CPLEX executable: {cplex_executable}", log_handle=log_handle)

            plan_rows, summary = _solve_milp(
                states=states,
                pair_metrics=pair_metrics,
                config=config,
                solver_name=str(args.solver).strip() or "auto",
                time_limit_sec=int(args.time_limit_sec) if args.time_limit_sec is not None else None,
                mip_gap=float(args.mip_gap) if args.mip_gap is not None else None,
                cplex_executable=cplex_executable,
                solver_log_path=solver_log_path,
                log_handle=log_handle,
            )

            _write_plan_csv(out_dir / "milp_policy_plan.csv", plan_rows)
            _write_json(out_dir / "milp_policy_plan.json", {"plan": plan_rows})

            summary_payload: Dict[str, Any] = {
                **summary,
                "eval_dir": str(eval_dir),
                "eval_task": eval_task,
                "eval_oracle_metric": eval_oracle_metric,
                "action_score_csv": str(action_csv),
                "action_score_source": action_score_source,
                "evaluable_state_ids_from_eval": list(evaluable_state_ids),
                "config_json": str(config_path),
                "split": str(args.split),
                "seed": str(args.seed),
                "score_transform": str(args.score_transform),
                "selection_direction": direction,
                "cplex_executable": cplex_executable,
                "budget_total": config.get("budget_total", None),
                "state_ids": [s.state_id for s in states],
                "state_info": state_info,
                "log_file": str(log_path),
                "solver_log_file": str(solver_log_path),
            }
            _write_json(out_dir / "milp_policy_summary.json", summary_payload)

            with (out_dir / "milp_policy_summary.txt").open("w", encoding="utf-8") as f:
                for key in sorted(summary_payload.keys()):
                    f.write(f"{key}: {summary_payload[key]}\n")

            _plot_timeline(plan_rows, out_dir / "milp_selected_actions_timeline.png")
            _plot_cumulative_gain(plan_rows, out_dir / "milp_cumulative_predicted_gain.png")

            final_msg = (
                f"MILP_POLICY_OK states={summary['n_states']} objective={summary['objective_value']:.6f} "
                f"total_cost={summary['total_cost']:.6f} solver={summary['solver_name']}"
            )
            _log(final_msg, log_handle=log_handle)
            print(final_msg, flush=True)

            log_handle.write("#" * 100 + "\n")
            log_handle.write(f"RUN END: {datetime.now().isoformat()} | STATUS=OK\n")
            log_handle.write("#" * 100 + "\n")
            log_handle.flush()
            return 0

        except Exception as exc:
            _log(f"ERROR: {exc}", log_handle=log_handle)
            log_handle.write("#" * 100 + "\n")
            log_handle.write(f"RUN END: {datetime.now().isoformat()} | STATUS=FAILED\n")
            log_handle.write("#" * 100 + "\n")
            log_handle.flush()
            raise


if __name__ == "__main__":
    raise SystemExit(main())
