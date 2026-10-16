#!/usr/bin/env python3
"""repeated-future oracle stability analysis.

This is a standalone add-on for the causal AMR repository. It does not train or
modify any predictive or policy model. It reruns only the original causal
simulator under the original fixed-parameter test cohort and quantifies how
stable the simulator oracle is across repeated matched stochastic futures.

Expected repository inputs (in --repo, default: current directory):
  - generate_amr_data.py
  - candidate_interventions.json

Default scientific design reproduced from the causal experiment:
  - 10 held-out trajectories, seeds 6000..6009
  - decision days 7, 14, ..., 49 (70 states total)
  - 7-day outcome horizon
  - baseline + four candidate screening/isolation actions
  - 32 matched reference futures per state/action
  - 64 independent matched evaluation futures per state/action
  - outcome: resistant transmission + resistant importation events

Outputs are written under --output (default: reviewer2_comment3_results) and
are checkpointed per state and stochastic replicate, so interrupted runs resume.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import t as student_t


SCRIPT_VERSION = "1.0.0"
DEFAULT_OUTPUT = "causal_repeated_features_results"
EXACT_TIE_ATOL = 1e-12


DEFAULT_SIM: Dict[str, Any] = {
    "num_patients": 50,
    "num_staff": 100,
    "num_wards": 5,
    "staff_wards_per_staff": 2,
    "isolation_mult": 0.8,
    "isolation_days": 3,
    "p_import_cs": 0.3,
    "p_import_cr": 0.1,
    "p_import_is": 0.3,
    "p_import_ir": 0.1,
    "p_admit_import_cs": 0.3,
    "p_admit_import_cr": 0.4,
    "p_admit_import_is": 0.3,
    "p_admit_import_ir": 0.4,
    "daily_discharge_frac": 0.3,
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
}


@dataclass(frozen=True)
class RunConfig:
    global_seed: int = 642701
    simulation_days: int = 60
    horizon: int = 7
    decision_days: Tuple[int, ...] = (7, 14, 21, 28, 35, 42, 49)
    test_seed_start: int = 6000
    test_trajectories: int = 10
    reference_rollouts: int = 32
    evaluation_rollouts: int = 64

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_seed": self.global_seed,
            "simulation_days": self.simulation_days,
            "horizon": self.horizon,
            "decision_days": list(self.decision_days),
            "test_seed_start": self.test_seed_start,
            "test_trajectories": self.test_trajectories,
            "reference_rollouts": self.reference_rollouts,
            "evaluation_rollouts": self.evaluation_rollouts,
            "sim": copy.deepcopy(DEFAULT_SIM),
        }


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % 2147483646 + 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_simulator(repo: Path):
    path = repo / "generate_amr_data.py"
    if not path.is_file():
        raise FileNotFoundError(f"Missing causal simulator: {path}")
    module_name = "_comment3_causal_sim_" + hashlib.sha256(
        str(path.resolve()).encode("utf-8")
    ).hexdigest()[:12]
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_simulator_api(module: Any) -> None:
    required = [
        "Params",
        "CausalInterventionSpec",
        "build_population",
        "init_states",
        "sample_contacts",
        "parse_amr_state",
        "_attach_operational_graph_metadata",
        "_discharge_and_admit_patients",
        "_apply_runtime_staff_intervention",
        "_apply_staff_presence_from_removed",
        "reset_daily_observation_flags",
        "apply_isolation_decay",
        "apply_pending_tests",
        "run_admission_screening",
        "run_screening",
        "run_day_transmission",
        "update_days_since_last_test",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "The causal simulator API does not match the version used for the "
            "Comment 3 analysis. Missing: " + ", ".join(missing)
        )


class CausalHospital:
    """Checkpointable wrapper around the repository's original causal simulator."""

    def __init__(
        self,
        module: Any,
        sim: Mapping[str, Any],
        seed: int,
        total_days: int,
    ) -> None:
        self.module = module
        self.sim = copy.deepcopy(dict(sim))
        self.seed = int(seed)
        self.day = 0
        self.params = module.Params()
        for key, value in self.sim.items():
            if hasattr(self.params, key):
                setattr(self.params, key, value)
        self.params.runtime_seed_base = self.seed
        self.params.runtime_region_index = 0
        self.params.causal_mode = 1
        self.params.causal_shared_noise_seed = stable_seed(self.seed, "history")
        self.intervention = None

        if self.params.admit_import_seasonality == "shock":
            rng = np.random.RandomState(self.seed + 777777)
            low = min(
                int(total_days),
                max(1, int(self.params.admit_import_shock_min_days)),
            )
            high = min(
                int(total_days),
                max(low, int(self.params.admit_import_shock_max_days)),
            )
            duration = int(rng.randint(low, high + 1))
            start = int(rng.randint(1, max(1, total_days - duration + 1) + 1))
            lo, hi = sorted(
                [
                    float(self.params.admit_import_shock_mult_min),
                    float(self.params.admit_import_shock_mult_max),
                ]
            )
            self.params.admit_import_shock_start_day = start
            self.params.admit_import_shock_duration_days = duration
            self.params.admit_import_shock_mult = (
                float(rng.uniform(lo, hi)) if hi > lo else lo
            )

        (
            self.patients,
            self.staff,
            self.ward_of,
            self.staff_wards,
        ) = module.build_population(
            num_patients=int(self.sim["num_patients"]),
            num_staff=int(self.sim["num_staff"]),
            num_wards=int(self.sim["num_wards"]),
            staff_wards_per_staff=int(self.sim.get("staff_wards_per_staff", 2)),
            seed=self.seed,
        )
        self.staff_removed = {staff: False for staff in self.staff}
        self.staff_timer = {staff: 0 for staff in self.staff}
        self.next_patient_id = int(self.sim["num_patients"])
        self.turnover_rng = np.random.RandomState(self.seed + 424242)
        self.graph = self._contacts(self.staff_wards, self.staff_removed)
        module.init_states(
            self.graph,
            self.patients,
            self.staff,
            self.ward_of,
            self.staff_wards,
            self.params,
        )
        self._force_superspreader()
        self._metadata({})

    def _active_superspreader(self) -> bool:
        staff = str(self.sim.get("superspreader_staff", ""))
        return bool(
            staff
            and int(self.sim.get("superspreader_start_day", 1))
            <= self.day
            <= int(self.sim.get("superspreader_end_day", 9999))
        )

    def _force_superspreader(self) -> None:
        if self._active_superspreader():
            staff = str(self.sim["superspreader_staff"])
            code = self.module.parse_amr_state(self.sim.get("superspreader_state"))
            if code is not None:
                self.graph.nodes[staff]["amr_state"] = int(code)

    def _contacts(self, effective_wards: Any, effective_removed: Any):
        return self.module.sample_contacts(
            patients=self.patients,
            staff=self.staff,
            ward_of=self.ward_of,
            staff_wards=effective_wards,
            num_wards=int(self.sim["num_wards"]),
            staff_removed=effective_removed,
            superspreader_staff=str(self.sim.get("superspreader_staff", "")) or None,
            superspreader_active=self._active_superspreader(),
            superspreader_patient_frac_mult=float(
                self.sim.get("superspreader_patient_frac_mult", 3.0)
            ),
            superspreader_patient_min_add=int(
                self.sim.get("superspreader_patient_min_add", 10)
            ),
            superspreader_staff_contacts=int(
                self.sim.get("superspreader_staff_contacts", 30)
            ),
            superspreader_edge_weight_mult=float(
                self.sim.get("superspreader_edge_weight_mult", 1.5)
            ),
            intervention=self.intervention,
            day=self.day,
            params=self.params,
        )

    def _metadata(self, admitted: Mapping[str, Any]) -> None:
        self.graph.graph.update(
            day=self.day,
            region=0,
            admitted_by_ward_json=json.dumps(admitted, sort_keys=True),
        )
        self.module._attach_operational_graph_metadata(
            self.graph,
            self.params,
            self.intervention,
            day=self.day,
        )

    def step(self):
        mod = self.module
        self.day += 1
        (
            self.patients,
            self.ward_of,
            self.next_patient_id,
            admitted,
        ) = mod._discharge_and_admit_patients(
            G_prev=self.graph,
            patients=self.patients,
            ward_of=self.ward_of,
            num_wards=int(self.sim["num_wards"]),
            day=self.day,
            params=self.params,
            next_patient_id=self.next_patient_id,
            rng=self.turnover_rng,
            intervention=self.intervention,
        )
        self._force_superspreader()
        effective_wards, effective_removed = mod._apply_runtime_staff_intervention(
            staff=self.staff,
            ward_of=self.ward_of,
            staff_wards=self.staff_wards,
            staff_removed=self.staff_removed,
            intervention=self.intervention,
            day=self.day,
        )
        new_graph = self._contacts(effective_wards, effective_removed)
        for node in new_graph.nodes:
            if node not in self.graph:
                raise RuntimeError(
                    "Unexpected new node without simulator admission initialisation"
                )
            new_graph.nodes[node].update(self.graph.nodes[node])
        self.graph = new_graph
        mod._apply_staff_presence_from_removed(self.graph, effective_removed)
        mod.reset_daily_observation_flags(self.graph, self.params)
        mod.apply_isolation_decay(self.graph)
        mod.apply_pending_tests(
            self.graph,
            self.params,
            day=self.day,
            intervention=self.intervention,
        )
        mod.run_admission_screening(
            self.graph,
            self.params,
            self.day,
            intervention=self.intervention,
        )
        mod.run_screening(
            self.graph,
            self.params,
            self.day,
            intervention=self.intervention,
        )
        mod.run_day_transmission(
            self.graph,
            self.params,
            self.staff_removed,
            self.staff_timer,
            intervention=self.intervention,
            day=self.day,
        )
        mod._apply_staff_presence_from_removed(self.graph, effective_removed)
        mod.update_days_since_last_test(self.graph)
        self._metadata(admitted)
        return self.graph

    def snapshot(self) -> Dict[str, Any]:
        result = {
            key: copy.deepcopy(value)
            for key, value in self.__dict__.items()
            if key not in {"module", "params", "intervention"}
        }
        result["params"] = dict(vars(self.params))
        result["intervention"] = None
        return result

    @classmethod
    def restore(cls, module: Any, snapshot: Mapping[str, Any]) -> "CausalHospital":
        obj = object.__new__(cls)
        obj.module = module
        obj.__dict__.update(copy.deepcopy(dict(snapshot)))
        obj.params = module.Params()
        obj.params.__dict__.update(snapshot["params"])
        obj.intervention = None
        return obj

    def fork(
        self,
        seed: int,
        action: Mapping[str, Any] | None = None,
        end_day: int | None = None,
    ) -> "CausalHospital":
        obj = self.restore(self.module, self.snapshot())
        obj.params.causal_shared_noise_seed = int(seed)
        if action and not bool(action["is_baseline"]):
            obj.intervention = self.module.CausalInterventionSpec.from_dict(
                {
                    "name": action["name"],
                    "target_type": action.get("target_type", "global"),
                    "target_id": action.get("target_id", "hospital"),
                    "params": action["params"],
                    "start_day": self.day + 1,
                    "end_day": end_day,
                }
            )
        return obj


def graph_fingerprint(graph: Any) -> str:
    payload = {
        "nodes": sorted((str(node), dict(data)) for node, data in graph.nodes(data=True)),
        "edges": sorted(
            (str(u), str(v), dict(data)) for u, v, data in graph.edges(data=True)
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def outcome_components(graph: Any) -> Tuple[int, int]:
    """Return only the two components required by the original policy endpoint."""
    attrs = graph.graph
    transmission = int(attrs.get("new_trans_cr_total", 0))
    imported = int(attrs.get("new_import_cr_total", 0)) + int(
        attrs.get("new_import_ir_total", 0)
    )
    return transmission, imported


def load_actions(repo: Path) -> List[Dict[str, Any]]:
    path = repo / "candidate_interventions.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing intervention catalogue: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or len(raw) != 4:
        raise RuntimeError(
            "Comment 3 analysis expects the original four non-baseline causal "
            f"actions; found {len(raw) if isinstance(raw, list) else 'non-list'} in {path}."
        )

    actions: List[Dict[str, Any]] = [
        {
            "action_id": "baseline",
            "action_label": "baseline",
            "action_index": 0,
            "is_baseline": True,
            "name": "baseline",
            "display_name": "Baseline",
            "target_type": "",
            "target_id": "",
            "params": {},
        }
    ]
    seen = {"baseline"}
    import re

    for index, item in enumerate(raw, 1):
        suffix = str(item["name"]).strip().lower()
        if item.get("target_type"):
            suffix += "__" + str(item["target_type"]).strip().lower()
        if item.get("target_id"):
            suffix += "__" + re.sub(
                r"[^A-Za-z0-9._-]+", "-", str(item["target_id"]).strip()
            )
        base_id = re.sub(r"[^A-Za-z0-9._-]+", "-", suffix).strip("-") or (
            f"action_{index:02d}"
        )
        action_id = base_id
        bump = 2
        while action_id in seen:
            action_id = f"{base_id}_{bump}"
            bump += 1
        seen.add(action_id)
        actions.append(
            {
                "action_id": action_id,
                "action_label": f"A{index}",
                "action_index": index,
                "is_baseline": False,
                "name": item["name"],
                "display_name": item.get("display_name", item["name"]),
                "target_type": item.get("target_type", "global"),
                "target_id": item.get("target_id", "hospital"),
                "params": copy.deepcopy(item.get("params", {})),
            }
        )
    return actions


def build_test_states(
    repo: Path,
    config: RunConfig,
) -> List[Dict[str, Any]]:
    module = load_simulator(repo)
    validate_simulator_api(module)
    states: List[Dict[str, Any]] = []
    maximum_day = max(config.decision_days)

    for trajectory_index in range(config.test_trajectories):
        seed = config.test_seed_start + trajectory_index
        trajectory_id = f"test_{trajectory_index:03d}"
        engine = CausalHospital(
            module,
            DEFAULT_SIM,
            seed=seed,
            total_days=config.simulation_days,
        )
        for day in range(1, maximum_day + 1):
            engine.step()
            if day in config.decision_days:
                states.append(
                    {
                        "state_id": f"{trajectory_id}__day{day}",
                        "trajectory_id": trajectory_id,
                        "trajectory_seed": seed,
                        "decision_day": day,
                        "snapshot": engine.snapshot(),
                        "fingerprint": graph_fingerprint(engine.graph),
                    }
                )
    expected = config.test_trajectories * len(config.decision_days)
    if len(states) != expected:
        raise AssertionError(f"Expected {expected} states but generated {len(states)}")
    return states


def final_stream_seed(
    global_seed: int,
    state_id: str,
    group: str,
    replicate: int,
) -> int:
    """Mirror the two-stage seed derivation used in the revision-package run."""
    base_seed = stable_seed(global_seed, state_id, group, replicate)
    return stable_seed(base_seed, group, 0)


def run_single_replicate(
    repo: Path,
    snapshot: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    horizon: int,
    stream_seed: int,
) -> np.ndarray:
    module = load_simulator(repo)
    validate_simulator_api(module)
    engine = CausalHospital.restore(module, snapshot)
    reference_fingerprint = graph_fingerprint(engine.graph)
    result = np.zeros((len(actions), horizon, 2), dtype=np.int32)

    for action_index, action in enumerate(actions):
        arm = engine.fork(stream_seed, action)
        if graph_fingerprint(arm.graph) != reference_fingerprint:
            raise AssertionError("A policy rollout changed the pre-decision state")
        for day_index in range(horizon):
            result[action_index, day_index] = outcome_components(arm.step())
    return result


def load_or_run_bank(
    repo: Path,
    output: Path,
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    config: RunConfig,
    group: str,
    count: int,
) -> np.ndarray:
    group_dir = output / "checkpoints" / str(state["state_id"]) / group
    group_dir.mkdir(parents=True, exist_ok=True)
    pieces: List[np.ndarray] = []

    for replicate in range(count):
        path = group_dir / f"{replicate:04d}.npz"
        expected_seed = final_stream_seed(
            config.global_seed,
            str(state["state_id"]),
            group,
            replicate,
        )
        if path.is_file():
            with np.load(path) as loaded:
                components = loaded["components"]
                saved_seed = int(loaded["stream_seed"])
            if saved_seed != expected_seed:
                raise RuntimeError(f"Seed mismatch in checkpoint {path}")
            if components.shape != (len(actions), config.horizon, 2):
                raise RuntimeError(f"Unexpected checkpoint shape in {path}: {components.shape}")
        else:
            components = run_single_replicate(
                repo,
                state["snapshot"],
                actions,
                config.horizon,
                expected_seed,
            )
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary,
                components=components,
                stream_seed=np.asarray(expected_seed, dtype=np.int64),
            )
            os.replace(temporary, path)
        pieces.append(components)

    return np.stack(pieces, axis=0)


def bank_gains(components: np.ndarray) -> np.ndarray:
    """Return baseline-minus-action gain for transmission+importation burden."""
    # Dimensions: replicate, action, day, component.
    burden = components.sum(axis=2).sum(axis=2).astype(np.float64)
    return burden[:, :1] - burden


def mean_se(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    se = float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, se


def analyse_state(
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    reference_gains: np.ndarray,
    evaluation_gains: np.ndarray,
    config: RunConfig,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    reference_mean = reference_gains.mean(axis=0)
    evaluation_mean = evaluation_gains.mean(axis=0)

    order = sorted(
        range(len(actions)),
        key=lambda index: (-float(reference_mean[index]), index),
    )
    best_index, runner_index = order[0], order[1]
    differences = evaluation_gains[:, best_index] - evaluation_gains[:, runner_index]
    gap_mean, gap_se = mean_se(differences)
    critical = float(student_t.ppf(0.975, len(differences) - 1))

    evaluation_best = int(np.argmax(evaluation_mean))
    evaluation_ties = [
        index
        for index in range(len(actions))
        if abs(float(evaluation_mean[index] - evaluation_mean[evaluation_best]))
        <= EXACT_TIE_ATOL
    ]
    reference_ties = [
        index
        for index in range(len(actions))
        if abs(float(reference_mean[index] - reference_mean[best_index]))
        <= EXACT_TIE_ATOL
    ]

    gap_row: Dict[str, Any] = {
        "state_id": state["state_id"],
        "trajectory_id": state["trajectory_id"],
        "trajectory_seed": state["trajectory_seed"],
        "decision_day": state["decision_day"],
        "horizon": config.horizon,
        "outcome": "trans_import_res",
        "reference_best_action_id": actions[best_index]["action_id"],
        "reference_best_action_label": actions[best_index]["action_label"],
        "reference_runner_up_action_id": actions[runner_index]["action_id"],
        "reference_runner_up_action_label": actions[runner_index]["action_label"],
        "reference_best_gain_mean": float(reference_mean[best_index]),
        "reference_runner_up_gain_mean": float(reference_mean[runner_index]),
        "independent_gap_mean": gap_mean,
        "independent_gap_se": gap_se,
        "nominal_95_ci_low": gap_mean - critical * gap_se,
        "nominal_95_ci_high": gap_mean + critical * gap_se,
        "ci_includes_zero": bool(
            gap_mean - critical * gap_se <= 0.0 <= gap_mean + critical * gap_se
        ),
        "reference_exact_tie_count": len(reference_ties),
        "evaluation_exact_tie_count": len(evaluation_ties),
        "reference_exact_tied_labels": ";".join(
            str(actions[index]["action_label"]) for index in reference_ties
        ),
        "evaluation_exact_tied_labels": ";".join(
            str(actions[index]["action_label"]) for index in evaluation_ties
        ),
        "n_reference_rollouts": config.reference_rollouts,
        "n_evaluation_rollouts": config.evaluation_rollouts,
        "ci_interpretation": (
            "Paired Monte Carlo t interval conditional on the fixed latent simulator "
            "state and the reference-selected action pair; no across-state multiplicity correction."
        ),
    }

    action_rows: List[Dict[str, Any]] = []
    for action_index, action in enumerate(actions):
        ref_mean, ref_se = mean_se(reference_gains[:, action_index])
        eval_mean, eval_se = mean_se(evaluation_gains[:, action_index])
        action_rows.append(
            {
                "state_id": state["state_id"],
                "trajectory_id": state["trajectory_id"],
                "decision_day": state["decision_day"],
                "action_id": action["action_id"],
                "action_label": action["action_label"],
                "action_index": action_index,
                "reference_gain_mean": ref_mean,
                "reference_gain_se": ref_se,
                "evaluation_gain_mean": eval_mean,
                "evaluation_gain_se": eval_se,
            }
        )

    full_best = int(np.argmax(reference_mean))
    stability_rows: List[Dict[str, Any]] = []
    counts = sorted(
        {
            count
            for count in (1, 2, 4, 8, 16, 32, config.reference_rollouts)
            if count <= config.reference_rollouts
        }
    )
    for count in counts:
        best_at_count = int(np.argmax(reference_gains[:count].mean(axis=0)))
        stability_rows.append(
            {
                "state_id": state["state_id"],
                "trajectory_id": state["trajectory_id"],
                "decision_day": state["decision_day"],
                "n_rollouts": count,
                "best_action_id": actions[best_at_count]["action_id"],
                "best_action_label": actions[best_at_count]["action_label"],
                "full_bank_best_action_id": actions[full_best]["action_id"],
                "full_bank_best_action_label": actions[full_best]["action_label"],
                "matches_full_bank": int(best_at_count == full_best),
            }
        )

    return gap_row, action_rows, stability_rows


def analyse_state_worker(
    repo_text: str,
    output_text: str,
    state: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
    config_dict: Mapping[str, Any],
) -> Dict[str, Any]:
    repo = Path(repo_text)
    output = Path(output_text)
    config = RunConfig(
        global_seed=int(config_dict["global_seed"]),
        simulation_days=int(config_dict["simulation_days"]),
        horizon=int(config_dict["horizon"]),
        decision_days=tuple(int(value) for value in config_dict["decision_days"]),
        test_seed_start=int(config_dict["test_seed_start"]),
        test_trajectories=int(config_dict["test_trajectories"]),
        reference_rollouts=int(config_dict["reference_rollouts"]),
        evaluation_rollouts=int(config_dict["evaluation_rollouts"]),
    )

    reference_components = load_or_run_bank(
        repo,
        output,
        state,
        actions,
        config,
        "train_reference",
        config.reference_rollouts,
    )
    evaluation_components = load_or_run_bank(
        repo,
        output,
        state,
        actions,
        config,
        "independent_evaluation",
        config.evaluation_rollouts,
    )
    reference_gains = bank_gains(reference_components)
    evaluation_gains = bank_gains(evaluation_components)
    gap_row, action_rows, stability_rows = analyse_state(
        state,
        actions,
        reference_gains,
        evaluation_gains,
        config,
    )
    return {
        "gap": gap_row,
        "actions": action_rows,
        "stability": stability_rows,
    }


def aggregate_results(
    output: Path,
    results: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
    config: RunConfig,
) -> None:
    gap_rows = sorted(
        (dict(result["gap"]) for result in results),
        key=lambda row: (str(row["trajectory_id"]), int(row["decision_day"])),
    )
    action_rows = sorted(
        (dict(row) for result in results for row in result["actions"]),
        key=lambda row: (
            str(row["trajectory_id"]),
            int(row["decision_day"]),
            int(row["action_index"]),
        ),
    )
    stability_rows = sorted(
        (dict(row) for result in results for row in result["stability"]),
        key=lambda row: (
            str(row["trajectory_id"]),
            int(row["decision_day"]),
            int(row["n_rollouts"]),
        ),
    )

    write_csv(output / "expected_benefit_gaps.csv", gap_rows)
    write_csv(output / "state_action_mean_gains.csv", action_rows)
    write_csv(output / "monte_carlo_stability.csv", stability_rows)

    action_summary: List[Dict[str, Any]] = []
    for action in actions:
        rows = [row for row in action_rows if row["action_id"] == action["action_id"]]
        values = np.asarray([float(row["evaluation_gain_mean"]) for row in rows])
        action_summary.append(
            {
                "action_id": action["action_id"],
                "action_label": action["action_label"],
                "display_name": action["display_name"],
                "n_states": len(rows),
                "mean_evaluation_gain": float(values.mean()),
                "sd_state_mean_gain": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        )
    write_csv(output / "action_mean_gains.csv", action_summary)

    stability_summary: List[Dict[str, Any]] = []
    for count in sorted({int(row["n_rollouts"]) for row in stability_rows}):
        rows = [row for row in stability_rows if int(row["n_rollouts"]) == count]
        stability_summary.append(
            {
                "n_rollouts": count,
                "n_states": len(rows),
                "matches_full_bank": sum(int(row["matches_full_bank"]) for row in rows),
                "match_rate": float(np.mean([int(row["matches_full_bank"]) for row in rows])),
            }
        )
    write_csv(output / "monte_carlo_stability_summary.csv", stability_summary)

    exact_ties = [row for row in gap_rows if int(row["evaluation_exact_tie_count"]) > 1]
    unresolved = [row for row in gap_rows if bool(row["ci_includes_zero"])]

    pair_counts: Dict[str, int] = {}
    for row in gap_rows:
        pair = sorted(
            [
                str(row["reference_best_action_label"]),
                str(row["reference_runner_up_action_label"]),
            ]
        )
        key = " vs ".join(pair)
        pair_counts[key] = pair_counts.get(key, 0) + 1

    summary = {
        "script_version": SCRIPT_VERSION,
        "analysis": "Reviewer 2 Comment 3 repeated-future oracle stability",
        "scope": "Original fixed-parameter causal test cohort only; no model training or predictive analysis.",
        "n_test_trajectories": config.test_trajectories,
        "n_decision_states": len(gap_rows),
        "reference_rollouts_per_state_action": config.reference_rollouts,
        "independent_evaluation_rollouts_per_state_action": config.evaluation_rollouts,
        "outcome": "baseline minus action change in 7-day transmission-plus-importation resistant-event burden",
        "evaluation_exact_tie_states": len(exact_ties),
        "evaluation_exact_tie_rate": len(exact_ties) / len(gap_rows),
        "best_vs_runner_ci_includes_zero_states": len(unresolved),
        "best_vs_runner_ci_includes_zero_rate": len(unresolved) / len(gap_rows),
        "reference_best_runner_pair_counts": pair_counts,
        "action_mean_evaluation_gains": {
            str(row["action_label"]): float(row["mean_evaluation_gain"])
            for row in action_summary
        },
        "interpretation": (
            "Repeated matched futures estimate action-specific mean benefit conditional "
            "on the fixed latent simulator state. They do not establish the action with "
            "highest expected benefit conditional only on routinely observed information."
        ),
    }
    atomic_json(output / "comment3_summary.json", summary)

    lines = [
        "Reviewer 2 Comment 3 repeated-future summary",
        "",
        f"Test trajectories: {summary['n_test_trajectories']}",
        f"Decision states: {summary['n_decision_states']}",
        f"Reference futures per state/action: {summary['reference_rollouts_per_state_action']}",
        f"Independent evaluation futures per state/action: {summary['independent_evaluation_rollouts_per_state_action']}",
        (
            "Exact ties between largest independent evaluation means: "
            f"{summary['evaluation_exact_tie_states']}/{summary['n_decision_states']} "
            f"({100.0 * summary['evaluation_exact_tie_rate']:.1f}%)"
        ),
        (
            "Best-vs-runner paired 95% interval includes zero: "
            f"{summary['best_vs_runner_ci_includes_zero_states']}/{summary['n_decision_states']} "
            f"({100.0 * summary['best_vs_runner_ci_includes_zero_rate']:.1f}%)"
        ),
        "",
        "Mean independent-evaluation gain by action (events prevented per 7-day state):",
    ]
    for row in action_summary:
        lines.append(
            f"  {row['action_label']} ({row['display_name']}): "
            f"{float(row['mean_evaluation_gain']):.6f}"
        )
    lines.extend(["", "Reference best/runner-up pair counts:"])
    for pair, count in sorted(pair_counts.items()):
        lines.append(f"  {pair}: {count}")
    lines.extend(["", summary["interpretation"], ""])
    (output / "comment3_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def prepare_run_identity(
    repo: Path,
    output: Path,
    config: RunConfig,
    actions: Sequence[Mapping[str, Any]],
) -> None:
    source_files = [repo / "generate_amr_data.py", repo / "candidate_interventions.json"]
    identity = {
        "script_version": SCRIPT_VERSION,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "sources": {str(path.name): sha256_file(path) for path in source_files},
        "config": config.to_dict(),
        "actions": list(actions),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    identity["fingerprint"] = fingerprint
    path = output / "run_identity.json"
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing output {output} was created with a different source/configuration. "
                "Use a new --output directory."
            )
    else:
        atomic_json(path, identity)
    atomic_json(output / "action_catalog.json", {"actions": list(actions)})
    atomic_json(output / "run_config.json", config.to_dict())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standalone repeated-future oracle stability analysis for Reviewer 2 Comment 3."
        )
    )
    parser.add_argument(
        "--repo",
        default=".",
        help="Causal repository containing generate_amr_data.py and candidate_interventions.json.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Separate output directory. Existing compatible checkpoints are resumed.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of decision states processed in parallel. Use 1 for minimum memory usage.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one trajectory/day with 2 reference and 3 evaluation futures for workflow testing only.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo = Path(args.repo).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    module = load_simulator(repo)
    validate_simulator_api(module)
    actions = load_actions(repo)

    if args.smoke:
        config = RunConfig(
            decision_days=(7,),
            test_trajectories=1,
            reference_rollouts=2,
            evaluation_rollouts=3,
        )
    else:
        config = RunConfig()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    prepare_run_identity(repo, output, config, actions)
    print(f"Repository: {repo}")
    print(f"Output: {output}")
    print(
        f"Design: {config.test_trajectories} test trajectories x "
        f"{len(config.decision_days)} decision days = "
        f"{config.test_trajectories * len(config.decision_days)} states"
    )
    print(
        f"Rollouts/state/action: {config.reference_rollouts} reference + "
        f"{config.evaluation_rollouts} independent evaluation"
    )

    states = build_test_states(repo, config)
    config_dict = config.to_dict()
    results: List[Mapping[str, Any]] = []

    if args.workers == 1:
        for position, state in enumerate(states, 1):
            print(f"[{position}/{len(states)}] {state['state_id']}", flush=True)
            results.append(
                analyse_state_worker(
                    str(repo),
                    str(output),
                    state,
                    actions,
                    config_dict,
                )
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_to_state = {
                executor.submit(
                    analyse_state_worker,
                    str(repo),
                    str(output),
                    state,
                    actions,
                    config_dict,
                ): state["state_id"]
                for state in states
            }
            completed = 0
            for future in as_completed(future_to_state):
                state_id = future_to_state[future]
                result = future.result()
                results.append(result)
                completed += 1
                print(f"[{completed}/{len(states)}] {state_id}", flush=True)

    aggregate_results(output, results, actions, config)
    print("\nCompleted. Key outputs:")
    for name in (
        "comment3_summary.txt",
        "comment3_summary.json",
        "expected_benefit_gaps.csv",
        "monte_carlo_stability.csv",
        "monte_carlo_stability_summary.csv",
        "state_action_mean_gains.csv",
        "action_mean_gains.csv",
    ):
        print(f"  {output / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
