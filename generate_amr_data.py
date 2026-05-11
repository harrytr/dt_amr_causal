#!/usr/bin/env python3
"""
generate_amr_data.py

Hospital contact-network AMR simulator that generates daily directed GraphML
snapshots for one or more regions.

Core model
- Patients are assigned to exactly one ward.
- Staff are assigned a home ward and may cover multiple wards.
- Contact graphs are re-sampled daily with directed edges, edge weights, and edge types.
- Node AMR states are encoded as:
    0: U  (uncolonised / susceptible)
    1: CS (colonised, susceptible)
    2: CR (colonised, resistant)
    3: IS (infected, susceptible)
    4: IR (infected, resistant)

Dynamics
- Transmission is driven by directed contacts and edge weights.
- Resistant sources can transmit with a configurable multiplier.
- Within-host progression includes colonisation-to-infection, clearance, and
  antibiotic-driven selection from susceptible to resistant states.
- Antibiotic exposure is simulated at node level.
- Staff can be temporarily removed from contacts under configurable infection-based policies.

Admissions, turnover, and importation
- Optional daily patient turnover via discharges and matched admissions.
- Admission importation into CS/CR can be configured separately from day-0 seeding.
- Admission importation can be modulated using:
    * none
    * sinusoid
    * piecewise high/low season
    * shock window

Screening and observability
- Routine screening supports weekly cadence or every-k-days cadence.
- Admission-day screening can be enabled for newly admitted patients.
- Test results can be immediate or delayed.
- Observation state can either reset daily or persist across days.
- Screening and delayed-result handling are supported for both patients and staff,
  while admission screening is patient-only.

Isolation
- Positive observed tests trigger isolation for a configurable duration.
- Isolation modifies effective transmission probability through a multiplicative factor.

Outputs
- Daily GraphML files with node attributes, edge attributes, and graph-level daily totals.
- Optional YAML export of run metadata for reproducibility.
- Optional high-resolution GIF summarising the simulation over time.
- Per-region state-evolution PNG summarising U, CS, CR, IS, and IR counts across days.

Graph-level daily totals written for downstream use include:
- new_cr_acq_total
- new_ir_inf_total
- new_import_cr_total
- new_import_cs_total
- new_trans_cr_total
- new_select_cr_total
- resistant_fraction
"""
import argparse
import io
import math
import hashlib
import os
import random
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
import json
import numpy as np
import networkx as nx

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

try:
    from PIL import Image
except Exception:
    Image = None


# =============================================================================
# AMR state encoding
# =============================================================================
AMR_STATE_STR = {0: "U", 1: "CS", 2: "CR", 3: "IS", 4: "IR"}
AMR_STATE_CODE = {v: k for k, v in AMR_STATE_STR.items()}


def parse_amr_state(value: Optional[str]) -> Optional[int]:
    """
    Parse AMR state from CLI/UI.

    Accepts:
      - None / empty -> None
      - "U","CS","CR","IS","IR" (case-insensitive)
      - "0".."4"

    Returns:
      int in [0,4] or None
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    s_up = s.upper()
    if s_up in AMR_STATE_CODE:
        return int(AMR_STATE_CODE[s_up])
    if s_up.isdigit():
        code = int(s_up)
        if code in AMR_STATE_STR:
            return code
    raise ValueError(
        f"Invalid AMR state '{value}'. Expected one of {sorted(AMR_STATE_CODE.keys())} or an integer in [0,4]."
    )


STATE_STYLE = {
    0: {"label": "U", "color": "#9E9E9E"},   # grey
    1: {"label": "CS", "color": "#0072B2"},  # blue
    2: {"label": "CR", "color": "#009E73"},  # green
    3: {"label": "IS", "color": "#F0E442"},  # yellow
    4: {"label": "IR", "color": "#D0021B"},  # red
}

ROLE_STYLE = {
    "patient": {"shape": "o", "size": 26, "lw": 0.20},
    "staff": {"shape": "s", "size": 52, "lw": 0.35},
}

GIF_FIGSIZE = (8.4, 6.3)  # inches
GIF_DPI = 220             # high DPI

EDGE_ALPHA = 0.08
EDGE_WIDTH = 0.35
EDGE_COLOR = "#111111"

ISO_RING_COLOR = "#000000"
ISO_RING_LW = 1.1


@dataclass
class Params:
    # Transmission intensities by contact type
    beta_pp: float = 0.015   # patient -> patient (within ward)
    beta_sp: float = 0.02    # staff  <-> patient
    beta_ss: float = 0.008   # staff  <-> staff
    beta_res_mult: float = 1.0  # eta_R: no extra resistant-type infectivity by default
    beta_state_mult_cs: float = 0.5  # source-state multiplier for CS
    beta_state_mult_cr: float = 0.5  # source-state multiplier for CR
    beta_state_mult_is: float = 1.0  # source-state multiplier for IS
    beta_state_mult_ir: float = 1.0  # source-state multiplier for IR
    crossward_hand_hygiene_mult: float = 0.55  # leaky hand hygiene for staff contacts outside home ward
    crossward_hand_hygiene_enabled: int = 1
    no_abx_sensitive_selection_weight: float = 4.0  # competing exposures: S favoured without antibiotics
    abx_resistant_selection_weight: float = 4.0     # competing exposures: R favoured under antibiotics

    # Natural history
    p_col_to_inf: float = 0.012  # CS/CR -> IS/IR per day
    p_inf_clear: float = 0.070   # IS/IR -> U per day
    p_col_clear: float = 0.005   # CS/CR -> U per day

    # Antibiotic selection (within-host)
    p_select_col: float = 0.03  # CS -> CR per day when on antibiotics
    p_select_inf: float = 0.02  # IS -> IR per day when on antibiotics

    # Antibiotic exposure (simple policy)
    p_start_abx_if_inf: float = 0.65
    p_start_abx_if_not_inf: float = 0.03
    p_stop_abx: float = 0.12
    n_abx_classes: int = 1

    # Importation / day-0 seeding. CS/CR are colonised states; IS/IR are
    # symptomatic/infected states. IS/IR defaults preserve previous behaviour.
    p_import_cs: float = 0.3
    p_import_cr: float = 0.1
    p_import_is: float = 0.3
    p_import_ir: float = 0.1

    # Importation on admission (dynamic census). Defaults match day-0 importation.
    # CS/CR are colonised states; IS/IR are symptomatic/infected states.
    # Defaults preserve previous behaviour: admitted imports are colonised unless
    # p_admit_import_is / p_admit_import_ir are explicitly set.
    p_admit_import_cs: float = 0.15
    p_admit_import_cr: float = 0.1
    p_admit_import_is: float = 0.15
    p_admit_import_ir: float = 0.1

    # Dynamic census (admissions/discharges). 0.0 disables turnover.
    daily_discharge_frac: float = 0.0
    daily_discharge_min_per_ward: int = 0

    # Screening (weekly)
    screen_sens: float = 0.90
    screen_spec: float = 0.99
    weekly_screen_day: int = 7

    # Alternative screening cadence
    screen_every_k_days: int = 0

    # Screening on admission for newly admitted patients
    screen_on_admission: int = 0

    # Test result delay in days (0 => immediate)
    screen_result_delay_days: int = 0

    # If 1, keep obs_status/observed_pos across days
    persist_observations: int = 0

    # Isolation effect multiplier (applied to transmission probability)
    isolation_mult: float = 0.35

    # Staff removal policy when infected (IS/IR)
    staff_removal_mode: int = 0
    staff_removal_prob: float = 0.5

    # Isolation duration in days after a positive screen
    isolation_days: int = 7

    passive_screen_capacity_per_week: int = 0  # 0 => screen all eligible on screening day
    active_symptom_screening: int = 1          # active screen when patients progress C -> I
    admission_screen_symptomatic_only: int = 1 # admission screening only for IS/IR patients
    isolation_trigger_policy: str = "resistant_only"  # resistant_only | all_positives

    # Seasonal importation on admission
    admit_import_seasonality: str = "none"
    admit_import_amp: float = 0.0
    admit_import_period_days: int = 365
    admit_import_phase_day: int = 0
    admit_import_pmax_cs: float = 1.0
    admit_import_pmax_cr: float = 1.0
    admit_import_pmax_is: float = 1.0
    admit_import_pmax_ir: float = 1.0

    # Piecewise-only
    admit_import_high_start_day: int = 1
    admit_import_high_end_day: int = 90
    admit_import_high_mult: float = 1.5
    admit_import_low_mult: float = 1.0

    # Shock-only
    admit_import_shock_min_days: int = 7
    admit_import_shock_max_days: int = 30
    admit_import_shock_mult_min: float = 1.5
    admit_import_shock_mult_max: float = 3.0

    # Filled in per region when mode == "shock"
    admit_import_shock_start_day: int = 0
    admit_import_shock_duration_days: int = 0
    admit_import_shock_mult: float = 1.0

    # Stage 1 causal metadata
    causal_mode: int = 0
    causal_pair_id: str = ""
    causal_role: str = "factual"
    causal_shared_noise_seed: int = 0
    causal_intervention_name: str = ""
    causal_intervention_target_type: str = ""
    causal_intervention_target_id: str = ""

    # Internal runtime RNG context (not user-facing configuration)
    runtime_seed_base: int = 0
    runtime_region_index: int = 0


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _stable_hash_int(*parts: Any) -> int:
    payload = "||".join(str(p) for p in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _use_common_random_numbers(params: Optional[Params]) -> bool:
    if params is None:
        return False
    try:
        return int(getattr(params, "causal_mode", 0)) == 1 and int(getattr(params, "causal_shared_noise_seed", 0)) != 0
    except Exception:
        return False


def _noise_base_seed(params: Optional[Params]) -> int:
    if params is None:
        return 0
    if _use_common_random_numbers(params):
        return int(getattr(params, "causal_shared_noise_seed", 0))
    try:
        return int(getattr(params, "runtime_seed_base", 0))
    except Exception:
        return 0


def _noise_region_salt(params: Optional[Params]) -> int:
    if params is None:
        return 0
    try:
        return int(getattr(params, "runtime_region_index", 0))
    except Exception:
        return 0


def _event_u01(
    params: Optional[Params],
    *parts: Any,
    fallback_rng: Optional[np.random.RandomState] = None,
) -> float:
    if _use_common_random_numbers(params):
        x = _stable_hash_int(_noise_base_seed(params), _noise_region_salt(params), *parts)
        return ((x >> 11) & ((1 << 53) - 1)) / float(1 << 53)
    if fallback_rng is not None:
        return float(fallback_rng.rand())
    return float(np.random.rand())


def _normalise_state_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Return non-negative state probabilities with total mass capped at one.

    This is used for both day-0 seeding and admission importation.  If the
    requested CS/CR/IS/IR probabilities sum to more than one, the relative
    composition is preserved and the total import/seeding probability is set
    to one, avoiding order-dependent dominance of the first checked state.
    """
    probs = {str(k): max(0.0, float(v)) for k, v in dict(weights).items()}
    total = float(sum(probs.values()))
    if total > 1.0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


def _draw_state_from_weights(
    u: float,
    weights: Mapping[str, float],
    state_order: Sequence[str] = ("CS", "CR", "IS", "IR"),
) -> Optional[str]:
    """Draw a state from capped probabilities; return None for residual U."""
    probs = _normalise_state_weights(weights)
    total = float(sum(probs.get(state, 0.0) for state in state_order))
    if total <= 0.0 or float(u) >= total:
        return None

    draw = float(u) / max(total, 1e-12)
    cumulative = 0.0
    for state in state_order:
        cumulative += probs.get(state, 0.0) / max(total, 1e-12)
        if draw <= cumulative:
            return state
    return state_order[-1]


def _event_uniform(
    params: Optional[Params],
    low: float,
    high: float,
    *parts: Any,
    fallback_rng: Optional[np.random.RandomState] = None,
) -> float:
    u = _event_u01(params, *parts, fallback_rng=fallback_rng)
    return float(low) + (float(high) - float(low)) * float(u)


def _event_randint(
    params: Optional[Params],
    low_inclusive: int,
    high_exclusive: int,
    *parts: Any,
    fallback_rng: Optional[np.random.RandomState] = None,
) -> int:
    lo = int(low_inclusive)
    hi = int(high_exclusive)
    if hi <= lo:
        return lo
    if _use_common_random_numbers(params):
        u = _event_u01(params, *parts)
        span = max(1, hi - lo)
        idx = min(span - 1, int(math.floor(u * span)))
        return lo + idx
    if fallback_rng is not None:
        return int(fallback_rng.randint(lo, hi))
    return int(np.random.randint(lo, hi))


def _stream_rng(params: Optional[Params], *parts: Any) -> np.random.RandomState:
    seed = _stable_hash_int(_noise_base_seed(params), _noise_region_salt(params), *parts) % (2**31 - 1)
    if seed <= 0:
        seed = 1
    return np.random.RandomState(int(seed))


def _stable_sample_without_replacement(
    candidates: List[Any],
    k: int,
    *,
    params: Optional[Params],
    fallback_rng: Optional[np.random.RandomState] = None,
    stream_parts: Tuple[Any, ...] = (),
) -> List[Any]:
    items = list(candidates)
    k = max(0, min(int(k), len(items)))
    if k <= 0:
        return []
    if not _use_common_random_numbers(params):
        rng = fallback_rng if fallback_rng is not None else np.random
        chosen = rng.choice(np.array(items, dtype=object), size=k, replace=False).tolist()
        return list(chosen)

    decorated = []
    base_parts = (_noise_base_seed(params), _noise_region_salt(params)) + tuple(stream_parts)
    for idx, item in enumerate(items):
        key = _stable_hash_int(*base_parts, str(item))
        decorated.append((key, idx, item))
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in decorated[:k]]


def _stable_shuffle(
    items: List[Any],
    *,
    params: Optional[Params],
    fallback_rng: Optional[np.random.RandomState] = None,
    stream_parts: Tuple[Any, ...] = (),
) -> List[Any]:
    out = list(items)
    if len(out) <= 1:
        return out
    if not _use_common_random_numbers(params):
        rng = fallback_rng if fallback_rng is not None else np.random
        rng.shuffle(out)
        return out

    decorated = []
    base_parts = (_noise_base_seed(params), _noise_region_salt(params)) + tuple(stream_parts)
    for idx, item in enumerate(out):
        key = _stable_hash_int(*base_parts, str(item))
        decorated.append((key, idx, item))
    decorated.sort(key=lambda t: (t[0], t[1]))
    return [item for _, _, item in decorated]


# =============================================================================
# Stage 1 causal helpers
# =============================================================================
_CAUSAL_SUPPORTED_INTERVENTIONS = {
    "reduce_ward_importation",
    "remove_staff_crossward_cover",
    "remove_specific_staff",
    "remove_edge",
    "set_screening_frequency",
    "set_screening_delay",
    "disable_isolation_response",
    "set_isolation_parameters",
}


@dataclass
class CausalInterventionSpec:
    name: str = ""
    target_type: str = ""
    target_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    start_day: Optional[int] = None
    end_day: Optional[int] = None
    description: str = ""

    @classmethod
    def from_dict(cls, payload: Optional[Dict[str, Any]]) -> "CausalInterventionSpec":
        if payload in (None, "", {}):
            return cls()
        if not isinstance(payload, dict):
            raise ValueError("Causal intervention must be a JSON object.")
        name = str(payload.get("name", "")).strip()
        if name and name not in _CAUSAL_SUPPORTED_INTERVENTIONS:
            raise ValueError(
                f"Unsupported causal intervention '{name}'. Supported: {sorted(_CAUSAL_SUPPORTED_INTERVENTIONS)}"
            )
        target_type = str(payload.get("target_type", "")).strip()
        target_id = str(payload.get("target_id", "")).strip()
        params = payload.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("Causal intervention params must be a JSON object.")
        raw_start_day = payload.get("start_day")
        raw_end_day = payload.get("end_day")
        start_day = None if raw_start_day in (None, "", "null") else int(raw_start_day)
        end_day = None if raw_end_day in (None, "", "null") else int(raw_end_day)
        if start_day is not None and end_day is not None and end_day < start_day:
            raise ValueError("Causal intervention end_day must be >= start_day.")
        description = str(payload.get("description", "")).strip()
        return cls(
            name=name,
            target_type=target_type,
            target_id=target_id,
            params=dict(params),
            start_day=start_day,
            end_day=end_day,
            description=description,
        )

    @property
    def is_active(self) -> bool:
        return self.name != ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass
class CounterfactualContext:
    pair_id: str = ""
    role: str = "factual"
    shared_noise_seed: int = 0
    intervention: CausalInterventionSpec = field(default_factory=CausalInterventionSpec)


def _load_causal_intervention_json(value: Optional[str]) -> CausalInterventionSpec:
    s = str(value or "").strip()
    if s == "":
        return CausalInterventionSpec()
    payload = json.loads(s)
    return CausalInterventionSpec.from_dict(payload)


def _causal_role_normalized(value: Optional[str]) -> str:
    role = str(value or "factual").strip().lower()
    if role not in {"factual", "counterfactual"}:
        raise ValueError("causal_role must be 'factual' or 'counterfactual'.")
    return role


def _causal_metadata_dict(ctx: Optional[CounterfactualContext]) -> Dict[str, Any]:
    if ctx is None:
        return {
            "cf_enabled": 0,
            "cf_pair_id": "",
            "cf_role": "",
            "cf_shared_noise_seed": 0,
            "cf_intervention_name": "",
            "cf_intervention_target_type": "",
            "cf_intervention_target_id": "",
            "cf_intervention_json": "",
        }
    return {
        "cf_enabled": 1,
        "cf_pair_id": str(ctx.pair_id),
        "cf_role": str(ctx.role),
        "cf_shared_noise_seed": int(ctx.shared_noise_seed),
        "cf_intervention_name": str(ctx.intervention.name),
        "cf_intervention_target_type": str(ctx.intervention.target_type),
        "cf_intervention_target_id": str(ctx.intervention.target_id),
        "cf_intervention_start_day": "" if ctx.intervention.start_day is None else int(ctx.intervention.start_day),
        "cf_intervention_end_day": "" if ctx.intervention.end_day is None else int(ctx.intervention.end_day),
        "cf_intervention_description": str(ctx.intervention.description),
        "cf_intervention_json": ctx.intervention.to_json() if ctx.intervention.is_active else "",
    }


def _apply_graph_causal_metadata(G: nx.DiGraph, ctx: Optional[CounterfactualContext]) -> None:
    for k, v in _causal_metadata_dict(ctx).items():
        G.graph[k] = v


def _intervention_active_on_day(intervention: Optional[CausalInterventionSpec], day: Optional[int]) -> bool:
    if intervention is None or not intervention.is_active:
        return False
    if day is None:
        return True

    d = int(day)
    start_day = intervention.start_day
    end_day = intervention.end_day

    if start_day is not None and d < int(start_day):
        return False
    if end_day is not None and d > int(end_day):
        return False
    return True


def _effective_screen_every_k_days(
    params: Params,
    intervention: Optional[CausalInterventionSpec],
    day: Optional[int] = None,
) -> int:
    k = int(getattr(params, "screen_every_k_days", 0))

    if _intervention_active_on_day(intervention, day) and intervention is not None and intervention.name == "set_screening_frequency":
        try:
            raw_k = intervention.params.get(
                "frequency_days",
                intervention.params.get(
                    "screen_every_k_days",
                    intervention.params.get("k_days", k),
                ),
            )
            k = int(raw_k)
        except Exception:
            pass

    return max(0, int(k))


def _effective_screen_result_delay_days(
    params: Params,
    intervention: Optional[CausalInterventionSpec],
    day: Optional[int] = None,
) -> int:
    d = int(getattr(params, "screen_result_delay_days", 0))
    if _intervention_active_on_day(intervention, day) and intervention is not None:
        try:
            if intervention.name == "set_screening_delay":
                d = int(intervention.params.get("delay_days", intervention.params.get("screen_result_delay_days", d)))
            elif intervention.name == "set_screening_frequency":
                if "delay_days" in intervention.params or "screen_result_delay_days" in intervention.params:
                    d = int(intervention.params.get("delay_days", intervention.params.get("screen_result_delay_days", d)))
        except Exception:
            pass
    return max(0, int(d))


def _screening_anchor_day(
    params: Params,
    intervention: Optional[CausalInterventionSpec],
    day: Optional[int] = None,
) -> int:
    """
    Anchor intervention-defined k-day screening to the intervention start day.

    This preserves the historical global day-1 phase for baseline screening, while
    ensuring that a delayed causal action with start_day=d begins its own cadence
    on d rather than inheriting an arbitrary global modulo phase.
    """
    if _intervention_active_on_day(intervention, day) and intervention is not None:
        if intervention.name == "set_screening_frequency" and intervention.start_day is not None:
            try:
                return int(intervention.start_day)
            except Exception:
                pass
    return 1


def _days_until_next_screen(
    day: int,
    params: Params,
    intervention: Optional[CausalInterventionSpec] = None,
) -> int:
    d = int(day)
    k = _effective_screen_every_k_days(params, intervention, day=d)
    if k > 0:
        anchor = _screening_anchor_day(params, intervention, day=d)
        return int((anchor - d) % int(k))

    weekly_screen_day = int(getattr(params, "weekly_screen_day", 7))
    weekly_screen_day = 1 if weekly_screen_day < 1 else (7 if weekly_screen_day > 7 else weekly_screen_day)
    today_slot = ((d - 1) % 7) + 1
    return int((weekly_screen_day - today_slot) % 7)


def _attach_operational_graph_metadata(
    G: nx.DiGraph,
    params: Params,
    intervention: Optional[CausalInterventionSpec],
    day: int,
) -> None:
    d = int(day)
    G.graph["current_screen_every_k_days"] = int(_effective_screen_every_k_days(params, intervention, day=d))
    G.graph["current_weekly_screen_day"] = int(getattr(params, "weekly_screen_day", 7))
    G.graph["current_screen_on_admission"] = int(_effective_screen_on_admission(params, intervention, day=d))
    G.graph["current_screen_result_delay_days"] = int(_effective_screen_result_delay_days(params, intervention, day=d))
    G.graph["current_isolation_mult"] = float(_effective_isolation_mult(params, intervention, day=d))
    G.graph["current_isolation_days"] = int(_effective_isolation_days(params, intervention, day=d))
    G.graph["current_isolation_trigger_policy"] = str(_effective_isolation_trigger_policy(params, intervention, day=d))
    G.graph["current_persist_observations"] = int(getattr(params, "persist_observations", 0))
    G.graph["current_is_screening_day"] = int(_is_screening_day_with_params(d, params, intervention=intervention))
    G.graph["current_days_until_next_screen"] = int(_days_until_next_screen(d, params, intervention=intervention))
    G.graph["current_causal_intervention_active"] = int(_intervention_active_on_day(intervention, d))


def _effective_screen_on_admission(
    params: Params,
    intervention: Optional[CausalInterventionSpec],
    day: Optional[int] = None,
) -> int:
    value = int(getattr(params, "screen_on_admission", 0))
    if _intervention_active_on_day(intervention, day) and intervention is not None and intervention.name == "set_screening_frequency":
        try:
            if "screen_on_admission" in intervention.params:
                value = int(intervention.params.get("screen_on_admission", value))
            elif "admission_screening" in intervention.params:
                value = int(intervention.params.get("admission_screening", value))
        except Exception:
            pass
    return 1 if int(value) == 1 else 0


def _effective_isolation_mult(
    params: Params,
    intervention: Optional[CausalInterventionSpec],
    day: Optional[int] = None,
) -> float:
    value = float(getattr(params, "isolation_mult", 0.35))
    if _intervention_active_on_day(intervention, day) and intervention is not None:
        try:
            if intervention.name in {"set_isolation_parameters", "set_screening_frequency"}:
                if "isolation_mult" in intervention.params:
                    value = float(intervention.params.get("isolation_mult", value))
                elif "transmission_multiplier" in intervention.params:
                    value = float(intervention.params.get("transmission_multiplier", value))
        except Exception:
            pass
    return max(0.0, float(value))


def _effective_isolation_days(
    params: Params,
    intervention: Optional[CausalInterventionSpec],
    day: Optional[int] = None,
) -> int:
    value = int(getattr(params, "isolation_days", 7))
    if _intervention_active_on_day(intervention, day) and intervention is not None:
        try:
            if intervention.name in {"set_isolation_parameters", "set_screening_frequency"}:
                value = int(intervention.params.get("isolation_days", value))
        except Exception:
            pass
    return max(0, int(value))


def _apply_runtime_staff_intervention(
    *,
    staff: List[str],
    ward_of: Dict[str, int],
    staff_wards: Dict[str, List[int]],
    staff_removed: Dict[str, bool],
    intervention: Optional[CausalInterventionSpec],
    day: Optional[int] = None,
) -> Tuple[Dict[str, List[int]], Dict[str, bool]]:
    eff_staff_wards = {str(k): list(v) for k, v in staff_wards.items()}
    eff_staff_removed = dict(staff_removed)
    if not _intervention_active_on_day(intervention, day):
        return eff_staff_wards, eff_staff_removed

    if intervention.name == "remove_staff_crossward_cover":
        sid = str(intervention.target_id).strip()
        if sid in eff_staff_wards:
            eff_staff_wards[sid] = [int(ward_of.get(sid, 0))]
    elif intervention.name == "remove_specific_staff":
        sid = str(intervention.target_id).strip()
        if sid in staff or sid in eff_staff_removed:
            eff_staff_removed[sid] = True
    return eff_staff_wards, eff_staff_removed


def _apply_contact_level_intervention(
    G: nx.DiGraph,
    intervention: Optional[CausalInterventionSpec],
    day: Optional[int] = None,
) -> None:
    if not _intervention_active_on_day(intervention, day):
        return
    if intervention.name != "remove_edge":
        return

    src = str(
        intervention.params.get(
            "source",
            intervention.params.get("src", ""),
        )
    ).strip()
    dst = str(
        intervention.params.get(
            "target",
            intervention.params.get("dst", ""),
        )
    ).strip()

    target = str(intervention.target_id).strip()
    if target and ":" in target and (src == "" or dst == ""):
        src, dst = [x.strip() for x in target.split(":", 1)]

    if src != "" and dst != "" and G.has_edge(src, dst):
        G.remove_edge(src, dst)


# =============================================================================
# State evolution plot
# =============================================================================
def _infer_split_label(output_dir: str) -> str:
    base = os.path.basename(os.path.normpath(str(output_dir))).lower()
    if "test" in base:
        return "test"
    if "learn" in base or "train" in base:
        return "train"
    return "run"


def save_state_evolution_png(
    *,
    days: List[int],
    u: List[int],
    cs: List[int],
    cr: List[int],
    is_: List[int],
    ir: List[int],
    out_path: str,
    title: str,
) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.figure(figsize=(10, 4.2))
    plt.plot(days, u, label="U")
    plt.plot(days, cs, label="CS")
    plt.plot(days, cr, label="CR")
    plt.plot(days, is_, label="IS")
    plt.plot(days, ir, label="IR")
    plt.xlabel("Day")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend(ncol=5, fontsize=9, frameon=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=600)
    plt.close()


# =============================================================================
# Population + ward assignment
# =============================================================================
def _choose_staff_wards(num_wards: int, wards_per_staff: int, rng: np.random.RandomState) -> List[int]:
    if num_wards <= 0:
        return [0]
    k = max(1, int(wards_per_staff))
    k = min(k, int(num_wards))
    wards = rng.choice(np.arange(int(num_wards)), size=k, replace=False)
    return sorted(int(w) for w in wards)


def _balanced_single_ward_assignments(
    n: int,
    num_wards: int,
    rng: np.random.RandomState,
) -> List[int]:
    """
    Balanced single-ward assignment with coverage guarantee when feasible.
    - If n >= W: guarantees each ward appears at least once.
    - If n < W: assigns all n to distinct wards.
    """
    n = int(n)
    W = int(num_wards)
    if W <= 0:
        return [0] * max(0, n)
    if n <= 0:
        return []

    if n >= W:
        wards = list(range(W))
        wards += [int(rng.randint(0, W)) for _ in range(n - W)]
        rng.shuffle(wards)
        return wards

    wards = rng.choice(np.arange(W), size=n, replace=False).tolist()
    wards = [int(w) for w in wards]
    rng.shuffle(wards)
    return wards


def _choose_staff_wards_with_home(
    num_wards: int,
    wards_per_staff: int,
    home_ward: int,
    rng: np.random.RandomState,
) -> List[int]:
    """
    Choose staff wards forcing inclusion of home_ward.
    Returns sorted unique ward indices.
    """
    W = int(num_wards)
    if W <= 0:
        return [0]

    home = int(home_ward) % W
    k = max(1, int(wards_per_staff))
    k = min(k, W)

    if k == 1:
        return [home]

    remaining = [w for w in range(W) if w != home]
    add_k = min(k - 1, len(remaining))
    extra = (
        rng.choice(np.array(remaining, dtype=int), size=add_k, replace=False).tolist()
        if add_k > 0
        else []
    )
    wards = sorted(set([home] + [int(x) for x in extra]))
    return wards


def _build_patients_by_ward(
    patients: List[str],
    ward_of: Dict[str, int],
    num_wards: int,
) -> Dict[int, List[str]]:
    """
    Group current patients by ward once and reuse the mapping in turnover/contact generation.
    """
    W = max(1, int(num_wards))
    out: Dict[int, List[str]] = {w: [] for w in range(W)}
    for p in patients:
        w = int(ward_of.get(p, 0))
        if w < 0:
            w = 0
        elif w >= W:
            w = W - 1
        out[w].append(p)
    return out


def build_population(
    num_patients: int,
    num_staff: int,
    num_wards: int,
    staff_wards_per_staff: int,
    seed: Optional[int] = None,
) -> Tuple[List[str], List[str], Dict[str, int], Dict[str, List[int]]]:
    """
    Patients: assigned exactly one ward.
    Staff:
      - home ward (ward_of[s]) is balanced with coverage guarantee if num_staff >= num_wards
      - multi-ward assignment (staff_wards[s]) always includes the home ward
    """
    patients = [f"p{i}" for i in range(int(num_patients))]
    staff = [f"s{i}" for i in range(int(num_staff))]

    rng = np.random.RandomState(seed if seed is not None else None)

    ward_of: Dict[str, int] = {}
    staff_wards: Dict[str, List[int]] = {}

    W = int(num_wards)
    if W <= 0:
        for p in patients:
            ward_of[p] = 0
        for s in staff:
            ward_of[s] = 0
            staff_wards[s] = [0]
        return patients, staff, ward_of, staff_wards

    patient_ward_list = _balanced_single_ward_assignments(len(patients), W, rng)
    for p, w in zip(patients, patient_ward_list):
        ward_of[p] = int(w)

    staff_home_list = _balanced_single_ward_assignments(len(staff), W, rng)
    for s, home in zip(staff, staff_home_list):
        home_w = int(home)
        wards = _choose_staff_wards_with_home(
            num_wards=W,
            wards_per_staff=int(staff_wards_per_staff),
            home_ward=home_w,
            rng=rng,
        )
        staff_wards[s] = wards
        ward_of[s] = home_w

    return patients, staff, ward_of, staff_wards


def _seasonal_multiplier(day: int, params: Params) -> float:
    """
    Returns a multiplicative factor for admission importation probabilities.
    """
    mode = str(getattr(params, "admit_import_seasonality", "none")).strip().lower()
    if mode in ("none", ""):
        return 1.0

    if mode == "sinusoid":
        amp = float(getattr(params, "admit_import_amp", 0.0))
        if amp <= 0.0:
            return 1.0
        amp = min(max(amp, 0.0), 0.99)
        period = int(getattr(params, "admit_import_period_days", 365))
        period = 1 if period <= 0 else period
        phase = int(getattr(params, "admit_import_phase_day", 0)) % period

        t = (int(day) - 1) % period
        x = 2.0 * math.pi * float(t - phase) / float(period)
        return float(1.0 + amp * math.sin(x))

    if mode == "piecewise":
        period = int(getattr(params, "admit_import_period_days", 365))
        period = 365 if period <= 0 else period
        t = ((int(day) - 1) % period) + 1

        hs = int(getattr(params, "admit_import_high_start_day", 1))
        he = int(getattr(params, "admit_import_high_end_day", 90))
        hs = 1 if hs < 1 else (period if hs > period else hs)
        he = 1 if he < 1 else (period if he > period else he)

        hi = float(getattr(params, "admit_import_high_mult", 1.5))
        lo = float(getattr(params, "admit_import_low_mult", 1.0))
        hi = max(0.0, hi)
        lo = max(0.0, lo)

        in_high = (hs <= t <= he) if hs <= he else (t >= hs or t <= he)
        return float(hi if in_high else lo)

    if mode == "shock":
        start = int(getattr(params, "admit_import_shock_start_day", 0))
        dur = int(getattr(params, "admit_import_shock_duration_days", 0))
        mult = float(getattr(params, "admit_import_shock_mult", 1.0))
        if start <= 0 or dur <= 0 or mult <= 0.0:
            return 1.0
        d = int(day)
        if start <= d <= (start + dur - 1):
            return float(mult)
        return 1.0

    return 1.0


# =============================================================================
# Dynamic census: admissions/discharges (patients only)
# =============================================================================
def _discharge_and_admit_patients(
    *,
    G_prev: nx.DiGraph,
    patients: List[str],
    ward_of: Dict[str, int],
    num_wards: int,
    day: int,
    params: Params,
    next_patient_id: int,
    rng: np.random.RandomState,
    intervention: Optional[CausalInterventionSpec] = None,
) -> Tuple[List[str], Dict[str, int], int, Dict[str, int]]:
    """
    Performs daily turnover for patients.
    """
    frac = float(getattr(params, "daily_discharge_frac", 0.0))
    min_per_ward = int(getattr(params, "daily_discharge_min_per_ward", 0))

    if frac <= 0.0 and min_per_ward <= 0:
        return patients, ward_of, next_patient_id, {w: 0 for w in range(int(num_wards))}

    for p in patients:
        if p in G_prev.nodes:
            G_prev.nodes[p]["new_import_cr_today"] = 0
            G_prev.nodes[p]["new_import_cs_today"] = 0
            G_prev.nodes[p]["new_import_ir_today"] = 0
            G_prev.nodes[p]["new_import_is_today"] = 0

    W = max(1, int(num_wards))
    current_by_ward = _build_patients_by_ward(
        patients=patients,
        ward_of=ward_of,
        num_wards=W,
    )

    discharged: List[str] = []
    admit_plan: Dict[int, int] = {w: 0 for w in range(W)}

    for w in range(W):
        ward_pat = current_by_ward.get(w, [])
        n_w = len(ward_pat)
        if n_w <= 0:
            continue

        k_frac = int(math.floor(frac * float(n_w))) if frac > 0.0 else 0
        k = max(k_frac, min_per_ward)
        k = min(k, n_w)

        if k <= 0:
            continue

        chosen = _stable_sample_without_replacement(ward_pat, k, params=params, fallback_rng=rng, stream_parts=("discharge", int(day), int(w)))
        discharged.extend([str(x) for x in chosen])
        admit_plan[w] = int(k)

    discharged_set = set(discharged)
    patients_kept = [p for p in patients if p not in discharged_set]

    for p in discharged:
        if p in G_prev:
            try:
                G_prev.remove_node(p)
            except Exception:
                pass

    admitted_counts_by_ward: Dict[str, int] = {}
    for w, k in admit_plan.items():
        if k <= 0:
            continue
        for _ in range(int(k)):
            pid = f"p{int(next_patient_id)}"
            next_patient_id += 1
            patients_kept.append(pid)
            ward_of[pid] = int(w)
            admitted_counts_by_ward[str(w)] = int(admitted_counts_by_ward.get(str(w), 0)) + 1

            if pid not in G_prev:
                G_prev.add_node(pid)

            G_prev.nodes[pid].update({
                "role": "patient",
                "ward_id": int(w),
                "ward_ids": str(int(w)),
                "amr_state": 0,
                "abx_class": 0,
                "is_isolated": 0,
                "isolation_days_remaining": 0,
                "screened_today": 0,
                "observed_pos": 0,
                "obs_status": 0,
                "days_since_last_test": 999,
                "pending_test_days": 0,
                "pending_test_result": 0,
                "pending_test_result_state": 0,
                "observed_positive_type": 0,
                "present_today": 1,
                "needs_admission_screen": 1,
                "new_cr_acq_today": 0,
                "new_ir_inf_today": 0,
                "new_import_cr_today": 0,
                "new_import_cs_today": 0,
                "new_import_ir_today": 0,
                "new_import_is_today": 0,
                "new_trans_cr_today": 0,
                "new_trans_cs_today": 0,
                "transmission_source": "",
                "transmission_type": "",
                "competing_exposure_count": 0,
                "new_select_cr_today": 0,
                "node_id": str(pid),
                "admission_day": int(day),
                "is_imported": 0,
            })

            r = _event_u01(params, "admission_importation", int(day), str(pid), fallback_rng=rng)

            base_p_cs = float(getattr(params, "p_admit_import_cs", params.p_import_cs))
            base_p_cr = float(getattr(params, "p_admit_import_cr", params.p_import_cr))
            base_p_is = float(getattr(params, "p_admit_import_is", 0.0))
            base_p_ir = float(getattr(params, "p_admit_import_ir", 0.0))

            m = float(_seasonal_multiplier(int(day), params))

            p_cs_max = float(getattr(params, "admit_import_pmax_cs", 1.0))
            p_cr_max = float(getattr(params, "admit_import_pmax_cr", 1.0))
            p_is_max = float(getattr(params, "admit_import_pmax_is", 1.0))
            p_ir_max = float(getattr(params, "admit_import_pmax_ir", 1.0))

            probs = {
                "CS": max(0.0, min(p_cs_max, base_p_cs * m)),
                "CR": max(0.0, min(p_cr_max, base_p_cr * m)),
                "IS": max(0.0, min(p_is_max, base_p_is * m)),
                "IR": max(0.0, min(p_ir_max, base_p_ir * m)),
            }

            if _intervention_active_on_day(intervention, day) and intervention is not None and intervention.name == "reduce_ward_importation":
                target = str(intervention.target_id).strip()
                target_matches = target in {str(w), f"w{w}", f"ward{w}", f"ward_{w}"}
                if target_matches:
                    mult = float(intervention.params.get("multiplier", 1.0))
                    probs["CS"] = max(0.0, min(p_cs_max, probs["CS"] * float(intervention.params.get("multiplier_cs", mult))))
                    probs["CR"] = max(0.0, min(p_cr_max, probs["CR"] * float(intervention.params.get("multiplier_cr", mult))))
                    probs["IS"] = max(0.0, min(p_is_max, probs["IS"] * float(intervention.params.get("multiplier_is", mult))))
                    probs["IR"] = max(0.0, min(p_ir_max, probs["IR"] * float(intervention.params.get("multiplier_ir", mult))))

            imported_state = _draw_state_from_weights(r, probs)
            if imported_state is not None:
                state_code = {"CS": 1, "CR": 2, "IS": 3, "IR": 4}[imported_state]
                G_prev.nodes[pid]["amr_state"] = int(state_code)
                G_prev.nodes[pid]["is_imported"] = 1
                if imported_state == "CS":
                    G_prev.nodes[pid]["new_import_cs_today"] = 1
                elif imported_state == "CR":
                    G_prev.nodes[pid]["new_import_cr_today"] = 1
                elif imported_state == "IS":
                    G_prev.nodes[pid]["new_import_is_today"] = 1
                elif imported_state == "IR":
                    G_prev.nodes[pid]["new_import_ir_today"] = 1

    for w in range(W):
        admitted_counts_by_ward.setdefault(str(w), 0)

    admitted_counts_int = {int(k): int(v) for k, v in admitted_counts_by_ward.items()}
    return patients_kept, ward_of, next_patient_id, admitted_counts_int


# =============================================================================
# Contact graph generation + superspreader injection
# =============================================================================
def _apply_superspreader_injection(
    G: nx.DiGraph,
    *,
    superspreader_staff: str,
    patients: List[str],
    staff: List[str],
    ward_of: Dict[str, int],
    staff_wards: Dict[str, List[int]],
    num_wards: int,
    staff_removed: Dict[str, bool],
    staff_patient_frac_per_ward: float,
    staff_patient_min_per_ward: int,
    superspreader_patient_frac_mult: float,
    superspreader_patient_min_add: int,
    superspreader_staff_contacts: int,
    superspreader_edge_weight_mult: float,
    params: Optional[Params] = None,
    day: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> None:
    """
    In-place injection for superspreader behavior.
    """
    s = str(superspreader_staff).strip()
    if s == "" or s not in G.nodes:
        return
    if staff_removed.get(s, False):
        return

    wards_for_s = staff_wards.get(s, [ward_of.get(s, 0)])
    wards_for_s = [w for w in sorted(set(int(w) for w in wards_for_s)) if 0 <= int(w) < int(num_wards)]

    for w in wards_for_s:
        ward_pat = [p for p in patients if ward_of.get(p, -1) == w]
        if not ward_pat:
            continue

        base_k = max(int(staff_patient_min_per_ward), int(len(ward_pat) * float(staff_patient_frac_per_ward)))
        boosted_k = int(math.ceil(float(base_k) * float(superspreader_patient_frac_mult))) + int(superspreader_patient_min_add)
        boosted_k = max(base_k, boosted_k)
        boosted_k = min(boosted_k, len(ward_pat))

        if boosted_k <= 0:
            continue

        chosen = _stable_sample_without_replacement(
            ward_pat,
            boosted_k,
            params=params,
            fallback_rng=rng,
            stream_parts=("superspreader_patients", int(day) if day is not None else -1, str(s), int(w)),
        )
        for p in chosen:
            w_sp = _event_uniform(params, 0.6, 1.2, "superspreader_s_to_p", int(day) if day is not None else -1, str(s), str(p), fallback_rng=rng)
            w_ps = _event_uniform(params, 0.2, 0.8, "superspreader_p_to_s", int(day) if day is not None else -1, str(p), str(s), fallback_rng=rng)
            G.add_edge(s, p, weight=w_sp, edge_type=1)
            G.add_edge(p, s, weight=w_ps, edge_type=1)

    active_staff = [x for x in staff if (x in G.nodes and not staff_removed.get(x, False) and x != s)]
    if len(active_staff) > 0 and int(superspreader_staff_contacts) > 0:
        k = min(int(superspreader_staff_contacts), len(active_staff))
        chosen_staff = _stable_sample_without_replacement(
            active_staff,
            k,
            params=params,
            fallback_rng=rng,
            stream_parts=("superspreader_staff", int(day) if day is not None else -1, str(s)),
        )
        for t in chosen_staff:
            w_st = _event_uniform(params, 0.3, 0.9, "superspreader_s_to_staff", int(day) if day is not None else -1, str(s), str(t), fallback_rng=rng)
            w_ts = _event_uniform(params, 0.3, 0.9, "superspreader_staff_to_s", int(day) if day is not None else -1, str(t), str(s), fallback_rng=rng)
            G.add_edge(s, t, weight=w_st, edge_type=2)
            G.add_edge(t, s, weight=w_ts, edge_type=2)

    mult = float(superspreader_edge_weight_mult)
    if mult != 1.0:
        for _, _, attrs in list(G.in_edges(s, data=True)):
            attrs["weight"] = float(attrs.get("weight", 1.0)) * mult
        for _, _, attrs in list(G.out_edges(s, data=True)):
            attrs["weight"] = float(attrs.get("weight", 1.0)) * mult


def _sample_unique_staff_pairs(
    active_staff: List[str],
    pair_prob: float,
    *,
    params: Optional[Params] = None,
    day: Optional[int] = None,
    rng: Optional[np.random.RandomState] = None,
) -> List[Tuple[str, str]]:
    """
    Sample distinct undirected staff-staff pairs without scanning all O(S^2) pairs.
    """
    S = len(active_staff)
    if S < 2:
        return []

    p = float(pair_prob)
    if p <= 0.0:
        return []
    if p >= 1.0:
        return [(active_staff[i], active_staff[j]) for i in range(S) for j in range(i + 1, S)]

    total_pairs = S * (S - 1) // 2
    if _use_common_random_numbers(params):
        pairs: List[Tuple[str, str]] = []
        d = int(day) if day is not None else -1
        for i in range(S - 1):
            for j in range(i + 1, S):
                u = _event_u01(params, "staff_staff_pair", d, str(active_staff[i]), str(active_staff[j]))
                if u < p:
                    pairs.append((active_staff[i], active_staff[j]))
        return pairs

    local_rng = rng if rng is not None else np.random
    k = int(local_rng.binomial(total_pairs, p))
    if k <= 0:
        return []
    if k >= total_pairs:
        return [(active_staff[i], active_staff[j]) for i in range(S) for j in range(i + 1, S)]

    chosen_idx = local_rng.choice(total_pairs, size=k, replace=False)
    chosen_idx.sort()

    pairs: List[Tuple[str, str]] = []
    idx_ptr = 0
    running = 0

    for i in range(S - 1):
        row_len = S - i - 1
        row_start = running
        row_end = running + row_len

        while idx_ptr < k and chosen_idx[idx_ptr] < row_end:
            offset = int(chosen_idx[idx_ptr] - row_start)
            j = i + 1 + offset
            pairs.append((active_staff[i], active_staff[j]))
            idx_ptr += 1

        running = row_end
        if idx_ptr >= k:
            break

    return pairs


def sample_contacts(
    patients: List[str],
    staff: List[str],
    ward_of: Dict[str, int],
    staff_wards: Dict[str, List[int]],
    num_wards: int,
    staff_removed: Dict[str, bool],
    staff_patient_frac_per_ward: float = 0.06,
    staff_patient_min_per_ward: int = 3,
    superspreader_staff: Optional[str] = None,
    superspreader_active: bool = False,
    superspreader_patient_frac_mult: float = 3.0,
    superspreader_patient_min_add: int = 10,
    superspreader_staff_contacts: int = 30,
    superspreader_edge_weight_mult: float = 1.5,
    intervention: Optional[CausalInterventionSpec] = None,
    day: Optional[int] = None,
    params: Optional[Params] = None,
) -> nx.DiGraph:
    G = nx.DiGraph()
    G.add_nodes_from(patients + staff)

    W = max(1, int(num_wards))
    stream_rng = _stream_rng(params, "contacts", int(day) if day is not None else -1) if _use_common_random_numbers(params) else None
    patients_by_ward = _build_patients_by_ward(
        patients=patients,
        ward_of=ward_of,
        num_wards=W,
    )

    for w in range(W):
        ward_pat = patients_by_ward.get(w, [])
        n_pat = len(ward_pat)
        if n_pat <= 1:
            continue

        if stream_rng is not None:
            sf_seed = int(_stable_hash_int(_noise_base_seed(params), _noise_region_salt(params), "scale_free", int(day) if day is not None else -1, int(w)) % (2**31 - 1))
            if sf_seed <= 0:
                sf_seed = 1
            Gp = nx.scale_free_graph(n_pat, seed=sf_seed)
        else:
            Gp = nx.scale_free_graph(n_pat)
        Gp = nx.DiGraph(Gp)
        mapping = {i: ward_pat[i] for i in range(n_pat)}
        Gp = nx.relabel_nodes(Gp, mapping)

        for u, v in Gp.edges:
            if u == v:
                continue
            w_uv = _event_uniform(params, 0.3, 1.0, "pp_edge_weight", int(day) if day is not None else -1, str(u), str(v), fallback_rng=stream_rng)
            G.add_edge(u, v, weight=w_uv, edge_type=0)

    active_staff: List[str] = [s for s in staff if not staff_removed.get(s, False)]

    staff_by_ward: Dict[int, List[str]] = {w: [] for w in range(W)}
    for s in active_staff:
        wards_for_s = staff_wards.get(s, [ward_of.get(s, 0)])
        wards_for_s = [w for w in sorted(set(int(w) for w in wards_for_s)) if 0 <= int(w) < W]
        for w in wards_for_s:
            staff_by_ward[w].append(s)

    staff_patient_contacts: Dict[str, set] = {s: set() for s in active_staff}

    for w in range(W):
        ward_pat = patients_by_ward.get(w, [])
        ward_staff = staff_by_ward.get(w, [])

        if not ward_pat or not ward_staff:
            continue

        ward_pat_shuffled = _stable_shuffle(list(ward_pat), params=params, fallback_rng=stream_rng, stream_parts=("ward_pat_shuffle", int(day) if day is not None else -1, int(w)))
        ward_staff_shuffled = _stable_shuffle(list(ward_staff), params=params, fallback_rng=stream_rng, stream_parts=("ward_staff_shuffle", int(day) if day is not None else -1, int(w)))

        n_staff = len(ward_staff_shuffled)

        for i, p in enumerate(ward_pat_shuffled):
            s = ward_staff_shuffled[i % n_staff]

            if p not in staff_patient_contacts[s]:
                w_sp = _event_uniform(params, 0.6, 1.2, "staff_patient_s_to_p", int(day) if day is not None else -1, str(s), str(p), fallback_rng=stream_rng)
                w_ps = _event_uniform(params, 0.2, 0.8, "staff_patient_p_to_s", int(day) if day is not None else -1, str(p), str(s), fallback_rng=stream_rng)
                G.add_edge(s, p, weight=w_sp, edge_type=1)
                G.add_edge(p, s, weight=w_ps, edge_type=1)
                staff_patient_contacts[s].add(p)

    for s in active_staff:
        wards_for_s = staff_wards.get(s, [ward_of.get(s, 0)])
        wards_for_s = [w for w in sorted(set(int(w) for w in wards_for_s)) if 0 <= int(w) < W]

        for w in wards_for_s:
            ward_pat = patients_by_ward.get(w, [])
            n_pat = len(ward_pat)
            if n_pat == 0:
                continue

            base_k = min(
                n_pat,
                max(int(staff_patient_min_per_ward), int(n_pat * float(staff_patient_frac_per_ward))),
            )

            already_assigned_here = [p for p in ward_pat if p in staff_patient_contacts[s]]
            extra_k = max(0, base_k - len(already_assigned_here))
            if extra_k <= 0:
                continue

            remaining = [p for p in ward_pat if p not in staff_patient_contacts[s]]
            if not remaining:
                continue

            extra_k = min(extra_k, len(remaining))
            chosen = _stable_sample_without_replacement(
                remaining,
                extra_k,
                params=params,
                fallback_rng=stream_rng,
                stream_parts=("extra_staff_patients", int(day) if day is not None else -1, str(s), int(w)),
            )

            for p in chosen:
                w_sp = _event_uniform(params, 0.6, 1.2, "extra_staff_patient_s_to_p", int(day) if day is not None else -1, str(s), str(p), fallback_rng=stream_rng)
                w_ps = _event_uniform(params, 0.2, 0.8, "extra_staff_patient_p_to_s", int(day) if day is not None else -1, str(p), str(s), fallback_rng=stream_rng)
                G.add_edge(s, p, weight=w_sp, edge_type=1)
                G.add_edge(p, s, weight=w_ps, edge_type=1)
                staff_patient_contacts[s].add(str(p))

    staff_staff_pairs = _sample_unique_staff_pairs(active_staff, pair_prob=0.05, params=params, day=day, rng=stream_rng)
    for s, t in staff_staff_pairs:
        w_st = _event_uniform(params, 0.3, 0.9, "staff_staff_s_to_t", int(day) if day is not None else -1, str(s), str(t), fallback_rng=stream_rng)
        w_ts = _event_uniform(params, 0.3, 0.9, "staff_staff_t_to_s", int(day) if day is not None else -1, str(t), str(s), fallback_rng=stream_rng)
        G.add_edge(s, t, weight=w_st, edge_type=2)
        G.add_edge(t, s, weight=w_ts, edge_type=2)

    if superspreader_active and superspreader_staff is not None and str(superspreader_staff).strip() != "":
        _apply_superspreader_injection(
            G,
            superspreader_staff=str(superspreader_staff).strip(),
            patients=patients,
            staff=staff,
            ward_of=ward_of,
            staff_wards=staff_wards,
            num_wards=W,
            staff_removed=staff_removed,
            staff_patient_frac_per_ward=staff_patient_frac_per_ward,
            staff_patient_min_per_ward=staff_patient_min_per_ward,
            superspreader_patient_frac_mult=float(superspreader_patient_frac_mult),
            superspreader_patient_min_add=int(superspreader_patient_min_add),
            superspreader_staff_contacts=int(superspreader_staff_contacts),
            superspreader_edge_weight_mult=float(superspreader_edge_weight_mult),
            params=params,
            day=day,
            rng=stream_rng,
        )

    _apply_contact_level_intervention(G, intervention, day=day)
    return G


# =============================================================================
# State helpers
# =============================================================================
def is_colonised(state: int) -> bool:
    return int(state) in (1, 2, 3, 4)


def is_infected(state: int) -> bool:
    return int(state) in (3, 4)


def is_resistant(state: int) -> bool:
    return int(state) in (2, 4)


def _count_states(G: nx.DiGraph) -> Dict[str, int]:
    cs = cr = is_ = ir = u = 0
    for _, attrs in G.nodes(data=True):
        st = int(attrs.get("amr_state", 0))
        if st == 0:
            u += 1
        elif st == 1:
            cs += 1
        elif st == 2:
            cr += 1
        elif st == 3:
            is_ += 1
        elif st == 4:
            ir += 1
    return {"u": u, "cs": cs, "cr": cr, "is": is_, "ir": ir}


def _resistant_fraction(counts: Dict[str, int]) -> float:
    denom = float(counts["cs"] + counts["cr"] + counts["is"] + counts["ir"])
    if denom <= 0.0:
        return 0.0
    return float(counts["cr"] + counts["ir"]) / denom


# =============================================================================
# Initialisation
# =============================================================================
def init_states(
    G: nx.DiGraph,
    patients: List[str],
    staff: List[str],
    ward_of: Dict[str, int],
    staff_wards: Dict[str, List[int]],
    params: Params,
) -> None:
    for n in G.nodes:
        n_key = n if isinstance(n, str) else str(n)
        is_staff = n_key.startswith("s")
        home_ward = int(ward_of.get(n_key, ward_of.get(n, 0)))

        if is_staff:
            wards = staff_wards.get(n_key, staff_wards.get(n, [home_ward]))
            wards = [w for w in sorted(set(int(w) for w in wards))]
            ward_ids_str = ",".join(str(w) for w in wards) if wards else str(home_ward)
        else:
            ward_ids_str = str(home_ward)

        G.nodes[n].update({
            "role": "staff" if is_staff else "patient",
            "ward_id": home_ward,
            "ward_ids": ward_ids_str,
            "amr_state": 0,
            "abx_class": 0,
            "is_isolated": 0,
            "isolation_days_remaining": 0,
            "screened_today": 0,
            "observed_pos": 0,
            "obs_status": 0,
            "days_since_last_test": 999,
            "pending_test_days": 0,
            "pending_test_result": 0,
                "pending_test_result_state": 0,
                "observed_positive_type": 0,
            "present_today": 1,
            "needs_admission_screen": 0,
            "new_cr_acq_today": 0,
            "new_ir_inf_today": 0,
            "new_import_cr_today": 0,
            "new_import_cs_today": 0,
            "new_import_ir_today": 0,
            "new_import_is_today": 0,
            "new_trans_cr_today": 0,
                "new_trans_cs_today": 0,
                "transmission_source": "",
                "transmission_type": "",
                "competing_exposure_count": 0,
            "new_select_cr_today": 0,
            "node_id": str(n),
        })

    for p in patients:
        r = _event_u01(params, "init_patient_amr", str(p))
        seeded_state = _draw_state_from_weights(
            r,
            {
                "CS": float(getattr(params, "p_import_cs", 0.0)),
                "CR": float(getattr(params, "p_import_cr", 0.0)),
                "IS": float(getattr(params, "p_import_is", 0.0)),
                "IR": float(getattr(params, "p_import_ir", 0.0)),
            },
        )
        if seeded_state is not None:
            G.nodes[p]["amr_state"] = {"CS": 1, "CR": 2, "IS": 3, "IR": 4}[seeded_state]

    for s in staff:
        if _event_u01(params, "init_staff_cs", str(s)) < 0.01:
            G.nodes[s]["amr_state"] = 1
        if _event_u01(params, "init_staff_cr", str(s)) < 0.003:
            G.nodes[s]["amr_state"] = 2

    G.graph["new_cr_acq_total"] = 0
    G.graph["new_ir_inf_total"] = 0
    G.graph["new_import_cr_total"] = 0
    G.graph["new_import_cs_total"] = 0
    G.graph["new_import_ir_total"] = 0
    G.graph["new_import_is_total"] = 0
    G.graph["new_trans_cr_total"] = 0
    G.graph["new_select_cr_total"] = 0
    counts = _count_states(G)
    G.graph["resistant_fraction"] = float(_resistant_fraction(counts))




def _state_resistance_label(state: int) -> str:
    return "resistant" if is_resistant(int(state)) else "sensitive"


def _state_from_resistance_label(label: str) -> int:
    return 2 if str(label).strip().lower().startswith("res") else 1


def _source_state_beta_multiplier(state: int, params: Params) -> float:
    st = int(state)
    if st == 1:
        return max(0.0, float(getattr(params, "beta_state_mult_cs", 0.5)))
    if st == 2:
        return max(0.0, float(getattr(params, "beta_state_mult_cr", 0.5)))
    if st == 3:
        return max(0.0, float(getattr(params, "beta_state_mult_is", 1.0)))
    if st == 4:
        return max(0.0, float(getattr(params, "beta_state_mult_ir", 1.0)))
    return 0.0


def _is_crossward_staff_patient_contact(G: nx.DiGraph, src: str, tgt: str, edge_type: int) -> bool:
    if int(edge_type) not in (1, 3):
        return False
    src_role = str(G.nodes[src].get("role", ""))
    tgt_role = str(G.nodes[tgt].get("role", ""))
    staff_node = src if src_role == "staff" else (tgt if tgt_role == "staff" else None)
    patient_node = src if src_role == "patient" else (tgt if tgt_role == "patient" else None)
    if staff_node is None or patient_node is None:
        return False
    staff_home = int(G.nodes[staff_node].get("ward_id", -1))
    patient_ward = int(G.nodes[patient_node].get("ward_id", -1))
    return staff_home >= 0 and patient_ward >= 0 and staff_home != patient_ward


def _hand_hygiene_multiplier(G: nx.DiGraph, src: str, tgt: str, edge_type: int, params: Params) -> float:
    if int(getattr(params, "crossward_hand_hygiene_enabled", 1)) != 1:
        return 1.0
    if not _is_crossward_staff_patient_contact(G, src, tgt, int(edge_type)):
        return 1.0
    return max(0.0, min(1.0, float(getattr(params, "crossward_hand_hygiene_mult", 0.55))))


def _choose_infecting_event(events: List[Dict[str, Any]], target_on_abx: bool) -> Dict[str, Any]:
    if len(events) == 1:
        return events[0]
    weights: List[float] = []
    for event in events:
        label = str(event.get("type", "sensitive"))
        if target_on_abx:
            weights.append(float(event.get("abx_weight", 1.0)) if label == "resistant" else 1.0)
        else:
            weights.append(float(event.get("no_abx_weight", 1.0)) if label == "sensitive" else 1.0)
    total = float(sum(max(0.0, w) for w in weights))
    if total <= 0.0:
        return events[int(np.random.randint(0, len(events)))]
    draw = float(np.random.rand()) * total
    acc = 0.0
    for event, weight in zip(events, weights):
        acc += max(0.0, float(weight))
        if draw <= acc:
            return event
    return events[-1]


def _choose_infecting_event_causal(
    events: List[Dict[str, Any]],
    target_on_abx: bool,
    params: Params,
    day: int,
    target: str,
) -> Dict[str, Any]:
    if len(events) == 1:
        return events[0]
    weights: List[float] = []
    for event in events:
        label = str(event.get("type", "sensitive"))
        if target_on_abx:
            weights.append(float(event.get("abx_weight", 1.0)) if label == "resistant" else 1.0)
        else:
            weights.append(float(event.get("no_abx_weight", 1.0)) if label == "sensitive" else 1.0)
    total = float(sum(max(0.0, w) for w in weights))
    if total <= 0.0:
        idx = _event_randint(params, 0, len(events), "competing_exposure_choice_uniform", int(day), str(target))
        return events[int(idx)]
    draw = _event_u01(params, "competing_exposure_choice", int(day), str(target)) * total
    acc = 0.0
    for event, weight in zip(events, weights):
        acc += max(0.0, float(weight))
        if draw <= acc:
            return event
    return events[-1]


# =============================================================================
# Screening + isolation
# =============================================================================
def _is_screening_day(day: int, weekly_screen_day: int) -> bool:
    d = int(day)
    w = int(weekly_screen_day)
    w = 1 if w < 1 else (7 if w > 7 else w)
    return ((d - 1) % 7) + 1 == w


def _is_screening_day_with_params(
    day: int,
    params: Params,
    intervention: Optional[CausalInterventionSpec] = None,
) -> bool:
    d = int(day)
    k = _effective_screen_every_k_days(params, intervention, day=d)
    if k > 0:
        anchor = _screening_anchor_day(params, intervention, day=d)
        return (d - anchor) % int(k) == 0
    return _is_screening_day(day, int(getattr(params, "weekly_screen_day", 7)))


def apply_isolation_decay(G: nx.DiGraph) -> None:
    for _, attrs in G.nodes(data=True):
        rem = int(attrs.get("isolation_days_remaining", 0))
        if rem > 0:
            rem -= 1
            attrs["isolation_days_remaining"] = rem
            if rem <= 0:
                attrs["is_isolated"] = 0


def _effective_isolation_trigger_policy(
    params: Params,
    intervention: Optional[CausalInterventionSpec],
    day: Optional[int] = None,
) -> str:
    value = str(getattr(params, "isolation_trigger_policy", "resistant_only")).strip().lower()
    if _intervention_active_on_day(intervention, day) and intervention is not None:
        try:
            if intervention.name in {"set_isolation_parameters", "set_screening_frequency"}:
                value = str(intervention.params.get("isolation_trigger_policy", value)).strip().lower()
        except Exception:
            pass
    if value in {"all", "all_positive", "all_positives", "any_positive", "any"}:
        return "all_positives"
    return "resistant_only"


def _test_result_resistance_code(state: int, obs_pos: int) -> int:
    if int(obs_pos) != 1:
        return 0
    st = int(state)
    if is_resistant(st):
        return 2
    if is_colonised(st):
        return 1
    return 0


def _should_isolate_from_result(
    attrs: Dict[str, Any],
    params: Params,
    obs_pos: int,
    result_state: int,
    intervention: Optional[CausalInterventionSpec] = None,
    day: Optional[int] = None,
) -> bool:
    if int(obs_pos) != 1:
        return False
    if (
        _intervention_active_on_day(intervention, day)
        and intervention is not None
        and intervention.name == "disable_isolation_response"
    ):
        return False
    policy = _effective_isolation_trigger_policy(params, intervention, day=day)
    if policy == "all_positives":
        return True
    return int(result_state) == 2


def _apply_test_result(
    attrs: Dict[str, Any],
    params: Params,
    obs_pos: int,
    result_state: int = 0,
    intervention: Optional[CausalInterventionSpec] = None,
    day: Optional[int] = None,
) -> None:
    obs_pos_i = int(obs_pos)
    result_state_i = int(result_state)
    attrs["observed_pos"] = obs_pos_i
    attrs["observed_positive_type"] = result_state_i
    attrs["obs_status"] = 2 if obs_pos_i == 1 else 1
    if _should_isolate_from_result(attrs, params, obs_pos_i, result_state_i, intervention=intervention, day=day):
        attrs["is_isolated"] = 1
        attrs["isolation_days_remaining"] = _effective_isolation_days(params, intervention, day=day)


def _schedule_test(
    attrs: Dict[str, Any],
    params: Params,
    intervention: Optional[CausalInterventionSpec] = None,
    day: Optional[int] = None,
) -> None:
    sens = max(0.0, min(1.0, float(getattr(params, "screen_sens", 0.90))))
    spec = max(0.0, min(1.0, float(getattr(params, "screen_spec", 0.99))))
    delay = _effective_screen_result_delay_days(params, intervention, day=day)
    delay = 0 if delay < 0 else delay

    node_id = str(attrs.get("node_id", attrs.get("id", "")))
    true_state = int(attrs.get("amr_state", 0))
    true_pos = is_colonised(true_state)
    u_test = _event_u01(params, "screen_test_result", int(day) if day is not None else -1, node_id)
    if true_pos:
        obs_pos = 1 if (u_test < sens) else 0
    else:
        obs_pos = 1 if (u_test < (1.0 - spec)) else 0

    result_state = _test_result_resistance_code(true_state, obs_pos)
    if delay > 0:
        attrs["pending_test_days"] = int(delay)
        attrs["pending_test_result"] = int(obs_pos)
        attrs["pending_test_result_state"] = int(result_state)
    else:
        attrs["pending_test_days"] = 0
        attrs["pending_test_result"] = int(obs_pos)
        attrs["pending_test_result_state"] = int(result_state)
        _apply_test_result(attrs, params, int(obs_pos), int(result_state), intervention=intervention, day=day)

def apply_pending_tests(
    G: nx.DiGraph,
    params: Params,
    day: int,
    intervention: Optional[CausalInterventionSpec] = None,
) -> None:
    """
    Applies delayed test results once their countdown reaches 0.
    """
    for _, attrs in G.nodes(data=True):
        role = attrs.get("role")
        if role not in ("patient", "staff"):
            continue
        pending = int(attrs.get("pending_test_days", 0))
        if pending > 0:
            pending -= 1
            attrs["pending_test_days"] = pending
            if pending == 0:
                res = int(attrs.get("pending_test_result", 0))
                res_state = int(attrs.get("pending_test_result_state", 0))
                _apply_test_result(attrs, params, res, res_state, intervention=intervention, day=day)


def update_days_since_last_test(G: nx.DiGraph) -> None:
    """
    Updates days_since_last_test counter.
    """
    for _, attrs in G.nodes(data=True):
        role = attrs.get("role")
        if role not in ("patient", "staff"):
            continue
        if int(attrs.get("screened_today", 0)) == 1:
            attrs["days_since_last_test"] = 0
        else:
            ds = int(attrs.get("days_since_last_test", 999))
            ds = 999 if ds >= 999 else (ds + 1)
            attrs["days_since_last_test"] = ds


def reset_daily_observation_flags(G: nx.DiGraph, params: Params) -> None:
    """
    Ensures "screened_today" is a true per-day indicator.
    """
    persist = int(getattr(params, "persist_observations", 0)) == 1
    for _, attrs in G.nodes(data=True):
        attrs["screened_today"] = 0
        if not persist:
            attrs["observed_pos"] = 0
            attrs["obs_status"] = 0


def run_admission_screening(
    G: nx.DiGraph,
    params: Params,
    day: int,
    intervention: Optional[CausalInterventionSpec] = None,
) -> None:
    """Admission-day testing for newly admitted patients.

    Semantics:
      - screen_on_admission=1 screens all newly admitted patients, including
        imported CS/CR/IS/IR and uncolonised U admissions.
      - screen_on_admission=0 with admission_screen_symptomatic_only=1 screens
        newly admitted symptomatic/infected patients only (IS/IR).
      - both switches off means no admission-day test.
    """
    d = int(day)
    universal_admission_screen = _effective_screen_on_admission(params, intervention, day=d) == 1
    symptomatic_admission_screen = int(getattr(params, "admission_screen_symptomatic_only", 1)) == 1
    if not universal_admission_screen and not symptomatic_admission_screen:
        return

    for _, attrs in G.nodes(data=True):
        if attrs.get("role") != "patient":
            continue
        if int(attrs.get("needs_admission_screen", 0)) != 1:
            continue
        if int(attrs.get("admission_day", -1)) != d:
            continue

        state = int(attrs.get("amr_state", 0))
        should_screen = bool(universal_admission_screen or (symptomatic_admission_screen and is_infected(state)))
        if not should_screen:
            attrs["needs_admission_screen"] = 0
            continue

        attrs["screened_today"] = 1
        _schedule_test(attrs, params, intervention=intervention, day=d)
        attrs["needs_admission_screen"] = 0


def run_screening(
    G: nx.DiGraph,
    params: Params,
    day: int,
    intervention: Optional[CausalInterventionSpec] = None,
) -> None:
    """
    Passive routine screening. If passive_screen_capacity_per_week > 0, only a convenient
    sample up to the weekly capacity is screened across eligible patients/staff.
    """
    if not _is_screening_day_with_params(day, params, intervention=intervention):
        return

    eligible: List[Tuple[str, Dict[str, Any]]] = []
    for n, attrs in G.nodes(data=True):
        role = attrs.get("role")
        if role not in ("patient", "staff"):
            continue
        if role == "staff" and int(attrs.get("present_today", 1)) != 1:
            continue
        if int(attrs.get("screened_today", 0)) == 1:
            continue
        if int(attrs.get("pending_test_days", 0)) > 0:
            continue
        eligible.append((str(n), attrs))

    capacity = int(getattr(params, "passive_screen_capacity_per_week", 0))
    if capacity > 0 and len(eligible) > capacity:
        keyed = sorted(eligible, key=lambda item: _stable_hash_int("passive_screen_rank", int(day), item[0], int(getattr(params, "runtime_seed_base", 0))))
        selected = keyed[:capacity]
    else:
        selected = eligible

    for _, attrs in selected:
        attrs["screened_today"] = 1
        _schedule_test(attrs, params, intervention=intervention, day=int(day))


def _apply_staff_presence_from_removed(G: nx.DiGraph, staff_removed_map: Dict[str, bool]) -> None:
    """Synchronise staff present_today with the effective removal map for the day."""
    for n, attrs in G.nodes(data=True):
        if attrs.get("role") == "staff":
            attrs["present_today"] = 0 if bool(staff_removed_map.get(str(n), False)) else 1


# =============================================================================
# Daily dynamics
# =============================================================================
def run_day_transmission(
    G: nx.DiGraph,
    params: Params,
    staff_removed: Dict[str, bool],
    staff_timer: Dict[str, int],
    intervention: Optional[CausalInterventionSpec] = None,
    day: Optional[int] = None,
) -> None:
    nodes = list(G.nodes)
    d = int(day) if day is not None else -1

    for n in nodes:
        G.nodes[n]["new_cr_acq_today"] = 0
        G.nodes[n]["new_ir_inf_today"] = 0
        G.nodes[n]["new_trans_cr_today"] = 0
        G.nodes[n]["new_trans_cs_today"] = 0
        G.nodes[n]["new_select_cr_today"] = 0
        G.nodes[n]["transmission_source"] = ""
        G.nodes[n]["transmission_type"] = ""
        G.nodes[n]["competing_exposure_count"] = 0

    for n in nodes:
        st = int(G.nodes[n]["amr_state"])
        abx = int(G.nodes[n]["abx_class"])

        if abx == 0:
            if is_infected(st) and _event_u01(params, "abx_start_if_inf", d, str(n)) < float(params.p_start_abx_if_inf):
                G.nodes[n]["abx_class"] = _event_randint(params, 1, max(1, int(params.n_abx_classes)) + 1, "abx_class", d, str(n))
            elif _event_u01(params, "abx_start_if_not_inf", d, str(n)) < float(params.p_start_abx_if_not_inf):
                G.nodes[n]["abx_class"] = _event_randint(params, 1, max(1, int(params.n_abx_classes)) + 1, "abx_class", d, str(n))
        else:
            if _event_u01(params, "abx_stop", d, str(n)) < float(params.p_stop_abx):
                G.nodes[n]["abx_class"] = 0

    for n in nodes:
        st_before = int(G.nodes[n]["amr_state"])
        abx = int(G.nodes[n]["abx_class"])

        if abx > 0:
            if st_before == 1 and _event_u01(params, "select_col", d, str(n)) < float(params.p_select_col):
                G.nodes[n]["amr_state"] = 2
                G.nodes[n]["new_cr_acq_today"] = 1
                G.nodes[n]["new_select_cr_today"] = 1

            st_now = int(G.nodes[n]["amr_state"])
            if st_now == 3 and _event_u01(params, "select_inf", d, str(n)) < float(params.p_select_inf):
                G.nodes[n]["amr_state"] = 4
                G.nodes[n]["new_ir_inf_today"] = 1

    new_state: Dict[str, int] = {n: int(G.nodes[n]["amr_state"]) for n in nodes}
    transmission_events: List[Dict[str, Any]] = []

    for tgt in nodes:
        if new_state[tgt] != 0:
            continue

        successful_events: List[Dict[str, Any]] = []
        for src in G.predecessors(tgt):
            st_src = int(G.nodes[src]["amr_state"])
            if not is_colonised(st_src):
                continue

            e = G[src][tgt]
            edge_type = int(e.get("edge_type", 0))
            weight = float(e.get("weight", 1.0))

            if edge_type == 0:
                beta = float(params.beta_pp)
            elif edge_type in (1, 3):
                beta = float(params.beta_sp)
            else:
                beta = float(params.beta_ss)

            beta *= _source_state_beta_multiplier(st_src, params)
            if is_resistant(st_src):
                beta *= float(getattr(params, "beta_res_mult", 1.0))

            iso_src = float(_effective_isolation_mult(params, intervention, day=day)) if int(G.nodes[src].get("is_isolated", 0)) else 1.0
            iso_tgt = float(_effective_isolation_mult(params, intervention, day=day)) if int(G.nodes[tgt].get("is_isolated", 0)) else 1.0
            hygiene_mult = _hand_hygiene_multiplier(G, src, tgt, edge_type, params)
            total_mult = iso_src * iso_tgt * hygiene_mult

            p = 1.0 - math.exp(-beta * weight)
            p_eff = max(0.0, min(1.0, p * total_mult))
            if _event_u01(params, "transmission_event", d, str(src), str(tgt)) < p_eff:
                label = _state_resistance_label(st_src)
                successful_events.append({
                    "source": str(src),
                    "target": str(tgt),
                    "type": label,
                    "source_state": AMR_STATE_STR.get(st_src, str(st_src)),
                    "edge_type": int(edge_type),
                    "weight": float(weight),
                    "probability": float(p_eff),
                    "hand_hygiene_multiplier": float(hygiene_mult),
                    "no_abx_weight": float(getattr(params, "no_abx_sensitive_selection_weight", 4.0)),
                    "abx_weight": float(getattr(params, "abx_resistant_selection_weight", 4.0)),
                })

        if successful_events:
            chosen = _choose_infecting_event_causal(successful_events, target_on_abx=int(G.nodes[tgt].get("abx_class", 0)) > 0, params=params, day=d, target=str(tgt))
            chosen["competing_exposure_count"] = int(len(successful_events))
            chosen["target_on_antibiotics"] = int(int(G.nodes[tgt].get("abx_class", 0)) > 0)
            transmission_events.append(chosen)
            new_state[tgt] = _state_from_resistance_label(str(chosen["type"]))
            G.nodes[tgt]["transmission_source"] = str(chosen["source"])
            G.nodes[tgt]["transmission_type"] = str(chosen["type"])
            G.nodes[tgt]["competing_exposure_count"] = int(len(successful_events))
            if new_state[tgt] == 2:
                G.nodes[tgt]["new_cr_acq_today"] = 1
                G.nodes[tgt]["new_trans_cr_today"] = 1
            else:
                G.nodes[tgt]["new_trans_cs_today"] = 1

    for n in nodes:
        st_before = int(new_state[n])

        if st_before in (1, 2) and _event_u01(params, "col_to_inf", d, str(n)) < float(params.p_col_to_inf):
            new_state[n] = 3 if st_before == 1 else 4
            if new_state[n] == 4 and st_before != 4:
                G.nodes[n]["new_ir_inf_today"] = 1

            if G.nodes[n].get("role") == "patient" and int(getattr(params, "active_symptom_screening", 1)) == 1:
                if int(G.nodes[n].get("screened_today", 0)) != 1 and int(G.nodes[n].get("pending_test_days", 0)) <= 0:
                    G.nodes[n]["amr_state"] = int(new_state[n])
                    G.nodes[n]["screened_today"] = 1
                    _schedule_test(G.nodes[n], params, intervention=intervention, day=d)

            if G.nodes[n].get("role") == "staff":
                remove = (
                    int(params.staff_removal_mode) == 1 or
                    (int(params.staff_removal_mode) == 2 and int(new_state[n]) == 4) or
                    (int(params.staff_removal_mode) == 3 and _event_u01(params, "staff_removal", d, str(n)) < float(params.staff_removal_prob))
                )
                if remove:
                    staff_removed[n] = True
                    denom = max(1e-12, float(params.p_inf_clear))
                    staff_timer[n] = max(1, int(math.ceil(1.0 / denom)))

        st_mid = int(new_state[n])
        if st_mid in (3, 4) and _event_u01(params, "inf_clear", d, str(n)) < float(params.p_inf_clear):
            new_state[n] = 0
            if G.nodes[n].get("role") == "staff":
                staff_removed[n] = False
                staff_timer[n] = 0

        st_mid2 = int(new_state[n])
        if st_mid2 in (1, 2) and _event_u01(params, "col_clear", d, str(n)) < float(params.p_col_clear):
            new_state[n] = 0

    for n in nodes:
        G.nodes[n]["amr_state"] = int(new_state[n])

    for s, removed in list(staff_removed.items()):
        if not removed or s not in G.nodes:
            continue
        t = int(staff_timer.get(s, 0))
        if t > 0:
            staff_timer[s] = t - 1
        if staff_timer.get(s, 0) <= 0 and not is_infected(int(G.nodes[s].get("amr_state", 0))):
            staff_removed[s] = False
            staff_timer[s] = 0

    for n in nodes:
        if G.nodes[n].get("role") == "staff":
            G.nodes[n]["present_today"] = 0 if staff_removed.get(n, False) else 1

    new_cr_total = int(sum(int(G.nodes[n].get("new_cr_acq_today", 0)) for n in nodes))
    new_ir_total = int(sum(int(G.nodes[n].get("new_ir_inf_today", 0)) for n in nodes))
    G.graph["new_cr_acq_total"] = new_cr_total
    G.graph["new_ir_inf_total"] = new_ir_total
    G.graph["new_import_cr_total"] = int(sum(int(G.nodes[n].get("new_import_cr_today", 0)) for n in nodes))
    G.graph["new_import_cs_total"] = int(sum(int(G.nodes[n].get("new_import_cs_today", 0)) for n in nodes))
    G.graph["new_import_ir_total"] = int(sum(int(G.nodes[n].get("new_import_ir_today", 0)) for n in nodes))
    G.graph["new_import_is_total"] = int(sum(int(G.nodes[n].get("new_import_is_today", 0)) for n in nodes))
    G.graph["new_trans_cr_total"] = int(sum(int(G.nodes[n].get("new_trans_cr_today", 0)) for n in nodes))
    G.graph["new_trans_cs_total"] = int(sum(int(G.nodes[n].get("new_trans_cs_today", 0)) for n in nodes))
    G.graph["new_select_cr_total"] = int(sum(int(G.nodes[n].get("new_select_cr_today", 0)) for n in nodes))
    G.graph["transmission_events_json"] = json.dumps(transmission_events, sort_keys=True)
    G.graph["transmission_event_count"] = int(len(transmission_events))
    G.graph["competing_transmission_event_count"] = int(sum(1 for ev in transmission_events if int(ev.get("competing_exposure_count", 1)) > 1))

    counts = _count_states(G)
    G.graph["resistant_fraction"] = float(_resistant_fraction(counts))


# =============================================================================
# YAML export
# =============================================================================
def _yaml_escape_string(s: str) -> str:
    special = any(
        ch in s
        for ch in [
            ":", "#", "\n", "\r", "\t", "\"", "'",
            "{", "}", "[", "]", ",", "&", "*", "?", "|",
            ">", "!", "%", "@", "`",
        ]
    )
    if special or s.strip() != s or s == "" or s.lower() in {"null", "true", "false", "yes", "no"}:
        return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return s


def _to_yaml(obj: Any, indent: int = 0) -> str:
    sp = "  " * indent

    if obj is None:
        return "null"
    if isinstance(obj, bool):
        return "true" if obj else "false"
    if isinstance(obj, (int, float)):
        return repr(obj)
    if isinstance(obj, str):
        return _yaml_escape_string(obj)

    if isinstance(obj, (list, tuple)):
        if len(obj) == 0:
            return "[]"
        lines = []
        for item in obj:
            if isinstance(item, (dict, list, tuple)):
                lines.append(f"{sp}-")
                nested = _to_yaml(item, indent + 1)
                lines.append((sp + "  ") + nested.replace("\n", "\n" + sp + "  "))
            else:
                lines.append(f"{sp}- {_to_yaml(item, 0)}")
        return "\n".join(lines)

    if isinstance(obj, dict):
        if len(obj) == 0:
            return "{}"
        lines = []
        for k, v in obj.items():
            key = str(k)
            if isinstance(v, (dict, list, tuple)):
                lines.append(f"{sp}{key}:")
                nested = _to_yaml(v, indent + 1)
                lines.append(nested if "\n" in nested else ("  " * (indent + 1) + nested))
            else:
                lines.append(f"{sp}{key}: {_to_yaml(v, 0)}")
        return "\n".join(lines)

    return _yaml_escape_string(str(obj))


def write_run_metadata_yaml(output_dir: str, args: argparse.Namespace, params: Params) -> str:
    ensure_dir(output_dir)

    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": os.path.basename(__file__),
        "python": {"version": sys.version.replace("\n", " ")},
        "packages": {
            "numpy": getattr(np, "__version__", "unknown"),
            "networkx": getattr(nx, "__version__", "unknown"),
            "matplotlib": getattr(matplotlib, "__version__", "unknown"),
            "pillow": getattr(Image, "__version__", "not_imported") if Image is not None else "not_imported",
        },
        "random_seeds": {
            "seed": int(args.seed),
            "python_random_seeded": True,
            "numpy_random_seeded": True,
        },
        "cli_args": {
            "output_dir": str(args.output_dir),
            "num_regions": int(args.num_regions),
            "seed": int(args.seed),
            "num_days": int(args.num_days),
            "num_patients": int(args.num_patients),
            "num_staff": int(args.num_staff),
            "num_wards": int(args.num_wards),
            "staff_wards_per_staff": int(args.staff_wards_per_staff),
            "daily_discharge_frac": float(args.daily_discharge_frac),
            "daily_discharge_min_per_ward": int(args.daily_discharge_min_per_ward),
            "p_import_cs": float(args.p_import_cs) if args.p_import_cs is not None else None,
            "p_import_cr": float(args.p_import_cr) if args.p_import_cr is not None else None,
            "p_import_is": float(args.p_import_is) if args.p_import_is is not None else None,
            "p_import_ir": float(args.p_import_ir) if args.p_import_ir is not None else None,
            "p_admit_import_cs": float(args.p_admit_import_cs) if args.p_admit_import_cs is not None else None,
            "p_admit_import_cr": float(args.p_admit_import_cr) if args.p_admit_import_cr is not None else None,
            "p_admit_import_is": float(args.p_admit_import_is) if args.p_admit_import_is is not None else None,
            "p_admit_import_ir": float(args.p_admit_import_ir) if args.p_admit_import_ir is not None else None,
            "isolation_mult": float(args.isolation_mult) if args.isolation_mult is not None else None,
            "isolation_days": int(args.isolation_days) if args.isolation_days is not None else None,
            "admit_import_seasonality": str(args.admit_import_seasonality) if getattr(args, "admit_import_seasonality", None) is not None else None,
            "admit_import_amp": float(args.admit_import_amp) if getattr(args, "admit_import_amp", None) is not None else None,
            "admit_import_period_days": int(args.admit_import_period_days) if getattr(args, "admit_import_period_days", None) is not None else None,
            "admit_import_phase_day": int(args.admit_import_phase_day) if getattr(args, "admit_import_phase_day", None) is not None else None,
            "admit_import_pmax_cs": float(args.admit_import_pmax_cs) if getattr(args, "admit_import_pmax_cs", None) is not None else None,
            "admit_import_pmax_cr": float(args.admit_import_pmax_cr) if getattr(args, "admit_import_pmax_cr", None) is not None else None,
            "admit_import_high_start_day": int(args.admit_import_high_start_day) if getattr(args, "admit_import_high_start_day", None) is not None else None,
            "admit_import_high_end_day": int(args.admit_import_high_end_day) if getattr(args, "admit_import_high_end_day", None) is not None else None,
            "admit_import_high_mult": float(args.admit_import_high_mult) if getattr(args, "admit_import_high_mult", None) is not None else None,
            "admit_import_low_mult": float(args.admit_import_low_mult) if getattr(args, "admit_import_low_mult", None) is not None else None,
            "admit_import_shock_min_days": int(args.admit_import_shock_min_days) if getattr(args, "admit_import_shock_min_days", None) is not None else None,
            "admit_import_shock_max_days": int(args.admit_import_shock_max_days) if getattr(args, "admit_import_shock_max_days", None) is not None else None,
            "admit_import_shock_mult_min": float(args.admit_import_shock_mult_min) if getattr(args, "admit_import_shock_mult_min", None) is not None else None,
            "admit_import_shock_mult_max": float(args.admit_import_shock_mult_max) if getattr(args, "admit_import_shock_mult_max", None) is not None else None,
            "screen_every_k_days": int(getattr(args, "screen_every_k_days", 0) or 0),
            "screen_on_admission": int(getattr(args, "screen_on_admission", 0) or 0),
            "screen_result_delay_days": int(getattr(args, "screen_result_delay_days", 0) or 0),
            "persist_observations": int(getattr(args, "persist_observations", 0) or 0),
            "export_yaml": bool(args.export_yaml),
            "export_gif": bool(args.export_gif),
            "gif_fps": float(args.gif_fps),
            "gif_max_edges_draw": int(args.gif_max_edges_draw),
            "gif_layout_seed": int(args.gif_layout_seed),
            "superspreader_staff": str(args.superspreader_staff) if args.superspreader_staff else "",
            "superspreader_state": str(args.superspreader_state) if getattr(args, "superspreader_state", None) else "",
            "superspreader_start_day": int(args.superspreader_start_day),
            "superspreader_end_day": int(args.superspreader_end_day),
            "superspreader_patient_frac_mult": float(args.superspreader_patient_frac_mult),
            "superspreader_patient_min_add": int(args.superspreader_patient_min_add),
            "superspreader_staff_contacts": int(args.superspreader_staff_contacts),
            "superspreader_edge_weight_mult": float(args.superspreader_edge_weight_mult),
        },
        "model_params": asdict(params),
    }

    out_path = os.path.join(output_dir, "run_metadata.yaml")
    if os.path.exists(out_path):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = os.path.join(output_dir, f"run_metadata_{stamp}.yaml")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_to_yaml(meta, indent=0) + "\n")

    return out_path


# =============================================================================
# GIF rendering
# =============================================================================
def _setup_mpl_pub_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "white",
        "text.color": "#111111",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "font.size": 10,
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _state_to_color(st: int) -> str:
    st = int(st)
    return STATE_STYLE.get(st, STATE_STYLE[0])["color"]


def _grid_centers(num_wards: int, spacing: float) -> Dict[int, np.ndarray]:
    W = int(num_wards)
    if W <= 0:
        return {0: np.array([0.0, 0.0], dtype=float)}

    cols = int(math.ceil(math.sqrt(W)))
    centers: Dict[int, np.ndarray] = {}
    for w in range(W):
        r = w // cols
        c = w % cols
        centers[w] = np.array([float(c) * spacing, float(-r) * spacing], dtype=float)
    return centers


def _normalize_layout(pos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    if not pos:
        return pos
    keys = list(pos.keys())
    P = np.vstack([pos[k] for k in keys])
    P = P - P.mean(axis=0, keepdims=True)
    scale = float(np.max(np.sqrt((P ** 2).sum(axis=1))))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    P = P / scale
    return {k: P[i] for i, k in enumerate(keys)}


def _compute_block_layout(
    G_day0: nx.DiGraph,
    patients: List[str],
    staff: List[str],
    ward_of: Dict[str, int],
    staff_wards: Dict[str, List[int]],
    num_wards: int,
    layout_seed: int,
    ward_spacing: float = 6.5,
    ward_pad: float = 1.2,
    staff_jitter: float = 0.35,
) -> Tuple[Dict[str, np.ndarray], Dict[int, Dict[str, Any]]]:
    W = int(num_wards)
    centers = _grid_centers(W, spacing=float(ward_spacing))

    G0u = G_day0.to_undirected()

    pos: Dict[str, np.ndarray] = {}
    ward_meta: Dict[int, Dict[str, Any]] = {}

    rng = np.random.RandomState(int(layout_seed))

    for w in range(W):
        ward_pat = [p for p in patients if int(ward_of.get(p, -1)) == w]
        if len(ward_pat) == 0:
            c = centers[w]
            ward_meta[w] = {"center": c, "box": (float(c[0]), float(c[0]), float(c[1]), float(c[1]))}
            continue

        Hw = G0u.subgraph(ward_pat).copy()

        if Hw.number_of_edges() == 0 and len(ward_pat) > 1:
            for i in range(len(ward_pat) - 1):
                Hw.add_edge(ward_pat[i], ward_pat[i + 1])

        ward_pos = nx.spring_layout(Hw, seed=int(layout_seed) + 101 * w, iterations=80)
        ward_pos = _normalize_layout(ward_pos)

        c = centers[w]
        for n, xy in ward_pos.items():
            pos[n] = np.array([float(xy[0]) + float(c[0]), float(xy[1]) + float(c[1])], dtype=float)

        P = np.vstack([pos[p] for p in ward_pat])
        xmin, xmax = float(P[:, 0].min()), float(P[:, 0].max())
        ymin, ymax = float(P[:, 1].min()), float(P[:, 1].max())
        ward_meta[w] = {
            "center": c,
            "box": (xmin - ward_pad, xmax + ward_pad, ymin - ward_pad, ymax + ward_pad),
        }

    for s in staff:
        wards = staff_wards.get(s, [int(ward_of.get(s, 0))])
        wards = [w for w in sorted(set(int(w) for w in wards)) if 0 <= int(w) < W]
        if len(wards) == 0:
            wards = [int(ward_of.get(s, 0))] if W > 0 else [0]

        C = np.vstack([centers[w] for w in wards]).mean(axis=0)

        jx = (rng.rand() - 0.5) * 2.0 * float(staff_jitter)
        jy = (rng.rand() - 0.5) * 2.0 * float(staff_jitter)
        pos[s] = np.array([float(C[0]) + float(jx), float(C[1]) + float(jy)], dtype=float)

    return pos, ward_meta


def _draw_ward_blocks(ax, ward_meta: Dict[int, Dict[str, Any]]) -> None:
    for w, m in ward_meta.items():
        xmin, xmax, ymin, ymax = m["box"]
        cx, cy = float(m["center"][0]), float(m["center"][1])
        width = float(xmax - xmin)
        height = float(ymax - ymin)

        patch = FancyBboxPatch(
            (float(xmin), float(ymin)),
            width,
            height,
            boxstyle="round,pad=0.02,rounding_size=0.25",
            linewidth=1.1,
            edgecolor="#1F77B4",
            facecolor="#1F77B4",
            alpha=0.06,
        )
        ax.add_patch(patch)

        ax.text(
            cx, cy,
            f"W{int(w)}",
            ha="center", va="center",
            fontsize=10,
            color="#1F77B4",
            alpha=0.85,
            fontweight="bold",
        )


def _draw_staff_assignment_lines(
    ax,
    staff: List[str],
    pos: Dict[str, np.ndarray],
    staff_wards: Dict[str, List[int]],
    ward_meta: Dict[int, Dict[str, Any]],
) -> None:
    for s in staff:
        if s not in pos:
            continue
        wards = staff_wards.get(s, [])
        if not wards:
            continue
        sx, sy = float(pos[s][0]), float(pos[s][1])
        for w in wards:
            if int(w) not in ward_meta:
                continue
            c = ward_meta[int(w)]["center"]
            ax.plot(
                [sx, float(c[0])],
                [sy, float(c[1])],
                linewidth=0.55,
                alpha=0.18,
                color="#111111",
                zorder=0,
            )


def _ensure_layout_positions(
    G: nx.DiGraph,
    pos: Dict[str, np.ndarray],
    ward_meta: Dict[int, Dict[str, Any]],
    ward_of: Dict[str, int],
    staff_wards: Dict[str, List[int]],
    num_wards: int,
    rng: np.random.RandomState,
    staff_jitter: float = 0.35,
    patient_jitter: float = 0.55,
) -> None:
    """Ensure `pos` contains coordinates for every node currently in `G`."""
    W = max(1, int(num_wards))
    for n, attrs in G.nodes(data=True):
        key = str(n)
        if key in pos:
            continue

        role = attrs.get("role", "patient")
        home = int(ward_of.get(key, attrs.get("ward_id", 0)))
        home = 0 if home < 0 else (W - 1 if home >= W else home)

        if role == "staff":
            wards = staff_wards.get(key, None)
            if wards is None:
                wards_str = str(attrs.get("ward_ids", "")).strip()
                if wards_str != "":
                    try:
                        wards = [int(x) for x in wards_str.split(",") if str(x).strip() != ""]
                    except Exception:
                        wards = [home]
                else:
                    wards = [home]
            wards = [w for w in sorted(set(int(w) for w in wards)) if 0 <= int(w) < W]
            if len(wards) == 0:
                wards = [home]

            centers = []
            for w in wards:
                if int(w) in ward_meta:
                    centers.append(np.array(ward_meta[int(w)]["center"], dtype=float))
            if len(centers) == 0:
                centers = [np.array([0.0, 0.0], dtype=float)]
            C = np.vstack(centers).mean(axis=0)

            jx = (rng.rand() - 0.5) * 2.0 * float(staff_jitter)
            jy = (rng.rand() - 0.5) * 2.0 * float(staff_jitter)
            pos[key] = np.array([float(C[0]) + float(jx), float(C[1]) + float(jy)], dtype=float)
        else:
            if int(home) in ward_meta:
                C = np.array(ward_meta[int(home)]["center"], dtype=float)
            else:
                C = np.array([0.0, 0.0], dtype=float)

            jx = (rng.rand() - 0.5) * 2.0 * float(patient_jitter)
            jy = (rng.rand() - 0.5) * 2.0 * float(patient_jitter)
            pos[key] = np.array([float(C[0]) + float(jx), float(C[1]) + float(jy)], dtype=float)


def _graph_to_frame(
    G: nx.DiGraph,
    pos: Dict[str, np.ndarray],
    ward_meta: Dict[int, Dict[str, Any]],
    staff_wards: Dict[str, List[int]],
    day: int,
    region: int,
    max_edges_draw: int,
    edge_sample_rng: np.random.RandomState,
) -> "Image.Image":
    if Image is None:
        raise RuntimeError("PIL (Pillow) is required for GIF export but could not be imported.")

    counts = _count_states(G)
    res_frac = _resistant_fraction(counts)

    fig = plt.figure(figsize=GIF_FIGSIZE, dpi=GIF_DPI)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_axis_off()

    _draw_ward_blocks(ax, ward_meta)

    patients = [n for n, a in G.nodes(data=True) if a.get("role") == "patient"]
    staff = [n for n, a in G.nodes(data=True) if a.get("role") == "staff"]

    try:
        W_eff = (max(int(w) for w in ward_meta.keys()) + 1) if len(ward_meta) > 0 else 1
    except Exception:
        W_eff = 1
    layout_rng = np.random.RandomState(int(day) + 1000 * int(region) + 12345)
    _ensure_layout_positions(
        G=G,
        pos=pos,
        ward_meta=ward_meta,
        ward_of={},
        staff_wards=staff_wards,
        num_wards=int(W_eff),
        rng=layout_rng,
    )

    critical_states = {3, 4}
    critical_mult = 1.70

    patient_sizes = []
    for n in patients:
        st = int(G.nodes[n].get("amr_state", 0))
        base = float(ROLE_STYLE["patient"]["size"])
        patient_sizes.append(base * critical_mult if st in critical_states else base)

    staff_sizes = []
    for n in staff:
        st = int(G.nodes[n].get("amr_state", 0))
        base = float(ROLE_STYLE["staff"]["size"])
        staff_sizes.append(base * critical_mult if st in critical_states else base)

    size_by_node: Dict[str, float] = {}
    for n, s in zip(patients, patient_sizes):
        size_by_node[n] = float(s)
    for n, s in zip(staff, staff_sizes):
        size_by_node[n] = float(s)

    _draw_staff_assignment_lines(ax, staff=staff, pos=pos, staff_wards=staff_wards, ward_meta=ward_meta)

    edges = list(G.edges())
    if len(edges) > int(max_edges_draw):
        idx = edge_sample_rng.choice(len(edges), size=int(max_edges_draw), replace=False)
        edges_draw = [edges[i] for i in idx]
    else:
        edges_draw = edges

    if len(edges_draw) > 0:
        edges_draw = [(u, v) for (u, v) in edges_draw if (str(u) in pos and str(v) in pos)]

    nx.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=edges_draw,
        ax=ax,
        arrows=False,
        alpha=EDGE_ALPHA,
        width=EDGE_WIDTH,
        edge_color=EDGE_COLOR,
    )

    patient_colors = [_state_to_color(int(G.nodes[n].get("amr_state", 0))) for n in patients]
    staff_colors = [_state_to_color(int(G.nodes[n].get("amr_state", 0))) for n in staff]

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=patients,
        node_color=patient_colors,
        node_size=patient_sizes,
        linewidths=ROLE_STYLE["patient"]["lw"],
        edgecolors="#000000",
        ax=ax,
        node_shape=ROLE_STYLE["patient"]["shape"],
        alpha=0.98,
    )

    nx.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=staff,
        node_color=staff_colors,
        node_size=staff_sizes,
        linewidths=ROLE_STYLE["staff"]["lw"],
        edgecolors="#000000",
        ax=ax,
        node_shape=ROLE_STYLE["staff"]["shape"],
        alpha=0.98,
    )

    iso_nodes = [n for n, a in G.nodes(data=True) if int(a.get("is_isolated", 0)) == 1]
    if len(iso_nodes) > 0:
        iso_sizes: List[float] = []
        for n in iso_nodes:
            base = float(size_by_node.get(n, ROLE_STYLE["patient"]["size"]))
            iso_sizes.append(base * 1.45)

        nx.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=iso_nodes,
            node_color="none",
            node_size=iso_sizes,
            linewidths=ISO_RING_LW,
            edgecolors=ISO_RING_COLOR,
            ax=ax,
            node_shape="o",
            alpha=0.90,
        )

    fig.text(
        0.02, 0.975,
        f"Region {region} | Day {day}",
        ha="left", va="top",
        fontsize=12, fontweight="bold",
        color="#111111",
    )
    fig.text(
        0.02, 0.935,
        f"U={counts['u']}  CS={counts['cs']}  CR={counts['cr']}  IS={counts['is']}  IR={counts['ir']}  "
        f"ResFrac={res_frac:.3f}",
        ha="left", va="top",
        fontsize=10,
        color="#111111",
    )

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Patient",
               markerfacecolor="#FFFFFF", markeredgecolor="#111111", markersize=7),
        Line2D([0], [0], marker="s", color="w", label="Staff",
               markerfacecolor="#FFFFFF", markeredgecolor="#111111", markersize=8),
    ]

    for st in sorted(STATE_STYLE.keys()):
        label = str(STATE_STYLE[st].get("label", AMR_STATE_STR.get(int(st), str(st))))
        col = str(STATE_STYLE[st].get("color", STATE_STYLE[0]["color"]))
        legend_elements.append(
            Line2D([0], [0], marker="o", color="w", label=label,
                   markerfacecolor=col, markeredgecolor=col, markersize=7)
        )

    legend_elements.extend([
        Line2D([0], [0], marker="o", color="w", label="Isolated",
               markerfacecolor="none", markeredgecolor=ISO_RING_COLOR, markersize=8),
        Line2D([0], [0], color="#111111", lw=1.0, alpha=0.35, label="Staff→ward\nassignment"),
    ])

    fig.subplots_adjust(left=0.02, right=0.74, top=0.84, bottom=0.02)

    leg = ax.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=9,
        frameon=True,
        borderpad=0.6,
        handletextpad=0.6,
        labelspacing=0.5,
    )
    leg.get_frame().set_alpha(0.92)
    leg.get_frame().set_edgecolor("#DDDDDD")

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


def _save_gif(frames: List["Image.Image"], out_path: str, fps: float) -> None:
    if Image is None:
        raise RuntimeError("PIL (Pillow) is required for GIF export but could not be imported.")
    if len(frames) == 0:
        return

    duration_ms = int(round(1000.0 / max(0.1, float(fps))))

    palette = frames[0].convert("P", palette=Image.Palette.ADAPTIVE, colors=256, dither=Image.Dither.FLOYDSTEINBERG)
    q_frames = [palette]
    for fr in frames[1:]:
        q_frames.append(fr.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG))

    q_frames[0].save(
        out_path,
        save_all=True,
        append_images=q_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


# =============================================================================
# MAIN
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_regions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_days", type=int, default=90)
    parser.add_argument("--num_patients", type=int, default=300)
    parser.add_argument("--num_staff", type=int, default=120)
    parser.add_argument("--num_wards", type=int, default=8)

    parser.add_argument("--admit_import_seasonality", type=str, default=None,
                        help="Seasonality mode for admission importation: none|sinusoid|piecewise|shock.")
    parser.add_argument("--admit_import_amp", type=float, default=None,
                        help="Sinusoid amplitude in [0,1). Example 0.5 gives ±50%%.")
    parser.add_argument("--admit_import_period_days", type=int, default=None,
                        help="Season period in days (365 yearly, 7 weekly, etc).")
    parser.add_argument("--admit_import_phase_day", type=int, default=None,
                        help="Phase shift.")
    parser.add_argument("--admit_import_pmax_cs", type=float, default=None,
                        help="Cap for seasonal p_admit_import_cs after scaling.")
    parser.add_argument("--admit_import_pmax_cr", type=float, default=None,
                        help="Cap for seasonal p_admit_import_cr after scaling.")
    parser.add_argument("--admit_import_pmax_is", type=float, default=None,
                        help="Cap for seasonal p_admit_import_is after scaling.")
    parser.add_argument("--admit_import_pmax_ir", type=float, default=None,
                        help="Cap for seasonal p_admit_import_ir after scaling.")

    parser.add_argument("--admit_import_high_start_day", type=int, default=None,
                        help="Piecewise: high-season start day within period (1..period).")
    parser.add_argument("--admit_import_high_end_day", type=int, default=None,
                        help="Piecewise: high-season end day within period (1..period).")
    parser.add_argument("--admit_import_high_mult", type=float, default=None,
                        help="Piecewise: multiplier in high season.")
    parser.add_argument("--admit_import_low_mult", type=float, default=None,
                        help="Piecewise: multiplier outside high season.")

    parser.add_argument("--admit_import_shock_min_days", type=int, default=None,
                        help="Shock: minimum duration in days.")
    parser.add_argument("--admit_import_shock_max_days", type=int, default=None,
                        help="Shock: maximum duration in days.")
    parser.add_argument("--admit_import_shock_mult_min", type=float, default=None,
                        help="Shock: minimum multiplier.")
    parser.add_argument("--admit_import_shock_mult_max", type=float, default=None,
                        help="Shock: maximum multiplier.")

    parser.add_argument(
        "--p_import_cs",
        type=float,
        default=None,
        help="Day-0 probability/weight that an initial patient is CS.",
    )
    parser.add_argument(
        "--p_import_cr",
        type=float,
        default=None,
        help="Day-0 probability/weight that an initial patient is CR.",
    )
    parser.add_argument(
        "--p_import_is",
        type=float,
        default=None,
        help="Day-0 probability/weight that an initial patient is IS; default preserves previous behaviour at 0.",
    )
    parser.add_argument(
        "--p_import_ir",
        type=float,
        default=None,
        help="Day-0 probability/weight that an initial patient is IR; default preserves previous behaviour at 0.",
    )

    parser.add_argument(
        "--daily_discharge_frac",
        type=float,
        default=0.0,
        help="Daily discharge fraction per ward.",
    )
    parser.add_argument(
        "--daily_discharge_min_per_ward",
        type=int,
        default=0,
        help="Minimum discharges per ward per day when turnover is enabled.",
    )
    parser.add_argument(
        "--p_admit_import_cs",
        type=float,
        default=None,
        help="Probability an admitted patient is CS on admission.",
    )
    parser.add_argument(
        "--p_admit_import_cr",
        type=float,
        default=None,
        help="Probability an admitted patient is CR on admission.",
    )
    parser.add_argument(
        "--p_admit_import_is",
        type=float,
        default=None,
        help="Probability an admitted patient is IS on admission; default preserves previous behaviour at 0.",
    )
    parser.add_argument(
        "--p_admit_import_ir",
        type=float,
        default=None,
        help="Probability an admitted patient is IR on admission; default preserves previous behaviour at 0.",
    )


    parser.add_argument("--beta_res_mult", type=float, default=None, help="Global resistant-source infectivity multiplier eta_R; default is 1.0.")
    parser.add_argument("--beta_state_mult_cs", type=float, default=None, help="Source-state multiplier for CS transmission.")
    parser.add_argument("--beta_state_mult_cr", type=float, default=None, help="Source-state multiplier for CR transmission.")
    parser.add_argument("--beta_state_mult_is", type=float, default=None, help="Source-state multiplier for IS transmission.")
    parser.add_argument("--beta_state_mult_ir", type=float, default=None, help="Source-state multiplier for IR transmission.")
    parser.add_argument("--crossward_hand_hygiene_mult", type=float, default=None, help="Leaky hand-hygiene multiplier for staff-patient contacts outside the staff home ward.")
    parser.add_argument("--crossward_hand_hygiene_enabled", type=int, default=None, help="If 1, apply leaky hand hygiene to cross-ward staff contacts.")
    parser.add_argument("--passive_screen_capacity_per_week", type=int, default=None, help="Passive screening capacity per screening day/week; 0 screens all eligible.")
    parser.add_argument("--active_symptom_screening", type=int, default=None, help="If 1, actively screen patients when they progress from C to I.")
    parser.add_argument("--admission_screen_symptomatic_only", type=int, default=None, help="If 1, admission screening is restricted to symptomatic IS/IR patients.")
    parser.add_argument("--isolation_trigger_policy", type=str, default=None, choices=["resistant_only", "all_positives"], help="Positive-test isolation policy.")
    parser.add_argument("--no_abx_sensitive_selection_weight", type=float, default=None, help="Competing-exposure weight favouring sensitive acquisition when target is not on antibiotics.")
    parser.add_argument("--abx_resistant_selection_weight", type=float, default=None, help="Competing-exposure weight favouring resistant acquisition when target is on antibiotics.")

    parser.add_argument(
        "--screen_every_k_days",
        type=int,
        default=None,
        help="If >0, perform screening every k days starting at day 1.",
    )
    parser.add_argument(
        "--screen_on_admission",
        type=int,
        default=None,
        help="If 1, schedule screening for newly admitted patients on their admission day.",
    )
    parser.add_argument(
        "--screen_result_delay_days",
        type=int,
        default=None,
        help="Delay (days) between sample and available result (0 means immediate).",
    )
    parser.add_argument(
        "--persist_observations",
        type=int,
        default=None,
        help="If 1, keep latest observed status across days; if 0, preserve legacy behavior.",
    )

    # PATCHED: expose isolation controls on the CLI so causal dataset generation can pass them through
    parser.add_argument(
        "--isolation_mult",
        type=float,
        default=None,
        help="Isolation transmission multiplier applied to isolated positive nodes (default from Params).",
    )
    parser.add_argument(
        "--isolation_days",
        type=int,
        default=None,
        help="Isolation duration in days after a positive screen (default from Params).",
    )

    parser.add_argument(
        "--staff_wards_per_staff",
        type=int,
        default=2,
        help="Number of wards each staff member is assigned to.",
    )

    parser.add_argument("--superspreader_staff", type=str, default=None, help="Staff node ID to act as superspreader.")
    parser.add_argument(
        "--superspreader_state",
        type=str,
        default=None,
        help="AMR state to force for superspreader during active superspreader window (U/CS/CR/IS/IR or 0..4).",
    )
    parser.add_argument("--superspreader_start_day", type=int, default=1, help="First day to apply superspreader injection.")
    parser.add_argument("--superspreader_end_day", type=int, default=9999, help="Last day to apply superspreader injection.")
    parser.add_argument("--superspreader_patient_frac_mult", type=float, default=3.0, help="Multiplier on staff->patient contact fraction for superspreader.")
    parser.add_argument("--superspreader_patient_min_add", type=int, default=10, help="Additive increase in patient contacts per ward for superspreader.")
    parser.add_argument("--superspreader_staff_contacts", type=int, default=30, help="Additional staff contacts per day for superspreader.")
    parser.add_argument("--superspreader_edge_weight_mult", type=float, default=1.5, help="Multiplier on edge weights incident to superspreader.")

    parser.add_argument(
        "--causal_mode",
        type=int,
        default=0,
        help="If 1, enable Stage 1 causal metadata/intervention mode.",
    )
    parser.add_argument(
        "--causal_pair_id",
        type=str,
        default="",
        help="Pair identifier shared by factual/counterfactual runs.",
    )
    parser.add_argument(
        "--causal_role",
        type=str,
        default="factual",
        help="Causal role: factual or counterfactual.",
    )
    parser.add_argument(
        "--causal_shared_noise_seed",
        type=int,
        default=0,
        help="Shared-noise seed placeholder for paired factual/counterfactual runs.",
    )
    parser.add_argument(
        "--causal_intervention_json",
        type=str,
        default="",
        help="JSON-encoded Stage 1 causal intervention specification.",
    )

    parser.add_argument(
        "--export_yaml",
        action="store_true",
        help="Export run metadata to YAML in output_dir.",
    )

    parser.add_argument(
        "--no_export_gif",
        action="store_true",
        help="Disable GIF export (GIF is exported by default).",
    )
    parser.add_argument(
        "--gif_fps",
        type=float,
        default=5.0,
        help="GIF frames per second.",
    )
    parser.add_argument(
        "--gif_max_edges_draw",
        type=int,
        default=6000,
        help="Maximum number of edges drawn per frame.",
    )
    parser.add_argument(
        "--gif_layout_seed",
        type=int,
        default=123,
        help="Seed for ward-blocked layout so frames are stable.",
    )

    args = parser.parse_args()
    args.export_gif = not bool(args.no_export_gif)

    set_seed(args.seed)
    ensure_dir(args.output_dir)

    _setup_mpl_pub_style()
    params = Params()
    params.runtime_seed_base = int(args.seed)
    params.runtime_region_index = 0

    params.daily_discharge_frac = float(args.daily_discharge_frac)
    params.daily_discharge_min_per_ward = int(args.daily_discharge_min_per_ward)
    if args.p_import_cs is not None:
        params.p_import_cs = float(args.p_import_cs)
    if args.p_import_cr is not None:
        params.p_import_cr = float(args.p_import_cr)
    if args.p_import_is is not None:
        params.p_import_is = float(args.p_import_is)
    if args.p_import_ir is not None:
        params.p_import_ir = float(args.p_import_ir)
    if args.p_admit_import_cs is not None:
        params.p_admit_import_cs = float(args.p_admit_import_cs)
    if args.p_admit_import_cr is not None:
        params.p_admit_import_cr = float(args.p_admit_import_cr)
    if args.p_admit_import_is is not None:
        params.p_admit_import_is = float(args.p_admit_import_is)
    if args.p_admit_import_ir is not None:
        params.p_admit_import_ir = float(args.p_admit_import_ir)


    for _name in (
        "beta_res_mult",
        "beta_state_mult_cs",
        "beta_state_mult_cr",
        "beta_state_mult_is",
        "beta_state_mult_ir",
        "crossward_hand_hygiene_mult",
        "crossward_hand_hygiene_enabled",
        "passive_screen_capacity_per_week",
        "active_symptom_screening",
        "admission_screen_symptomatic_only",
        "no_abx_sensitive_selection_weight",
        "abx_resistant_selection_weight",
    ):
        _value = getattr(args, _name, None)
        if _value is not None:
            setattr(params, _name, _value)
    if args.isolation_trigger_policy is not None:
        params.isolation_trigger_policy = str(args.isolation_trigger_policy)

    if args.screen_every_k_days is not None:
        params.screen_every_k_days = int(args.screen_every_k_days)
    if args.screen_on_admission is not None:
        params.screen_on_admission = int(args.screen_on_admission)
    if args.screen_result_delay_days is not None:
        params.screen_result_delay_days = int(args.screen_result_delay_days)
    if args.persist_observations is not None:
        params.persist_observations = int(args.persist_observations)

    # PATCHED: wire CLI isolation args into Params
    if args.isolation_mult is not None:
        params.isolation_mult = float(args.isolation_mult)
    if args.isolation_days is not None:
        params.isolation_days = int(args.isolation_days)

    if args.admit_import_seasonality is not None:
        params.admit_import_seasonality = str(args.admit_import_seasonality)
    if args.admit_import_amp is not None:
        params.admit_import_amp = float(args.admit_import_amp)
    if args.admit_import_period_days is not None:
        params.admit_import_period_days = int(args.admit_import_period_days)
    if args.admit_import_phase_day is not None:
        params.admit_import_phase_day = int(args.admit_import_phase_day)
    if args.admit_import_pmax_cs is not None:
        params.admit_import_pmax_cs = float(args.admit_import_pmax_cs)
    if args.admit_import_pmax_cr is not None:
        params.admit_import_pmax_cr = float(args.admit_import_pmax_cr)

    if args.admit_import_high_start_day is not None:
        params.admit_import_high_start_day = int(args.admit_import_high_start_day)
    if args.admit_import_high_end_day is not None:
        params.admit_import_high_end_day = int(args.admit_import_high_end_day)
    if args.admit_import_high_mult is not None:
        params.admit_import_high_mult = float(args.admit_import_high_mult)
    if args.admit_import_low_mult is not None:
        params.admit_import_low_mult = float(args.admit_import_low_mult)

    if args.admit_import_shock_min_days is not None:
        params.admit_import_shock_min_days = int(args.admit_import_shock_min_days)
    if args.admit_import_shock_max_days is not None:
        params.admit_import_shock_max_days = int(args.admit_import_shock_max_days)
    if args.admit_import_shock_mult_min is not None:
        params.admit_import_shock_mult_min = float(args.admit_import_shock_mult_min)
    if args.admit_import_shock_mult_max is not None:
        params.admit_import_shock_mult_max = float(args.admit_import_shock_mult_max)

    if args.export_gif and Image is None:
        raise RuntimeError(
            "GIF export requested but Pillow is not installed/importable. "
            "Install it (pip install pillow) or run with --no_export_gif."
        )

    causal_ctx: Optional[CounterfactualContext] = None
    causal_intervention = CausalInterventionSpec()
    if int(args.causal_mode) == 1:
        causal_intervention = _load_causal_intervention_json(args.causal_intervention_json)
        causal_ctx = CounterfactualContext(
            pair_id=str(args.causal_pair_id).strip(),
            role=_causal_role_normalized(args.causal_role),
            shared_noise_seed=int(args.causal_shared_noise_seed),
            intervention=causal_intervention,
        )
        params.causal_mode = 1
        params.causal_pair_id = str(causal_ctx.pair_id)
        params.causal_role = str(causal_ctx.role)
        params.causal_shared_noise_seed = int(causal_ctx.shared_noise_seed)
        params.causal_intervention_name = str(causal_intervention.name)
        params.causal_intervention_target_type = str(causal_intervention.target_type)
        params.causal_intervention_target_id = str(causal_intervention.target_id)

    if args.export_yaml:
        meta_path = write_run_metadata_yaml(args.output_dir, args, params)
        print(f"DT_SIM_META yaml={meta_path}", flush=True)

    frames: List["Image.Image"] = []

    ss_staff = str(args.superspreader_staff).strip() if args.superspreader_staff else ""
    ss_start = int(args.superspreader_start_day)
    ss_end = int(args.superspreader_end_day)
    ss_state_code = parse_amr_state(getattr(args, "superspreader_state", None))
    ss_state_str = str(args.superspreader_state).strip() if args.superspreader_state else ""

    split_label = _infer_split_label(args.output_dir)

    for r in range(int(args.num_regions)):
        region_seed = int(args.seed) + 1000 * int(r)
        params.runtime_seed_base = int(region_seed)
        params.runtime_region_index = int(r)
        set_seed(region_seed)

        if str(getattr(params, "admit_import_seasonality", "none")).strip().lower() == "shock":
            shock_rng = np.random.RandomState(int(region_seed) + 777777)

            sim_days = int(args.num_days)
            min_dur = max(1, int(getattr(params, "admit_import_shock_min_days", 7)))
            max_dur = max(1, int(getattr(params, "admit_import_shock_max_days", 30)))
            max_dur = min(max_dur, sim_days)
            min_dur = min(min_dur, max_dur)

            dur = int(shock_rng.randint(min_dur, max_dur + 1)) if max_dur >= min_dur else int(min_dur)
            latest_start = max(1, sim_days - dur + 1)
            start = int(shock_rng.randint(1, latest_start + 1)) if latest_start >= 1 else 1

            mmn = float(getattr(params, "admit_import_shock_mult_min", 1.5))
            mmx = float(getattr(params, "admit_import_shock_mult_max", 3.0))
            if mmx < mmn:
                mmn, mmx = mmx, mmn
            mmn = max(0.0, mmn)
            mmx = max(0.0, mmx)
            mult = float(shock_rng.uniform(mmn, mmx)) if mmx > mmn else float(mmn)
            mult = 1.0 if mult <= 0.0 else mult

            params.admit_import_shock_start_day = int(start)
            params.admit_import_shock_duration_days = int(dur)
            params.admit_import_shock_mult = float(mult)

        patients, staff, ward_of, staff_wards = build_population(
            num_patients=args.num_patients,
            num_staff=args.num_staff,
            num_wards=args.num_wards,
            staff_wards_per_staff=args.staff_wards_per_staff,
            seed=region_seed,
        )

        staff_removed: Dict[str, bool] = {s: False for s in staff}
        staff_timer: Dict[str, int] = {s: 0 for s in staff}
        base_staff_wards, base_staff_removed = _apply_runtime_staff_intervention(
            staff=staff,
            ward_of=ward_of,
            staff_wards=staff_wards,
            staff_removed=staff_removed,
            intervention=causal_intervention if int(args.causal_mode) == 1 else None,
            day=0,
        )
        effective_staff_wards = {str(k): list(v) for k, v in base_staff_wards.items()}
        effective_staff_removed = dict(base_staff_removed)

        evo_days: List[int] = []
        evo_u: List[int] = []
        evo_cs: List[int] = []
        evo_cr: List[int] = []
        evo_is: List[int] = []
        evo_ir: List[int] = []

        effective_staff_wards, effective_staff_removed = _apply_runtime_staff_intervention(
            staff=staff,
            ward_of=ward_of,
            staff_wards=base_staff_wards,
            staff_removed=staff_removed,
            intervention=causal_intervention if int(args.causal_mode) == 1 else None,
            day=0,
        )

        G_day = sample_contacts(
            patients=patients,
            staff=staff,
            ward_of=ward_of,
            staff_wards=effective_staff_wards,
            num_wards=args.num_wards,
            staff_removed=effective_staff_removed,
            superspreader_staff=ss_staff if ss_staff else None,
            superspreader_active=(ss_staff != "" and ss_start <= 0 <= ss_end),
            superspreader_patient_frac_mult=args.superspreader_patient_frac_mult,
            superspreader_patient_min_add=args.superspreader_patient_min_add,
            superspreader_staff_contacts=args.superspreader_staff_contacts,
            superspreader_edge_weight_mult=args.superspreader_edge_weight_mult,
            intervention=causal_intervention if int(args.causal_mode) == 1 else None,
            day=0,
            params=params,
        )
        init_states(G_day, patients, staff, ward_of, effective_staff_wards, params)
        G_day.graph["day"] = 0
        G_day.graph["region"] = int(r)
        _apply_graph_causal_metadata(G_day, causal_ctx)
        _attach_operational_graph_metadata(G_day, params, causal_intervention if int(args.causal_mode) == 1 else None, day=0)

        next_patient_id = int(args.num_patients)
        turnover_rng = np.random.RandomState(region_seed + 424242)

        if ss_staff != "" and ss_state_code is not None and int(ss_start) <= 0 <= int(ss_end):
            if ss_staff not in G_day.nodes:
                raise ValueError(
                    f"Superspreader staff '{ss_staff}' not found in graph nodes. "
                    "Expected staff IDs like s0..s(n-1)."
                )
            G_day.nodes[ss_staff]["amr_state"] = int(ss_state_code)
            counts0 = _count_states(G_day)
            G_day.graph["resistant_fraction"] = float(_resistant_fraction(counts0))

        G_day.graph["superspreader_staff"] = ss_staff
        G_day.graph["superspreader_state"] = ss_state_str
        G_day.graph["superspreader_active"] = int(ss_staff != "" and 0 >= ss_start and 0 <= ss_end)
        G_day.graph["superspreader_params_json"] = json.dumps({
            "state": ss_state_str,
            "start_day": ss_start,
            "end_day": ss_end,
            "patient_frac_mult": float(args.superspreader_patient_frac_mult),
            "patient_min_add": int(args.superspreader_patient_min_add),
            "staff_contacts": int(args.superspreader_staff_contacts),
            "edge_weight_mult": float(args.superspreader_edge_weight_mult),
        }, sort_keys=True)

        if str(getattr(params, "admit_import_seasonality", "none")).strip().lower() == "shock":
            G_day.graph["admit_import_shock_json"] = json.dumps({
                "start_day": int(getattr(params, "admit_import_shock_start_day", 0)),
                "duration_days": int(getattr(params, "admit_import_shock_duration_days", 0)),
                "mult": float(getattr(params, "admit_import_shock_mult", 1.0)),
            }, sort_keys=True)

        c0 = _count_states(G_day)
        evo_days.append(0)
        evo_u.append(int(c0["u"]))
        evo_cs.append(int(c0["cs"]))
        evo_cr.append(int(c0["cr"]))
        evo_is.append(int(c0["is"]))
        evo_ir.append(int(c0["ir"]))

        edge_rng = np.random.RandomState(region_seed + 9999)

        if args.export_gif:
            pos, ward_meta = _compute_block_layout(
                G_day0=G_day,
                patients=patients,
                staff=staff,
                ward_of=ward_of,
                staff_wards=effective_staff_wards,
                num_wards=args.num_wards,
                layout_seed=int(args.gif_layout_seed) + 1000 * int(r),
                ward_spacing=6.5,
                ward_pad=1.2,
                staff_jitter=0.35,
            )

            frames.append(_graph_to_frame(
                G=G_day,
                pos=pos,
                ward_meta=ward_meta,
                staff_wards=effective_staff_wards,
                day=0,
                region=r,
                max_edges_draw=int(args.gif_max_edges_draw),
                edge_sample_rng=edge_rng,
            ))

        for day in range(1, int(args.num_days) + 1):
            ss_active = ss_staff != "" and day >= ss_start and day <= ss_end

            patients, ward_of, next_patient_id, admitted_by_ward = _discharge_and_admit_patients(
                G_prev=G_day,
                patients=patients,
                ward_of=ward_of,
                num_wards=int(args.num_wards),
                day=int(day),
                params=params,
                next_patient_id=int(next_patient_id),
                rng=turnover_rng,
                intervention=causal_intervention if int(args.causal_mode) == 1 else None,
            )

            if ss_active and ss_state_code is not None:
                if ss_staff not in G_day.nodes:
                    raise ValueError(
                        f"Superspreader staff '{ss_staff}' not found in graph nodes. "
                        "Expected staff IDs like s0..s(n-1)."
                    )
                G_day.nodes[ss_staff]["amr_state"] = int(ss_state_code)

            effective_staff_wards, effective_staff_removed = _apply_runtime_staff_intervention(
                staff=staff,
                ward_of=ward_of,
                staff_wards=base_staff_wards,
                staff_removed=staff_removed,
                intervention=causal_intervention if int(args.causal_mode) == 1 else None,
                day=int(day),
            )

            G_new = sample_contacts(
                patients=patients,
                staff=staff,
                ward_of=ward_of,
                staff_wards=effective_staff_wards,
                num_wards=args.num_wards,
                staff_removed=effective_staff_removed,
                superspreader_staff=ss_staff if ss_staff else None,
                superspreader_active=bool(ss_active),
                superspreader_patient_frac_mult=args.superspreader_patient_frac_mult,
                superspreader_patient_min_add=args.superspreader_patient_min_add,
                superspreader_staff_contacts=args.superspreader_staff_contacts,
                superspreader_edge_weight_mult=args.superspreader_edge_weight_mult,
                intervention=causal_intervention if int(args.causal_mode) == 1 else None,
                day=int(day),
                params=params,
            )

            for n in G_new.nodes:
                if n in G_day.nodes:
                    G_new.nodes[n].update(G_day.nodes[n])
                else:
                    n_key = n if isinstance(n, str) else str(n)
                    is_staff = n_key.startswith("s")
                    home_ward = int(ward_of.get(n_key, ward_of.get(n, 0)))

                    if is_staff:
                        wards = staff_wards.get(n_key, staff_wards.get(n, [home_ward]))
                        wards = [w for w in sorted(set(int(w) for w in wards))]
                        ward_ids_str = ",".join(str(w) for w in wards) if wards else str(home_ward)
                    else:
                        ward_ids_str = str(home_ward)

                    G_new.nodes[n].update({
                        "role": "staff" if is_staff else "patient",
                        "ward_id": home_ward,
                        "ward_ids": ward_ids_str,
                        "amr_state": 0,
                        "abx_class": 0,
                        "is_isolated": 0,
                        "isolation_days_remaining": 0,
                        "screened_today": 0,
                        "observed_pos": 0,
                        "obs_status": 0,
                        "days_since_last_test": 999,
                        "pending_test_days": 0,
                        "pending_test_result": 0,
                "pending_test_result_state": 0,
                "observed_positive_type": 0,
                        "needs_admission_screen": 0,
                        "new_cr_acq_today": 0,
                        "new_ir_inf_today": 0,
                        "new_import_cr_today": 0,
                        "new_import_cs_today": 0,
                        "new_import_ir_today": 0,
                        "new_import_is_today": 0,
                        "new_trans_cr_today": 0,
                "new_trans_cs_today": 0,
                "transmission_source": "",
                "transmission_type": "",
                "competing_exposure_count": 0,
                        "new_select_cr_today": 0,
                        "node_id": str(n_key),
                    })

            G_day = G_new
            _apply_staff_presence_from_removed(G_day, effective_staff_removed)

            reset_daily_observation_flags(G_day, params)
            apply_isolation_decay(G_day)
            apply_pending_tests(G_day, params, day=int(day), intervention=causal_intervention if int(args.causal_mode) == 1 else None)

            run_admission_screening(G_day, params, day, intervention=causal_intervention if int(args.causal_mode) == 1 else None)
            run_screening(G_day, params, day, intervention=causal_intervention if int(args.causal_mode) == 1 else None)

            run_day_transmission(
                G_day,
                params,
                staff_removed,
                staff_timer,
                intervention=causal_intervention if int(args.causal_mode) == 1 else None,
                day=int(day),
            )
            _apply_staff_presence_from_removed(G_day, effective_staff_removed)
            update_days_since_last_test(G_day)

            _apply_graph_causal_metadata(G_day, causal_ctx)
            G_day.graph["day"] = int(day)
            G_day.graph["region"] = int(r)
            _attach_operational_graph_metadata(G_day, params, causal_intervention if int(args.causal_mode) == 1 else None, day=int(day))
            G_day.graph["admitted_by_ward_json"] = json.dumps(admitted_by_ward, sort_keys=True)

            G_day.graph["superspreader_staff"] = ss_staff
            G_day.graph["superspreader_state"] = ss_state_str
            G_day.graph["superspreader_active"] = int(bool(ss_active))
            G_day.graph["superspreader_params_json"] = json.dumps({
                "state": ss_state_str,
                "start_day": ss_start,
                "end_day": ss_end,
                "patient_frac_mult": float(args.superspreader_patient_frac_mult),
                "patient_min_add": int(args.superspreader_patient_min_add),
                "staff_contacts": int(args.superspreader_staff_contacts),
                "edge_weight_mult": float(args.superspreader_edge_weight_mult),
            }, sort_keys=True)

            if str(getattr(params, "admit_import_seasonality", "none")).strip().lower() == "shock":
                G_day.graph["admit_import_shock_json"] = json.dumps({
                    "start_day": int(getattr(params, "admit_import_shock_start_day", 0)),
                    "duration_days": int(getattr(params, "admit_import_shock_duration_days", 0)),
                    "mult": float(getattr(params, "admit_import_shock_mult", 1.0)),
                }, sort_keys=True)

            out_path = os.path.join(args.output_dir, f"amr_r{r}_t{day}.graphml")
            nx.write_graphml(G_day, out_path)
            print(f"DT_SIM_DAY {day}", flush=True)

            cd = _count_states(G_day)
            evo_days.append(int(day))
            evo_u.append(int(cd["u"]))
            evo_cs.append(int(cd["cs"]))
            evo_cr.append(int(cd["cr"]))
            evo_is.append(int(cd["is"]))
            evo_ir.append(int(cd["ir"]))

            if args.export_gif:
                frames.append(_graph_to_frame(
                    G=G_day,
                    pos=pos,
                    ward_meta=ward_meta,
                    staff_wards=effective_staff_wards,
                    day=day,
                    region=r,
                    max_edges_draw=int(args.gif_max_edges_draw),
                    edge_sample_rng=edge_rng,
                ))

        evo_png = os.path.join(args.output_dir, f"state_evolution_{split_label}_r{r}.png")
        save_state_evolution_png(
            days=evo_days,
            u=evo_u,
            cs=evo_cs,
            cr=evo_cr,
            is_=evo_is,
            ir=evo_ir,
            out_path=evo_png,
            title=f"AMR state evolution ({split_label}) | region {r}",
        )

    if args.export_gif:
        out_gif = os.path.join(args.output_dir, "amr_simulation.gif")
        _save_gif(frames, out_gif, fps=float(args.gif_fps))
        print(f"DT_SIM_GIF gif={out_gif} frames={len(frames)} fps={float(args.gif_fps)}", flush=True)

    print("✅ AMR simulation complete.", flush=True)


if __name__ == "__main__":
    main()
