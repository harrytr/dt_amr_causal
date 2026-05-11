#!/usr/bin/env python3
from __future__ import annotations

"""
causal_interventions.py

Utility helpers for Stage 1 simulator-world causal interventions.

This module validates and normalizes intervention specifications that are later
consumed by generate_amr_data.py and counterfactual_rollout.py.

The implementation is intentionally conservative: it accepts only a small,
explicit intervention family so the first causal layer remains auditable.
"""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


ALLOWED_INTERVENTION_NAMES = {
    "reduce_ward_importation",
    "remove_staff_crossward_cover",
    "remove_specific_staff",
    "remove_edge",
    "set_screening_frequency",
    "set_screening_delay",
    "disable_isolation_response",
    "set_isolation_parameters",
}


class InterventionValidationError(ValueError):
    """Raised when an intervention specification is invalid."""


@dataclass(frozen=True)
class CausalInterventionSpec:
    name: str
    target_type: str
    target_id: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    start_day: Optional[int] = None
    end_day: Optional[int] = None
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _require_mapping(obj: Any, *, ctx: str) -> Mapping[str, Any]:
    if not isinstance(obj, Mapping):
        raise InterventionValidationError(f"{ctx} must be a JSON object / mapping.")
    return obj


def _coerce_positive_int(value: Any, *, field_name: str, allow_zero: bool = False) -> int:
    try:
        ivalue = int(value)
    except Exception as exc:
        raise InterventionValidationError(f"{field_name} must be an integer.") from exc
    if allow_zero:
        if ivalue < 0:
            raise InterventionValidationError(f"{field_name} must be >= 0.")
    else:
        if ivalue <= 0:
            raise InterventionValidationError(f"{field_name} must be > 0.")
    return ivalue


def _coerce_float(value: Any, *, field_name: str) -> float:
    try:
        return float(value)
    except Exception as exc:
        raise InterventionValidationError(f"{field_name} must be numeric.") from exc


def _coerce_probability_multiplier(value: Any, *, field_name: str) -> float:
    fvalue = _coerce_float(value, field_name=field_name)
    if fvalue < 0.0:
        raise InterventionValidationError(f"{field_name} must be >= 0.")
    return fvalue



def _coerce_isolation_trigger_policy(value: Any, *, field_name: str = "isolation_trigger_policy") -> str:
    sval = str(value).strip().lower()
    aliases = {
        "all": "all_positives",
        "all_positive": "all_positives",
        "all_positives": "all_positives",
        "any": "all_positives",
        "any_positive": "all_positives",
        "resistant": "resistant_only",
        "resistant_only": "resistant_only",
    }
    if sval not in aliases:
        raise InterventionValidationError(f"{field_name} must be one of resistant_only or all_positives.")
    return aliases[sval]

def _validate_day_window(start_day: Optional[Any], end_day: Optional[Any]) -> tuple[Optional[int], Optional[int]]:
    sd = None if start_day is None else _coerce_positive_int(start_day, field_name="start_day")
    ed = None if end_day is None else _coerce_positive_int(end_day, field_name="end_day")
    if sd is not None and ed is not None and ed < sd:
        raise InterventionValidationError("end_day must be >= start_day.")
    return sd, ed


def validate_intervention_dict(spec: Mapping[str, Any]) -> CausalInterventionSpec:
    spec = _require_mapping(spec, ctx="intervention")
    name = str(spec.get("name", "")).strip()
    if name not in ALLOWED_INTERVENTION_NAMES:
        raise InterventionValidationError(
            f"Unsupported intervention '{name}'. Allowed: {sorted(ALLOWED_INTERVENTION_NAMES)}"
        )

    target_type = str(spec.get("target_type", "")).strip()
    target_id = str(spec.get("target_id", "")).strip()
    params_raw = spec.get("params", {})
    params: Dict[str, Any] = dict(_require_mapping(params_raw, ctx="intervention.params"))
    description = str(spec.get("description", "")).strip()
    start_day, end_day = _validate_day_window(spec.get("start_day"), spec.get("end_day"))

    if name == "reduce_ward_importation":
        if target_type != "ward":
            raise InterventionValidationError("reduce_ward_importation requires target_type='ward'.")
        if not target_id:
            raise InterventionValidationError("reduce_ward_importation requires a non-empty target_id.")
        params.setdefault("multiplier_cs", 1.0)
        params.setdefault("multiplier_cr", 1.0)
        params["multiplier_cs"] = _coerce_probability_multiplier(
            params["multiplier_cs"], field_name="multiplier_cs"
        )
        params["multiplier_cr"] = _coerce_probability_multiplier(
            params["multiplier_cr"], field_name="multiplier_cr"
        )

    elif name == "remove_staff_crossward_cover":
        if target_type != "staff":
            raise InterventionValidationError("remove_staff_crossward_cover requires target_type='staff'.")
        if not target_id:
            raise InterventionValidationError("remove_staff_crossward_cover requires a non-empty target_id.")

    elif name == "remove_specific_staff":
        if target_type != "staff":
            raise InterventionValidationError("remove_specific_staff requires target_type='staff'.")
        if not target_id:
            raise InterventionValidationError("remove_specific_staff requires a non-empty target_id.")

    elif name == "remove_edge":
        if target_type != "edge":
            raise InterventionValidationError("remove_edge requires target_type='edge'.")
        src = str(params.get("source", "")).strip()
        dst = str(params.get("target", "")).strip()
        if not src or not dst:
            raise InterventionValidationError("remove_edge requires params.source and params.target.")
        params["source"] = src
        params["target"] = dst

    elif name == "set_screening_frequency":
        if target_type not in {"global", "policy"}:
            raise InterventionValidationError("set_screening_frequency requires target_type='global' or 'policy'.")
        params["frequency_days"] = _coerce_positive_int(
            params.get("frequency_days"), field_name="frequency_days"
        )
        if "screen_on_admission" in params:
            params["screen_on_admission"] = _coerce_positive_int(
                params.get("screen_on_admission"), field_name="screen_on_admission", allow_zero=True
            )
            if params["screen_on_admission"] not in {0, 1}:
                raise InterventionValidationError("screen_on_admission must be 0 or 1.")
        if "delay_days" in params or "screen_result_delay_days" in params:
            delay_value = params.get("delay_days", params.get("screen_result_delay_days"))
            delay_value = _coerce_positive_int(delay_value, field_name="delay_days", allow_zero=True)
            params["delay_days"] = delay_value
            params["screen_result_delay_days"] = delay_value
        if "isolation_mult" in params or "transmission_multiplier" in params:
            if "isolation_mult" in params:
                mult_value = _coerce_probability_multiplier(
                    params.get("isolation_mult"), field_name="isolation_mult"
                )
            else:
                mult_value = _coerce_probability_multiplier(
                    params.get("transmission_multiplier"), field_name="transmission_multiplier"
                )
            params["isolation_mult"] = mult_value
            params["transmission_multiplier"] = mult_value
        if "isolation_days" in params:
            params["isolation_days"] = _coerce_positive_int(
                params.get("isolation_days"), field_name="isolation_days", allow_zero=True
            )
        if "isolation_trigger_policy" in params:
            params["isolation_trigger_policy"] = _coerce_isolation_trigger_policy(params["isolation_trigger_policy"])

    elif name == "set_screening_delay":
        if target_type not in {"global", "policy"}:
            raise InterventionValidationError("set_screening_delay requires target_type='global' or 'policy'.")
        params["delay_days"] = _coerce_positive_int(
            params.get("delay_days"), field_name="delay_days", allow_zero=True
        )

    elif name == "disable_isolation_response":
        if target_type not in {"global", "policy"}:
            raise InterventionValidationError("disable_isolation_response requires target_type='global' or 'policy'.")

    elif name == "set_isolation_parameters":
        if target_type not in {"global", "policy"}:
            raise InterventionValidationError("set_isolation_parameters requires target_type='global' or 'policy'.")

        if "isolation_mult" in params:
            params["isolation_mult"] = _coerce_probability_multiplier(
                params.get("isolation_mult"), field_name="isolation_mult"
            )
        elif "transmission_multiplier" in params:
            params["transmission_multiplier"] = _coerce_probability_multiplier(
                params.get("transmission_multiplier"), field_name="transmission_multiplier"
            )
        else:
            raise InterventionValidationError(
                "set_isolation_parameters requires params.isolation_mult or params.transmission_multiplier."
            )

        params["isolation_days"] = _coerce_positive_int(
            params.get("isolation_days"), field_name="isolation_days", allow_zero=True
        )
        if "isolation_trigger_policy" in params:
            params["isolation_trigger_policy"] = _coerce_isolation_trigger_policy(params["isolation_trigger_policy"])

    return CausalInterventionSpec(
        name=name,
        target_type=target_type,
        target_id=target_id,
        params=params,
        start_day=start_day,
        end_day=end_day,
        description=description,
    )


def load_intervention_json(path: str | Path) -> CausalInterventionSpec:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return validate_intervention_dict(payload)


def load_interventions_json(path: str | Path) -> List[CausalInterventionSpec]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        return [validate_intervention_dict(payload)]
    if not isinstance(payload, list):
        raise InterventionValidationError("Intervention file must contain either one object or a list of objects.")
    return [validate_intervention_dict(item) for item in payload]


def save_intervention_json(spec: CausalInterventionSpec, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def describe_intervention(spec: CausalInterventionSpec | Mapping[str, Any]) -> str:
    if not isinstance(spec, CausalInterventionSpec):
        spec = validate_intervention_dict(spec)

    if spec.name == "reduce_ward_importation":
        return (
            f"Reduce ward importation for {spec.target_id} "
            f"(CR x{spec.params['multiplier_cr']:.3g}, CS x{spec.params['multiplier_cs']:.3g})."
        )
    if spec.name == "remove_staff_crossward_cover":
        return f"Restrict staff member {spec.target_id} to home-ward coverage only."
    if spec.name == "remove_specific_staff":
        return f"Remove staff member {spec.target_id} from daily contact generation."
    if spec.name == "remove_edge":
        return f"Remove directed edge {spec.params['source']} -> {spec.params['target']}."
    if spec.name == "set_screening_frequency":
        desc = f"Set screening frequency to every {spec.params['frequency_days']} day(s)."
        if "screen_on_admission" in spec.params:
            desc += f" Admission screening={int(spec.params['screen_on_admission'])}."
        if "delay_days" in spec.params or "screen_result_delay_days" in spec.params:
            delay_val = spec.params.get("delay_days", spec.params.get("screen_result_delay_days", 0))
            desc += f" Result delay={int(delay_val)} day(s)."
        if "isolation_mult" in spec.params or "transmission_multiplier" in spec.params:
            mult = spec.params.get("isolation_mult", spec.params.get("transmission_multiplier", 1.0))
            desc += f" Isolation multiplier={float(mult):.3g}."
        if "isolation_days" in spec.params:
            desc += f" Isolation duration={int(spec.params['isolation_days'])} day(s)."
        if "isolation_trigger_policy" in spec.params:
            desc += f" Isolation trigger={spec.params['isolation_trigger_policy']}."
        return desc
    if spec.name == "set_screening_delay":
        return f"Set screening-result delay to {spec.params['delay_days']} day(s)."
    if spec.name == "disable_isolation_response":
        return "Disable isolation response after positive results while preserving observation state."
    if spec.name == "set_isolation_parameters":
        if "isolation_mult" in spec.params:
            mult = spec.params["isolation_mult"]
        else:
            mult = spec.params["transmission_multiplier"]
        return (
            f"Set isolation parameters to transmission multiplier {mult:.3g} "
            f"for {int(spec.params['isolation_days'])} day(s) "
            f"with trigger {spec.params.get('isolation_trigger_policy', 'resistant_only')}."
        )
    return f"Intervention {spec.name}."


def specs_to_jsonable(specs: Iterable[CausalInterventionSpec]) -> List[Dict[str, Any]]:
    return [spec.to_dict() for spec in specs]


__all__ = [
    "ALLOWED_INTERVENTION_NAMES",
    "CausalInterventionSpec",
    "InterventionValidationError",
    "describe_intervention",
    "load_intervention_json",
    "load_interventions_json",
    "save_intervention_json",
    "specs_to_jsonable",
    "validate_intervention_dict",
]
