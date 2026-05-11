#!/usr/bin/env python3
"""
temporal_graph_dataset.py
========================

AMR-only temporal dataset.

Supports two modes:

1. Standard folder mode
   - Reads a folder of daily .pt graphs
   - Produces sliding windows of length T

2. Policy-manifest mode
   - Reads policy_manifest.csv produced by build_causal_policy_dataset.py
   - Each row explicitly defines one supervised sample:
         (window_pt_json, action metadata) -> action-conditioned outcome
   - This preserves repeated copies of the same pre-intervention state across
     different candidate actions.

Trajectory grouping (standard folder mode):
- Prefer Data.sim_id + Data.day (robust to mixing pooled folders).
- Fallback to filename parsing: <sim_prefix>_t<day>[ _L<label> ].pt

Enhancement (stable node identity support):
- Builds (or loads) a per-folder node vocabulary from Data.node_names
- Attaches Data.node_id (LongTensor) aligned to Data.x row order

Returns:
  graphs      : list[T] of PyG Data
  labels_dict : dict[str, Tensor]
      - kept for interface compatibility
      - in policy-manifest mode includes action_features and selected manifest tensors
"""

import json
import os
import re
import hashlib
import math
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


def natural_key(s: str):
    parts = re.split(r"(\d+)", s)
    out: List[object] = []
    for p in parts:
        if p.isdigit():
            out.append(int(p))
        else:
            out.append(p.lower())
    return out


def _parse_sim_day_label(fname: str, file_ext: str):
    """
    Parse filenames of the form:
        <sim_prefix>_t<day>[ _L<label> ]<file_ext>

    Returns:
        (sim_prefix: str, day: int, label: Optional[int])
    or None if the filename does not match.
    """
    fe = str(file_ext)
    pat = rf"^(?P<prefix>.+?)_t(?P<t>\d+)(?:_L(?P<label>\d+))?{re.escape(fe)}$"
    m = re.match(pat, str(fname))
    if not m:
        return None
    prefix = str(m.group("prefix"))
    t = int(m.group("t"))
    lab = m.group("label")
    label = int(lab) if lab is not None else None
    return prefix, t, label


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        try:
            if hasattr(x, "item"):
                return int(x.item())
        except Exception:
            return None
    return None


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        try:
            if hasattr(x, "item"):
                return float(x.item())
        except Exception:
            return float(default)
    return float(default)


def _safe_json_loads(s: str, default):
    try:
        return json.loads(str(s))
    except Exception:
        return default




def _encode_multiplier_feature(x, default: float = 1.0) -> float:
    """Neutral-centred encoding for strictly positive multiplier-like quantities."""
    z = _safe_float(x, default=default)
    if not math.isfinite(z) or z <= 0.0:
        return 0.0
    return float(math.log(z))

def _read_pt_metadata(path: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Read (sim_id, day) from a .pt Data object if present.
    Returns (None, None) if not present or not parseable.

    Note: This loads the object.
    """
    try:
        data = torch.load(path, weights_only=False)
    except Exception:
        return None, None

    sim_id = getattr(data, "sim_id", None)
    day = getattr(data, "day", None)

    if sim_id is None or day is None:
        return None, None

    sim_id_s = str(sim_id)
    day_i = _safe_int(day)
    if sim_id_s.strip() == "" or day_i is None:
        return None, None

    return sim_id_s, int(day_i)


def _hash_to_unit_interval(text: str) -> float:
    h = hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:12]
    return int(h, 16) / float(16**12 - 1)


def _normalize_day(value: Any, fallback: float = 0.0, scale: float = 365.0) -> float:
    v = _safe_float(value, default=fallback)
    if scale <= 0:
        scale = 365.0
    return float(v) / float(scale)


def _normalize_action_family(action_name: str, payload: Dict[str, Any]) -> str:
    name = str(action_name).strip().lower()
    if name == "":
        name = str(payload.get("name", payload.get("intervention_name", ""))).strip().lower()

    if name in {"", "baseline"}:
        return "baseline"

    alias_map = {
        "set_screening_frequency_2": "set_screening_frequency",
        "set_screening_frequency_3": "set_screening_frequency",
        "set_screening_frequency_4": "set_screening_frequency",
    }
    if name in alias_map:
        return alias_map[name]

    if re.match(r"^set_screening_frequency_\d+$", name):
        return "set_screening_frequency"

    return name


def _build_action_features_from_manifest_row(row: Dict[str, Any]) -> torch.Tensor:
    """
    Stable fixed-width action feature encoding derived from policy-manifest metadata.

    The encoding uses:
      - a coarse intervention-family block
      - target-type indicators
      - normalized numeric intervention parameters
      - lightweight hashed identifiers so sibling actions within the same family
        can still be distinguished without requiring a global fitted vocabulary
    """
    raw_action_name = str(row.get("action_name", "")).strip().lower()
    raw_action_id = str(row.get("action_id", raw_action_name)).strip().lower()
    is_baseline = 1.0 if str(row.get("is_baseline", "0")).strip() in {"1", "true", "True"} else 0.0

    payload = _safe_json_loads(row.get("action_intervention_json", "") or "", {})
    if payload in ("", None):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    action_family = _normalize_action_family(raw_action_name, payload)

    target_type = str(payload.get("target_type", row.get("action_target_type", ""))).strip().lower()
    target_id = str(payload.get("target_id", row.get("action_target_id", ""))).strip()
    params = payload.get("params", {})
    if not isinstance(params, dict):
        params = {}

    intervention_names = [
        "baseline",
        "set_screening_frequency",
        "set_screening_delay",
        "disable_isolation_response",
        "set_isolation_parameters",
        "reduce_ward_importation",
        "remove_staff_crossward_cover",
        "remove_specific_staff",
        "remove_edge",
        "other",
    ]
    type_block = [0.0] * len(intervention_names)
    if is_baseline >= 0.5 or action_family == "baseline":
        type_block[0] = 1.0
    else:
        matched = False
        for i, name in enumerate(intervention_names[1:-1], start=1):
            if action_family == name:
                type_block[i] = 1.0
                matched = True
                break
        if not matched:
            type_block[-1] = 1.0

    target_types = ["", "global", "ward", "staff", "edge", "patient", "region", "hospital"]
    tgt_block = [0.0] * len(target_types)
    tgt_idx = 0
    for i, tt in enumerate(target_types):
        if target_type == tt:
            tgt_idx = i
            break
    tgt_block[tgt_idx] = 1.0

    screen_freq = _safe_float(
        params.get("frequency_days", params.get("screen_every_k_days", params.get("k_days", 0))),
        default=0.0,
    )
    screen_delay = _safe_float(
        params.get("delay_days", params.get("screen_result_delay_days", 0)),
        default=0.0,
    )
    multiplier = _safe_float(
        params.get("multiplier", params.get("multiplier_cr", params.get("multiplier_cs", 1.0))),
        default=1.0,
    )
    multiplier_cr = _safe_float(params.get("multiplier_cr", multiplier), default=multiplier)
    multiplier_cs = _safe_float(params.get("multiplier_cs", multiplier), default=multiplier)
    isolation_mult = _safe_float(
        params.get("isolation_mult", params.get("transmission_multiplier", multiplier)),
        default=multiplier,
    )
    isolation_days = _safe_float(params.get("isolation_days", 0), default=0.0)
    screen_on_admission = _safe_float(params.get("screen_on_admission", 0), default=0.0)

    start_day = row.get("action_start_day", payload.get("start_day", 0))
    end_day = payload.get("end_day", 0)

    action_name_hash = _hash_to_unit_interval(raw_action_name) if raw_action_name != "" else 0.0
    action_id_hash = _hash_to_unit_interval(raw_action_id) if raw_action_id != "" else action_name_hash
    target_hash = _hash_to_unit_interval(target_id) if target_id != "" else 0.0

    feats = (
        type_block
        + tgt_block
        + [
            float(is_baseline),
            float(1.0 if str(row.get("policy_valid", "0")).strip() in {"1", "true", "True"} else 0.0),
            min(max(screen_freq / 30.0, 0.0), 1.0),
            min(max(screen_delay / 30.0, 0.0), 1.0),
            _encode_multiplier_feature(multiplier, 1.0),
            _encode_multiplier_feature(multiplier_cr, 1.0),
            _encode_multiplier_feature(multiplier_cs, 1.0),
            _encode_multiplier_feature(isolation_mult, 1.0),
            min(max(isolation_days / 30.0, 0.0), 1.0),
            min(max(screen_on_admission, 0.0), 1.0),
            _normalize_day(start_day),
            _normalize_day(end_day),
            action_name_hash,
            action_id_hash,
            target_hash,
        ]
    )
    return torch.tensor(feats, dtype=torch.float32)




_STATE_SUMMARY_SCALE_QUANTILE = 0.95


def _safe_log1p(value: Any) -> float:
    z = _safe_float(value, default=0.0)
    if not math.isfinite(z):
        return 0.0
    return float(math.log1p(max(0.0, z)))


def _scaled_log_feature(value: Any, reference: Any) -> float:
    ref = _safe_float(reference, default=1.0)
    if not math.isfinite(ref) or ref <= 1e-12:
        ref = 1.0
    return float(min(max(_safe_log1p(value) / ref, 0.0), 1.0))


def _positive_squash_unit(value: Any) -> float:
    z = _safe_float(value, default=0.0)
    if not math.isfinite(z) or z <= 0.0:
        return 0.0
    return float(z / (1.0 + z))


def _normalized_entropy_from_tensor(values: Optional[torch.Tensor]) -> float:
    if not torch.is_tensor(values) or values.numel() <= 0:
        return 0.0
    flat = values.to(torch.float32).view(-1)
    unique, counts = torch.unique(flat, return_counts=True)
    k = int(unique.numel())
    if k <= 1:
        return 0.0
    probs = counts.to(torch.float32) / float(counts.sum().item())
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    return float(entropy / math.log(float(k)))


def _default_state_summary_calibration() -> Dict[str, Any]:
    return {
        "num_nodes_log_q": 1.0,
        "edge_count_log_q": 1.0,
        "mean_out_degree_log_q": 1.0,
        "quantile": float(_STATE_SUMMARY_SCALE_QUANTILE),
        "reference_split": "none",
        "n_reference_graphs": 0,
    }


def _graph_summary_calibration_payload(data) -> Dict[str, float]:
    x = getattr(data, "x", None)
    if torch.is_tensor(x) and x.dim() == 2:
        num_nodes = int(x.size(0))
    else:
        try:
            num_nodes = int(getattr(data, "num_nodes", 0))
        except Exception:
            num_nodes = 0

    edge_index = getattr(data, "edge_index", None)
    edge_count = int(edge_index.size(1)) if torch.is_tensor(edge_index) and edge_index.dim() == 2 else 0
    mean_out_degree = float(edge_count) / float(max(1, num_nodes))

    return {
        "num_nodes": float(max(0, num_nodes)),
        "edge_count": float(max(0, edge_count)),
        "mean_out_degree": float(max(0.0, mean_out_degree)),
    }


def _upgrade_legacy_operational_context_features(data) -> None:
    op = getattr(data, "operational_context_features", None)
    if not torch.is_tensor(op):
        return

    version = getattr(data, "operational_context_encoding_version", None)
    version_i = _safe_int(version)
    if version_i is not None and version_i >= 2:
        return

    flat = op.to(torch.float32).view(-1).clone()
    if flat.numel() >= 5:
        legacy_val = float(flat[4].item())
        recovered = max(0.0, 5.0 * legacy_val)
        flat[4] = float(_encode_multiplier_feature(recovered, 1.0))

    data.operational_context_features = flat.view_as(op)
    data.operational_context_encoding_version = 2


def _build_state_summary_features_from_graph(
    data,
    calibration: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    """
    Build explicit decision-state summary features from tensors already attached
    to the graph object for the current track.

    Important:
      - This function must NOT read latent/raw simulator attributes directly from
        GraphML at dataset time.
      - It uses graph-attached tensors such as g.x, g.edge_index, g.edge_attr,
        and optional operational_context_features that were already materialized
        during PT conversion.
    """
    x = getattr(data, "x", None)
    if not torch.is_tensor(x) or x.dim() != 2 or x.size(0) <= 0:
        return torch.zeros(49, dtype=torch.float32)

    if calibration is None:
        calibration = _default_state_summary_calibration()

    x = x.to(torch.float32)
    num_nodes = int(x.size(0))
    denom = float(max(1, num_nodes))

    feat_means = x.mean(dim=0)
    feat_stds = x.std(dim=0, unbiased=False) if num_nodes > 1 else torch.zeros_like(feat_means)

    role_col = x[:, 0] if x.size(1) >= 1 else torch.zeros((num_nodes,), dtype=torch.float32)

    zeros = torch.zeros((num_nodes,), dtype=torch.float32)
    obs_positive_col = zeros
    obs_known_col = zeros
    screened_today_col = zeros
    days_since_last_test_col = zeros
    pending_test_days_col = zeros
    pending_test_result_col = zeros
    needs_admission_screen_col = zeros
    present_today_col = torch.ones((num_nodes,), dtype=torch.float32)
    isolation_days_remaining_col = zeros
    admission_age_col = zeros

    if x.size(1) >= 22:
        # ground-truth layout with operational/testing features appended
        state_col = x[:, 3] + x[:, 4] + 2.0 * x[:, 5]
        abx_col = x[:, 6]
        iso_col = x[:, 7]
        new_cr_col = x[:, 8]
        new_ir_col = x[:, 9]
        obs_positive_col = x[:, 12]
        obs_known_col = x[:, 13]
        screened_today_col = x[:, 14]
        days_since_last_test_col = x[:, 15]
        pending_test_days_col = x[:, 16]
        pending_test_result_col = x[:, 17]
        needs_admission_screen_col = x[:, 18]
        present_today_col = x[:, 19]
        isolation_days_remaining_col = x[:, 20]
        admission_age_col = x[:, 21]
    elif x.size(1) >= 16:
        # partial-observation layout with operational/testing features appended
        state_col = x[:, 1]
        abx_col = x[:, 2]
        iso_col = x[:, 3]
        new_cr_col = x[:, 4]
        new_ir_col = x[:, 5]
        obs_positive_col = x[:, 1]
        obs_known_col = x[:, 8]
        screened_today_col = x[:, 9]
        days_since_last_test_col = x[:, 10]
        pending_test_days_col = x[:, 11]
        needs_admission_screen_col = x[:, 12]
        present_today_col = x[:, 13]
        isolation_days_remaining_col = x[:, 14]
        admission_age_col = x[:, 15]
    elif x.size(1) >= 12:
        # legacy ground-truth one-hot state layout
        state_col = x[:, 3] + x[:, 4] + 2.0 * x[:, 5]
        abx_col = x[:, 6]
        iso_col = x[:, 7]
        new_cr_col = x[:, 8]
        new_ir_col = x[:, 9]
    elif x.size(1) >= 8:
        # legacy partial-observation layout
        state_col = x[:, 1]
        abx_col = x[:, 2]
        iso_col = x[:, 3]
        new_cr_col = x[:, 4]
        new_ir_col = x[:, 5]
        obs_positive_col = x[:, 1]
    else:
        # legacy layout fallback
        state_col = x[:, 1] if x.size(1) >= 2 else zeros
        abx_col = x[:, 2] if x.size(1) >= 3 else zeros
        iso_col = x[:, 3] if x.size(1) >= 4 else zeros
        new_cr_col = x[:, 4] if x.size(1) >= 5 else zeros
        new_ir_col = x[:, 5] if x.size(1) >= 6 else zeros

    staff_frac = float((role_col >= 0.5).to(torch.float32).mean().item())
    patient_frac = float(1.0 - staff_frac)
    state_mean = float(state_col.mean().item())
    state_std = float(state_col.std(unbiased=False).item()) if state_col.numel() > 1 else 0.0
    state_min = float(state_col.min().item()) if state_col.numel() > 0 else 0.0
    state_max = float(state_col.max().item()) if state_col.numel() > 0 else 0.0
    state_positive_frac = float((state_col > 0.0).to(torch.float32).mean().item())
    abx_positive_frac = float((abx_col > 0.0).to(torch.float32).mean().item())
    iso_positive_frac = float((iso_col > 0.0).to(torch.float32).mean().item())

    edge_index = getattr(data, "edge_index", None)
    edge_count = int(edge_index.size(1)) if torch.is_tensor(edge_index) and edge_index.dim() == 2 else 0
    mean_out_degree = float(edge_count) / float(max(1, num_nodes))
    density = float(edge_count) / float(max(1, num_nodes * num_nodes))

    edge_attr = getattr(data, "edge_attr", None)
    edge_weight_mean = 0.0
    edge_weight_std = 0.0
    edge_type_entropy = 0.0
    if torch.is_tensor(edge_attr) and edge_attr.numel() > 0:
        ea = edge_attr.to(torch.float32)
        if ea.dim() == 1:
            ea = ea.view(-1, 1)
        if ea.size(1) >= 1:
            edge_weight_mean = float(ea[:, 0].mean().item())
            edge_weight_std = float(ea[:, 0].std(unbiased=False).item()) if ea.size(0) > 1 else 0.0
        if ea.size(1) >= 2:
            edge_type_entropy = _normalized_entropy_from_tensor(ea[:, 1])

    operational_context = getattr(data, "operational_context_features", None)
    if torch.is_tensor(operational_context):
        op = operational_context.to(torch.float32).view(-1)
    else:
        op = torch.zeros(9, dtype=torch.float32)

    feat_values = [
        float(feat_means[i].item()) if i < feat_means.numel() else 0.0
        for i in range(min(6, int(feat_means.numel())))
    ]
    feat_std_values = [
        float(feat_stds[i].item()) if i < feat_stds.numel() else 0.0
        for i in range(min(6, int(feat_stds.numel())))
    ]

    num_nodes_log_q = calibration.get("num_nodes_log_q", 1.0)
    edge_count_log_q = calibration.get("edge_count_log_q", 1.0)
    mean_out_degree_log_q = calibration.get("mean_out_degree_log_q", 1.0)

    feats = feat_values + feat_std_values + [
        staff_frac,
        patient_frac,
        state_mean,
        state_std,
        state_min,
        state_max,
        state_positive_frac,
        abx_positive_frac,
        iso_positive_frac,
        float(new_cr_col.sum().item()) / denom,
        float(new_ir_col.sum().item()) / denom,
        _scaled_log_feature(float(num_nodes), num_nodes_log_q),
        _scaled_log_feature(float(edge_count), edge_count_log_q),
        _scaled_log_feature(mean_out_degree, mean_out_degree_log_q),
        min(max(density, 0.0), 1.0),
        _positive_squash_unit(edge_weight_mean),
        _positive_squash_unit(edge_weight_std),
        edge_type_entropy,
        float(obs_positive_col.mean().item()),
        float(obs_known_col.mean().item()),
        float(screened_today_col.mean().item()),
        float(days_since_last_test_col.mean().item()),
        float(pending_test_days_col.mean().item()),
        float(pending_test_result_col.mean().item()),
        float(needs_admission_screen_col.mean().item()),
        float(present_today_col.mean().item()),
        float(isolation_days_remaining_col.mean().item()),
        float(admission_age_col.mean().item()),
    ]
    return torch.cat([torch.tensor(feats, dtype=torch.float32), op], dim=0)

def _looks_like_bool_str(x: Any) -> bool:
    return str(x).strip().lower() in {"0", "1", "true", "false"}


def _tensor_from_manifest_value(key: str, value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value

    key_s = str(key)

    if key_s in {
        "seed",
        "shared_noise_seed",
        "decision_day",
        "action_start_day",
        "window_T",
        "policy_valid",
    }:
        return torch.tensor([int(_safe_float(value, 0.0))], dtype=torch.long)

    if key_s == "is_baseline":
        return torch.tensor([1 if str(value).strip() in {"1", "true", "True"} else 0], dtype=torch.long)

    if key_s.startswith("y_h"):
        # Manifest targets with suffixes like _majority, _cls, and _emergence are
        # classification labels. Gain targets are regression targets and must stay
        # float-valued so burden/gain tasks are not silently quantized.
        if key_s.endswith("_majority") or key_s.endswith("_cls") or key_s.endswith("_emergence"):
            return torch.tensor([int(_safe_float(value, 0.0))], dtype=torch.long)
        return torch.tensor([float(_safe_float(value, 0.0))], dtype=torch.float32)

    if _looks_like_bool_str(value):
        return torch.tensor([1 if str(value).strip().lower() in {"1", "true"} else 0], dtype=torch.long)

    try:
        f = float(value)
    except Exception:
        return None

    if float(int(f)) == float(f):
        return torch.tensor([int(f)], dtype=torch.long)
    return torch.tensor([float(f)], dtype=torch.float32)


class TemporalGraphDataset(Dataset):
    """Temporal windows dataset for AMR daily graphs."""

    def __init__(
        self,
        folder: str,
        T: int,
        sliding_step: int = 1,
        file_ext: str = ".pt",
        cache_all: bool = False,
        build_node_vocab: bool = True,
        node_vocab_filename: str = "node_vocab.json",
        prefer_pt_metadata: bool = True,
        require_pt_metadata: bool = True,
        fail_on_noncontiguous: bool = True,
        policy_manifest_csv: Optional[str] = None,
        use_policy_manifest: Optional[bool] = None,
    ):
        self.folder = os.path.abspath(folder)
        self.T = int(T)
        self.sliding_step = int(sliding_step)
        self.file_ext = str(file_ext)

        self.cache_all = bool(cache_all)
        self._cache_max = max(1, 2 * self.T)
        self._cache: "OrderedDict[str, object]" = OrderedDict()
        self._disk_paths: Dict[str, str] = {}

        self.groups: List[List[str]] = []
        self.samples: List[Dict[str, Any]] = []
        self.state_summary_calibration: Dict[str, Any] = _default_state_summary_calibration()

        self.prefer_pt_metadata = bool(prefer_pt_metadata)
        self.require_pt_metadata = bool(require_pt_metadata)
        self.fail_on_noncontiguous = bool(fail_on_noncontiguous)

        default_manifest = os.path.join(self.folder, "policy_manifest.csv")
        self.policy_manifest_csv = (
            os.path.abspath(policy_manifest_csv)
            if policy_manifest_csv is not None
            else (os.path.abspath(default_manifest) if os.path.isfile(default_manifest) else None)
        )
        if use_policy_manifest is None:
            self.use_policy_manifest = bool(self.policy_manifest_csv is not None and os.path.isfile(self.policy_manifest_csv))
        else:
            self.use_policy_manifest = bool(use_policy_manifest)

        self.build_node_vocab = bool(build_node_vocab)
        self.node_vocab_filename = str(node_vocab_filename)
        self.node_vocab_path = os.path.join(self.folder, self.node_vocab_filename)
        self.node_vocab: Dict[str, int] = {}
        self.node_vocab_inv: List[str] = []

        if self.use_policy_manifest:
            self._scan_policy_manifest()
        else:
            self._scan_folder()
        self._init_state_summary_calibration()

        if self.build_node_vocab:
            self._init_node_vocab()

    def _scan_policy_manifest(self) -> None:
        manifest_path = self.policy_manifest_csv
        if manifest_path is None or not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Policy manifest not found: {manifest_path}")

        import csv

        n_rows = 0
        n_kept = 0
        with open(manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                n_rows += 1

                window_paths_raw = _safe_json_loads(row.get("window_pt_json", ""), [])
                if not isinstance(window_paths_raw, list):
                    raise RuntimeError(f"Invalid window_pt_json in manifest row {n_rows} of '{manifest_path}'.")

                window_paths = [os.path.abspath(str(p)) for p in window_paths_raw]
                if len(window_paths) != self.T:
                    continue

                missing = [p for p in window_paths if not os.path.isfile(p)]
                if missing:
                    raise FileNotFoundError(
                        f"Manifest row {n_rows} references missing PT files. Example missing path: {missing[0]}"
                    )

                days: List[int] = []
                sim_ids: List[str] = []
                for p in window_paths:
                    sim_id, day = _read_pt_metadata(p)
                    if sim_id is not None and day is not None:
                        sim_ids.append(str(sim_id))
                        days.append(int(day))

                if len(days) == self.T:
                    if len(set(sim_ids)) != 1:
                        raise RuntimeError(
                            f"Policy manifest row {n_rows} mixes sim_id values within one window in '{manifest_path}'."
                        )
                    for k in range(1, self.T):
                        if int(days[k]) != int(days[0]) + k:
                            raise RuntimeError(
                                f"Non-contiguous policy-manifest window at row {n_rows} in '{manifest_path}': got days {days}"
                            )

                file_keys: List[str] = []
                for p in window_paths:
                    key = os.path.abspath(p)
                    self._disk_paths[key] = key
                    file_keys.append(key)

                labels_dict: Dict[str, torch.Tensor] = {}
                labels_dict["action_features"] = _build_action_features_from_manifest_row(row)

                for key, value in row.items():
                    t = _tensor_from_manifest_value(key, value)
                    if t is not None:
                        labels_dict[key] = t

                self.samples.append(
                    {
                        "file_keys": file_keys,
                        "labels_dict": labels_dict,
                        "row_meta": dict(row),
                    }
                )
                n_kept += 1

        if n_kept == 0:
            raise RuntimeError(f"No usable policy-manifest rows were loaded from '{manifest_path}'.")

        print(
            f"🧩 AMR dataset (policy-manifest): {n_kept} windows from manifest '{manifest_path}' "
            f"(read_rows={n_rows}, T={self.T})",
            flush=True,
        )

    def _iter_state_summary_calibration_reference_keys(self) -> List[str]:
        if self.use_policy_manifest:
            train_keys: List[str] = []
            fallback_keys: List[str] = []
            seen_train = set()
            seen_all = set()

            for sample in self.samples:
                file_keys = sample.get("file_keys", [])
                if not file_keys:
                    continue
                final_key = os.path.abspath(str(file_keys[-1]))
                if final_key not in seen_all:
                    fallback_keys.append(final_key)
                    seen_all.add(final_key)

                row_meta = sample.get("row_meta", {}) or {}
                split_raw = str(row_meta.get("split", "")).strip().lower()
                split_norm = "validation" if split_raw == "val" else split_raw
                if split_norm == "train" and final_key not in seen_train:
                    train_keys.append(final_key)
                    seen_train.add(final_key)

            return train_keys if train_keys else fallback_keys

        return sorted(self._disk_paths.keys(), key=natural_key)

    def _init_state_summary_calibration(self) -> None:
        self.state_summary_calibration = _default_state_summary_calibration()

        reference_keys = self._iter_state_summary_calibration_reference_keys()
        if len(reference_keys) <= 0:
            return

        num_nodes_logs: List[float] = []
        edge_count_logs: List[float] = []
        mean_out_degree_logs: List[float] = []

        for key in reference_keys:
            path = self._disk_paths.get(key, key)
            try:
                data = torch.load(path, weights_only=False)
            except Exception:
                continue

            payload = _graph_summary_calibration_payload(data)
            num_nodes_logs.append(_safe_log1p(payload.get("num_nodes", 0.0)))
            edge_count_logs.append(_safe_log1p(payload.get("edge_count", 0.0)))
            mean_out_degree_logs.append(_safe_log1p(payload.get("mean_out_degree", 0.0)))

        if len(num_nodes_logs) <= 0:
            return

        q = float(_STATE_SUMMARY_SCALE_QUANTILE)

        def _quantile(values: List[float]) -> float:
            vals = torch.tensor(values, dtype=torch.float32)
            try:
                ref = float(torch.quantile(vals, q).item())
            except Exception:
                ref = float(vals.max().item()) if vals.numel() > 0 else 1.0
            if not math.isfinite(ref) or ref <= 1e-12:
                ref = 1.0
            return float(ref)

        reference_split = "all"
        if self.use_policy_manifest:
            has_train = any(
                str((sample.get("row_meta", {}) or {}).get("split", "")).strip().lower() == "train"
                for sample in self.samples
            )
            reference_split = "train" if has_train else "all"

        self.state_summary_calibration = {
            "num_nodes_log_q": _quantile(num_nodes_logs),
            "edge_count_log_q": _quantile(edge_count_logs),
            "mean_out_degree_log_q": _quantile(mean_out_degree_logs),
            "quantile": q,
            "reference_split": reference_split,
            "n_reference_graphs": int(len(num_nodes_logs)),
        }

        print(
            "✅ State-summary calibration "
            f"(split={reference_split}, q={q:.2f}, n_graphs={len(num_nodes_logs)}, "
            f"num_nodes_log_q={self.state_summary_calibration['num_nodes_log_q']:.4f}, "
            f"edge_count_log_q={self.state_summary_calibration['edge_count_log_q']:.4f}, "
            f"mean_out_degree_log_q={self.state_summary_calibration['mean_out_degree_log_q']:.4f})",
            flush=True,
        )

    def _scan_folder(self):
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(f"Folder not found: {self.folder}")

        files = [f for f in os.listdir(self.folder) if f.endswith(self.file_ext)]
        files.sort(key=natural_key)

        for f in files:
            self._disk_paths[f] = os.path.join(self.folder, f)

        use_metadata = False

        if self.prefer_pt_metadata or self.require_pt_metadata:
            any_meta = False
            all_meta = True
            meta_cache: Dict[str, Tuple[Optional[str], Optional[int]]] = {}

            for f in files:
                sim_id, day = _read_pt_metadata(self._disk_paths[f])
                meta_cache[f] = (sim_id, day)
                if sim_id is not None and day is not None:
                    any_meta = True
                else:
                    all_meta = False

            if self.require_pt_metadata and not all_meta:
                missing = [f for f, (sid, d) in meta_cache.items() if sid is None or d is None]
                ex = missing[0] if missing else "unknown"
                raise RuntimeError(
                    f"PT metadata required but missing for {len(missing)}/{len(files)} files "
                    f"(example: {ex}). Ensure convert_to_pt.py writes Data.sim_id and Data.day."
                )

            use_metadata = bool(any_meta) if self.prefer_pt_metadata else False
        else:
            meta_cache = {}

        if use_metadata:
            sim_groups: Dict[str, List[Tuple[int, str]]] = {}
            skipped_no_meta = 0
            for f in files:
                sim_id, day = meta_cache.get(f, (None, None))
                if sim_id is None or day is None:
                    skipped_no_meta += 1
                    continue
                sim_groups.setdefault(str(sim_id), []).append((int(day), f))

            n_nonempty = 0
            n_skipped_windows = 0

            for sim_id, tf in sim_groups.items():
                if not tf:
                    continue
                n_nonempty += 1
                tf.sort(key=lambda z: int(z[0]))

                ds = [int(d) for d, _ in tf]
                fs = [f for _, f in tf]

                for start in range(0, len(fs) - self.T + 1, self.sliding_step):
                    d0 = ds[start]
                    ok = True
                    for k in range(1, self.T):
                        if ds[start + k] != d0 + k:
                            ok = False
                            break
                    if not ok:
                        n_skipped_windows += 1
                        if self.fail_on_noncontiguous:
                            raise RuntimeError(
                                f"Non-contiguous window detected (metadata mode) in '{self.folder}' "
                                f"sim_id='{sim_id}' start_day={d0} expected={list(range(d0, d0 + self.T))} "
                                f"got={ds[start:start + self.T]}"
                            )
                        continue
                    self.groups.append(fs[start: start + self.T])

            if skipped_no_meta > 0:
                print(
                    f"⚠️ Skipped {skipped_no_meta} files missing Data.sim_id/Data.day in '{self.folder}'",
                    flush=True,
                )
            if n_skipped_windows > 0 and not self.fail_on_noncontiguous:
                print(
                    f"⚠️ Skipped {n_skipped_windows} non-contiguous windows (missing days) in '{self.folder}'",
                    flush=True,
                )

            print(
                f"🧩 AMR dataset (metadata): {len(self.groups)} windows from {n_nonempty} trajectories in '{self.folder}'",
                flush=True,
            )
            return

        prefix_groups: Dict[str, List[Tuple[int, str]]] = {}
        skipped = 0
        for f in files:
            parsed = _parse_sim_day_label(f, self.file_ext)
            if parsed is None:
                skipped += 1
                continue
            prefix, t, _ = parsed
            prefix_groups.setdefault(prefix, []).append((int(t), f))

        n_nonempty = 0
        n_skipped_windows = 0
        for prefix, tf in prefix_groups.items():
            if not tf:
                continue
            n_nonempty += 1
            tf.sort(key=lambda z: int(z[0]))

            ts = [int(t) for t, _ in tf]
            fs = [f for _, f in tf]

            for start in range(0, len(fs) - self.T + 1, self.sliding_step):
                t0 = ts[start]
                ok = True
                for k in range(1, self.T):
                    if ts[start + k] != t0 + k:
                        ok = False
                        break
                if not ok:
                    n_skipped_windows += 1
                    if self.fail_on_noncontiguous:
                        raise RuntimeError(
                            f"Non-contiguous window detected (filename mode) in '{self.folder}' "
                            f"prefix='{prefix}' start_t={t0} expected={list(range(t0, t0 + self.T))} "
                            f"got={ts[start:start + self.T]}"
                        )
                    continue
                self.groups.append(fs[start: start + self.T])

        if skipped > 0:
            print(
                f"⚠️ Skipped {skipped} files that do not match '*_t<day>[ _L<label> ]{self.file_ext}'",
                flush=True,
            )
        if n_skipped_windows > 0 and not self.fail_on_noncontiguous:
            print(
                f"⚠️ Skipped {n_skipped_windows} non-contiguous windows (missing days) in '{self.folder}'",
                flush=True,
            )

        print(
            f"🧩 AMR dataset (filename): {len(self.groups)} windows from {n_nonempty} simulations in '{self.folder}'",
            flush=True,
        )

    def _load_vocab_from_disk(self) -> bool:
        if not os.path.isfile(self.node_vocab_path):
            return False
        try:
            with open(self.node_vocab_path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            vocab = obj.get("node_vocab", None)
            inv = obj.get("node_vocab_inv", None)
            if not isinstance(vocab, dict) or not isinstance(inv, list):
                return False

            self.node_vocab = {str(k): int(v) for k, v in vocab.items()}
            self.node_vocab_inv = [str(x) for x in inv]

            if len(self.node_vocab_inv) == 0 or len(self.node_vocab) == 0:
                return False

            return True
        except Exception:
            return False

    def _save_vocab_to_disk(self) -> None:
        try:
            obj = {
                "node_vocab": self.node_vocab,
                "node_vocab_inv": self.node_vocab_inv,
            }
            with open(self.node_vocab_path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, sort_keys=True)
        except Exception:
            pass

    def _iter_all_pt_keys_for_vocab(self) -> List[str]:
        return sorted(self._disk_paths.keys(), key=natural_key)

    def _init_node_vocab(self) -> None:
        if self._load_vocab_from_disk():
            print(
                f"✅ Loaded node vocabulary from '{self.node_vocab_path}' ({len(self.node_vocab)} nodes).",
                flush=True,
            )
            return

        vocab: Dict[str, int] = {}
        inv: List[str] = []

        for file_key in self._iter_all_pt_keys_for_vocab():
            path = self._disk_paths[file_key]
            try:
                data = torch.load(path, weights_only=False)
            except Exception:
                continue

            node_names = getattr(data, "node_names", None)
            if node_names is None:
                try:
                    n = int(getattr(data, "num_nodes", 0))
                except Exception:
                    n = 0
                node_names = [str(i) for i in range(n)]

            for name in node_names:
                key = str(name)
                if key not in vocab:
                    vocab[key] = len(inv)
                    inv.append(key)

        self.node_vocab = vocab
        self.node_vocab_inv = inv

        if len(self.node_vocab) == 0:
            print(
                "⚠️ Node vocabulary build produced 0 entries; node_id will fall back to positional indices.",
                flush=True,
            )
        else:
            print(
                f"✅ Built node vocabulary for dataset root '{self.folder}' ({len(self.node_vocab)} nodes).",
                flush=True,
            )

        self._save_vocab_to_disk()

    def _attach_node_ids(self, data) -> None:
        node_names = getattr(data, "node_names", None)
        if node_names is None:
            try:
                n = int(getattr(data, "num_nodes", 0))
            except Exception:
                n = 0
            node_names = [str(i) for i in range(n)]
            data.node_names = node_names

        if not self.node_vocab:
            data.node_id = torch.arange(len(node_names), dtype=torch.long)
            return

        ids: List[int] = []
        vocab_changed = False
        for name in node_names:
            key = str(name)
            if key not in self.node_vocab:
                self.node_vocab[key] = len(self.node_vocab_inv)
                vocab_changed = True
                self.node_vocab_inv.append(key)
            ids.append(self.node_vocab[key])

        data.node_id = torch.tensor(ids, dtype=torch.long)

        if vocab_changed:
            self._save_vocab_to_disk()

    def __len__(self) -> int:
        return len(self.samples) if self.use_policy_manifest else len(self.groups)

    def _load_graph(self, file_key: str):
        if file_key in self._cache:
            self._cache.move_to_end(file_key)
            return self._cache[file_key]

        if file_key not in self._disk_paths:
            raise KeyError(f"Missing file in index: {file_key}")

        data = torch.load(self._disk_paths[file_key], weights_only=False)
        _upgrade_legacy_operational_context_features(data)

        if self.build_node_vocab:
            self._attach_node_ids(data)

        self._cache[file_key] = data

        if not self.cache_all and len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

        return data

    def __getitem__(self, idx: int):
        if self.use_policy_manifest:
            sample = self.samples[idx]
            graphs = [self._load_graph(k) for k in sample["file_keys"]]

            labels_dict = {
                k: (v.clone() if torch.is_tensor(v) else v)
                for k, v in sample["labels_dict"].items()
            }

            row_meta = sample.get("row_meta", {}) or {}

            labels_dict["state_id"] = str(row_meta.get("state_id", "")).strip()
            labels_dict["action_id"] = str(row_meta.get("action_id", "")).strip()
            labels_dict["pair_id"] = str(row_meta.get("pair_id", "")).strip()
            labels_dict["split"] = str(row_meta.get("split", "")).strip()
            labels_dict["action_name"] = str(row_meta.get("action_name", "")).strip()

            state_summary_features = _build_state_summary_features_from_graph(
                graphs[-1],
                calibration=self.state_summary_calibration,
            )
            labels_dict["state_summary_features"] = state_summary_features.clone()

            action_features = labels_dict.get("action_features", None)
            if torch.is_tensor(action_features):
                clean_action_features = action_features.to(torch.float32).view(-1)
                labels_dict["action_features"] = clean_action_features
                for g in graphs:
                    g.action_features = clean_action_features.clone()
                    g.state_summary_features = state_summary_features.clone()
            else:
                for g in graphs:
                    g.state_summary_features = state_summary_features.clone()

            for g in graphs:
                g.state_id = labels_dict["state_id"]
                g.action_id = labels_dict["action_id"]
                g.pair_id = labels_dict["pair_id"]

            return graphs, labels_dict

        fnames = self.groups[idx]
        graphs = [self._load_graph(f) for f in fnames]
        labels_dict: Dict[str, torch.Tensor] = {}
        return graphs, labels_dict


def collate_temporal_graph_batch(batch):
    graphs_list, labels_list = zip(*batch)
    T = len(graphs_list[0])

    from torch_geometric.data import Batch as PyGBatch

    graphs_per_t: List[List[object]] = [[] for _ in range(T)]
    for graphs in graphs_list:
        for t, g in enumerate(graphs):
            graphs_per_t[t].append(g)

    batched_graphs = [PyGBatch.from_data_list(graphs_per_t[t]) for t in range(T)]

    merged: Dict[str, Any] = {}
    if labels_list and labels_list[0]:
        common_keys = set(labels_list[0].keys())
        for d in labels_list[1:]:
            common_keys &= set(d.keys())

        for k in sorted(common_keys):
            vals = [d[k] for d in labels_list]

            if all(torch.is_tensor(v) for v in vals):
                try:
                    merged[k] = torch.stack(vals, dim=0)
                    continue
                except Exception:
                    pass

            merged[k] = list(vals)

    return batched_graphs, merged
