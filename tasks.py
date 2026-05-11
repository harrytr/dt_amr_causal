#!/usr/bin/env python3
"""
tasks.py
========

AMR-only task registry.

Tasks read graph-level labels from the LAST graph in the window (graphs[-1]),
which are stored by convert_to_pt.py as attributes like:
  last.y_h7_cr_acq  (Tensor shape [B,1] after batching)

IMPORTANT:
- During training: graphs = list[T] of Batch
- During evaluation: graphs = list[list[T]] (list of windows)

All tasks MUST handle both cases.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import re
import torch
import torch.nn.functional as F


# =============================================================================
# Base classes
# =============================================================================

class BaseTask:
    name: str
    out_dim: int
    is_classification: bool = False

    def __init__(self, name: str, out_dim: int):
        self.name = str(name)
        self.out_dim = int(out_dim)

        # model hyperparameters (can be overridden)
        self.model_config: Dict = {
            "hidden_channels": 64,
            "heads": 2,
            "dropout": 0.2,
            "transformer_layers": 2,
            "sage_layers": 3,
            "use_cls_token": False,
        }

        # training hyperparameters (can be overridden)
        self.train_config: Dict = {
            "batch_size": 16,
            "epochs": 20,
            "lr": 1e-4,
            "max_neighbors": 20,
        }

        # Output activation for the model head.
        # - classification tasks: identity (raw logits)
        # - regression tasks: softplus by default (nonnegative), override per task if needed
        self.output_activation: str = (
            "identity" if bool(getattr(self, "is_classification", False)) else "softplus"
        )

    def get_targets(self, graphs: List, labels_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        graphs: List,
        labels_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        y_true = self.get_targets(graphs, labels_dict)
        return F.mse_loss(y_hat, y_true)

    def compute_eval_metrics(
        self,
        y_hat_all: torch.Tensor,
        graphs_list: List,
        labels_dict_all: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        y_true = self.get_targets(graphs_list, labels_dict_all)
        mse = F.mse_loss(y_hat_all, y_true).item()
        rmse = mse ** 0.5
        mae = F.l1_loss(y_hat_all, y_true).item()
        return {"mse": mse, "rmse": rmse, "mae": mae}


class RegressionTask(BaseTask):
    pass


class ClassificationTask(BaseTask):
    is_classification: bool = True

    def __init__(self, name: str, num_classes: int = 2):
        super().__init__(name, out_dim=num_classes)
        self.is_classification = True
        self.model_config["use_cls_token"] = True
        self.output_activation = "identity"

    def compute_loss(
        self,
        y_hat: torch.Tensor,
        graphs: List,
        labels_dict: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        y_true = self.get_targets(graphs, labels_dict)
        return F.cross_entropy(y_hat, y_true)

    def compute_eval_metrics(
        self,
        y_hat_all: torch.Tensor,
        graphs_list: List,
        labels_dict_all: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

        y_true = self.get_targets(graphs_list, labels_dict_all).detach().cpu().numpy().astype(int).reshape(-1)
        logits = y_hat_all.detach().cpu()

        if logits.numel() == 0 or y_true.size == 0:
            return {}

        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(axis=1)

        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, preds)),
        }

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true,
            preds,
            average="macro",
            zero_division=0,
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true,
            preds,
            average="weighted",
            zero_division=0,
        )

        metrics.update(
            {
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "precision_weighted": float(precision_weighted),
                "recall_weighted": float(recall_weighted),
                "f1_weighted": float(f1_weighted),
            }
        )

        try:
            unique_classes = sorted(set(int(x) for x in y_true.tolist()))
            if len(unique_classes) >= 2:
                if self.out_dim == 2:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, probs[:, 1]))
                else:
                    metrics["roc_auc_macro_ovr"] = float(
                        roc_auc_score(
                            y_true,
                            probs[:, : self.out_dim],
                            multi_class="ovr",
                            average="macro",
                        )
                    )
        except Exception:
            pass

        return metrics


# =============================================================================
# Helper: extract last graph robustly
# =============================================================================

def _get_last_graphs(graphs: List) -> List:
    """
    Returns a list of last-step Batch objects, one per window.

    Handles:
      - graphs = list[T]              (single window)
      - graphs = list[list[T]]        (list of windows)
    """
    if len(graphs) == 0:
        return []

    if isinstance(graphs[0], list):
        return [window[-1] for window in graphs]

    return [graphs[-1]]


def _require_attr(obj, attr: str):
    val = getattr(obj, attr, None)
    if val is None:
        raise AttributeError(f"Missing label '{attr}'")
    return val


def _require_label(labels_dict: Dict[str, torch.Tensor], obj, attr: str):
    """
    Prefer policy-manifest labels when present, otherwise fall back to graph attrs.
    This preserves existing folder-based training while enabling manifest-native targets.
    """
    val = labels_dict.get(attr, None)
    if val is not None:
        return val
    return _require_attr(obj, attr)


# =============================================================================
# Horizon-parameterised task templates (NEW, backward compatible)
# =============================================================================

class _HorizonMixin:
    """Adds a numeric prediction horizon H and convenience for attribute naming."""
    horizon: int

    def __init__(self, horizon: int):
        self.horizon = int(horizon)

    def _h_attr(self, suffix: str) -> str:
        # suffix example: "cr_acq" -> "y_h7_cr_acq"
        return f"y_h{self.horizon}_{suffix}"




def _labels_dict_get_string_list(labels_dict: Dict[str, torch.Tensor], key: str, n: int) -> List[str]:
    raw = labels_dict.get(key, None)
    if raw is None:
        return ["" for _ in range(int(n))]
    if isinstance(raw, (list, tuple)):
        out = [str(x) for x in raw]
        if len(out) == int(n):
            return out
        if len(out) < int(n):
            return out + [""] * (int(n) - len(out))
        return out[: int(n)]
    if torch.is_tensor(raw):
        flat = raw.detach().cpu().view(-1).tolist()
        out = [str(x) for x in flat]
        if len(out) == int(n):
            return out
    return [str(raw) for _ in range(int(n))]


def _labels_dict_get_long_tensor(labels_dict: Dict[str, torch.Tensor], key: str) -> Optional[torch.Tensor]:
    raw = labels_dict.get(key, None)
    if raw is None:
        return None
    if torch.is_tensor(raw):
        return raw.view(-1).long()
    if isinstance(raw, (list, tuple)):
        try:
            return torch.tensor([int(x) for x in raw], dtype=torch.long)
        except Exception:
            return None
    try:
        return torch.tensor([int(raw)], dtype=torch.long)
    except Exception:
        return None

# =============================================================================
# Existing AMR tasks (kept working via wrappers)
# =============================================================================

class AMR_CR_Acquisitions(RegressionTask, _HorizonMixin):
    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"amr_cr_acq_h{self.horizon}", out_dim=1)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("cr_acq")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class AMR_IR_Infections(RegressionTask, _HorizonMixin):
    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"amr_ir_inf_h{self.horizon}", out_dim=1)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("ir_inf")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class AMR_Outbreak_CR(ClassificationTask, _HorizonMixin):
    def __init__(self, horizon: int, threshold: int = 10):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"amr_outbreak_cr_h{self.horizon}", num_classes=2)
        self.threshold = int(threshold)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("cr_acq")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1))
        y_all = torch.cat(ys, dim=0)
        return (y_all >= float(self.threshold)).long()


# Wrappers to preserve the original class names (and exact behaviour)
class AMR_CR_Acquisitions_H7(AMR_CR_Acquisitions):
    def __init__(self):
        super().__init__(horizon=7)


class AMR_CR_Acquisitions_H14(AMR_CR_Acquisitions):
    def __init__(self):
        super().__init__(horizon=14)


class AMR_IR_Infections_H7(AMR_IR_Infections):
    def __init__(self):
        super().__init__(horizon=7)


class AMR_Outbreak_CR_H7(AMR_Outbreak_CR):
    def __init__(self, threshold: int = 10):
        super().__init__(horizon=7, threshold=threshold)


class TransmissionResistantBurden(RegressionTask, _HorizonMixin):
    """
    Regression: future transmission-driven resistant burden over horizon H.
    Reads y_h{H}_trans_res from the last graph in the window.
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"transmission_resistant_burden_h{self.horizon}", out_dim=1)

    def get_targets(self, graphs, labels_dict):
        attr = self._h_attr("trans_res")
        y_manifest = labels_dict.get(attr, None)
        if y_manifest is not None:
            return y_manifest.view(-1, 1).float()

        lasts = _get_last_graphs(graphs)
        ys = []
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class TransmissionResistantBurden_H7(TransmissionResistantBurden):
    def __init__(self):
        super().__init__(horizon=7)


class TransmissionResistantBurdenGain(RegressionTask, _HorizonMixin):
    """
    Regression: baseline-relative improvement in future transmission-driven resistant
    burden over horizon H. Positive values mean the action improves over baseline.
    Reads y_h{H}_trans_res_gain from policy-manifest labels when available and
    falls back to graph attrs if present.
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"transmission_resistant_burden_gain_h{self.horizon}", out_dim=1)
        self.output_activation = "identity"

    def get_targets(self, graphs, labels_dict):
        attr = self._h_attr("trans_res_gain")
        y_manifest = labels_dict.get(attr, None)
        if y_manifest is not None:
            return y_manifest.view(-1, 1).float()

        lasts = _get_last_graphs(graphs)
        ys = []
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class TransmissionImportationResistantBurden(RegressionTask, _HorizonMixin):
    """
    Regression: future resistant burden over horizon H counting both
    transmission-driven and imported CR burden.
    Reads y_h{H}_trans_import_res from policy-manifest labels when available and
    falls back to graph attrs if present.
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"transmission_importation_resistant_burden_h{self.horizon}", out_dim=1)

    def get_targets(self, graphs, labels_dict):
        attr = self._h_attr("trans_import_res")
        y_manifest = labels_dict.get(attr, None)
        if y_manifest is not None:
            return y_manifest.view(-1, 1).float()

        lasts = _get_last_graphs(graphs)
        ys = []
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class TransmissionImportationResistantBurdenGain(RegressionTask, _HorizonMixin):
    """
    Regression: baseline-relative improvement in future resistant burden over
    horizon H counting both transmission-driven and imported CR burden.
    Positive values mean the action improves over baseline.
    Reads y_h{H}_trans_import_res_gain from policy-manifest labels when available
    and falls back to graph attrs if present.
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"transmission_importation_resistant_burden_gain_h{self.horizon}", out_dim=1)
        self.output_activation = "identity"

    def get_targets(self, graphs, labels_dict):
        attr = self._h_attr("trans_import_res_gain")
        y_manifest = labels_dict.get(attr, None)
        if y_manifest is not None:
            return y_manifest.view(-1, 1).float()

        lasts = _get_last_graphs(graphs)
        ys = []
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)



class OracleBestActionPolicyTask(ClassificationTask, _HorizonMixin):
    """
    Grouped multiclass policy-selection task over action-conditioned rows.

    Each manifest row corresponds to one candidate action for a shared decision
    state. The stable target is the configured oracle-best action index for that
    state. Training uses a state-wise softmax cross-entropy in the trainer, so
    this task intentionally exposes a scalar output head rather than a rowwise
    C-class head.
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        ClassificationTask.__init__(self, f"oracle_best_action_h{self.horizon}", num_classes=2)
        self.out_dim = 1
        self.output_activation = "identity"

    def get_targets(self, graphs, labels_dict):
        attr = f"oracle_best_action_index_h{self.horizon}"
        y_manifest = labels_dict.get(attr, None)
        if y_manifest is None:
            raise AttributeError(
                f"Missing grouped policy label '{attr}'. Rebuild the policy manifest with oracle action columns."
            )
        return y_manifest.view(-1).long()

    def compute_eval_metrics(self, y_hat_all, graphs_list, labels_dict_all):
        logits = y_hat_all.detach().view(-1).cpu()
        n = int(logits.numel())
        if n == 0:
            return {}

        oracle_idx = self.get_targets(graphs_list, labels_dict_all).detach().view(-1).cpu()
        action_idx = _labels_dict_get_long_tensor(labels_dict_all, "action_index")
        if action_idx is None:
            return {}
        action_idx = action_idx.view(-1).cpu()
        state_ids = _labels_dict_get_string_list(labels_dict_all, "state_id", n)

        groups: Dict[str, List[int]] = {}
        for i, sid in enumerate(state_ids):
            groups.setdefault(str(sid), []).append(i)

        n_states = 0
        n_match = 0
        regrets: List[float] = []
        for sid, idxs in groups.items():
            if len(idxs) < 2:
                continue
            idx_t = torch.tensor(idxs, dtype=torch.long)
            group_logits = logits.index_select(0, idx_t)
            group_action_idx = action_idx.index_select(0, idx_t)
            group_oracle_idx = oracle_idx.index_select(0, idx_t)
            valid = torch.isfinite(group_logits)
            if int(valid.sum().item()) < 2:
                continue
            group_logits = group_logits[valid]
            group_action_idx = group_action_idx[valid]
            group_oracle_idx = group_oracle_idx[valid]
            if group_logits.numel() < 2:
                continue

            order = torch.tensor(
                sorted(
                    range(int(group_action_idx.numel())),
                    key=lambda j: (int(group_action_idx[j].item()), int(j)),
                ),
                dtype=torch.long,
            )
            group_logits = group_logits.index_select(0, order)
            group_action_idx = group_action_idx.index_select(0, order)
            group_oracle_idx = group_oracle_idx.index_select(0, order)

            oracle_class = int(group_oracle_idx[0].item())
            if not bool(torch.all(group_oracle_idx == oracle_class).item()):
                continue

            oracle_matches = (group_action_idx == oracle_class).nonzero(as_tuple=False).view(-1)
            if int(oracle_matches.numel()) != 1:
                continue

            pred_local = int(torch.argmax(group_logits).item())
            pred_class = int(group_action_idx[pred_local].item())
            match = 1 if pred_class == oracle_class else 0
            n_states += 1
            n_match += match
            try:
                oracle_local = int(oracle_matches[0].item())
                regret = float(group_logits[oracle_local].item() - group_logits[pred_local].item())
                regrets.append(max(0.0, regret))
            except Exception:
                pass

        if n_states <= 0:
            return {}

        out = {
            "policy_accuracy": float(n_match / float(n_states)),
            "n_policy_states": float(n_states),
        }
        if regrets:
            out["mean_logit_regret"] = float(sum(regrets) / float(len(regrets)))
        return out


# =============================================================================
# NEW TASK FAMILY 1 — Resistance emergence (node → graph aggregated)
# =============================================================================

class PredictResistanceEmergence(ClassificationTask, _HorizonMixin):
    """
    Binary label: any NEW CR or IR emergence in next H days.
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"predict_resistance_emergence_h{self.horizon}", num_classes=2)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("any_res_emergence")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1))
        return torch.cat(ys, dim=0).long()


class PredictResistanceEmergence_H7(PredictResistanceEmergence):
    def __init__(self):
        super().__init__(horizon=7)


# =============================================================================
# NEW TASK FAMILY 2 — Staff mediation (counterfactual sensitivity)
# =============================================================================

class StaffMediationEffect(RegressionTask, _HorizonMixin):
    """
    Predict future infections; intended to be run
    with and without staff features.
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"staff_mediation_effect_h{self.horizon}", out_dim=1)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("total_inf")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class StaffMediationEffect_H7(StaffMediationEffect):
    def __init__(self):
        super().__init__(horizon=7)


# =============================================================================
# NEW TASK FAMILY 3 — Early outbreak warning
# =============================================================================

class EarlyOutbreakWarning(ClassificationTask, _HorizonMixin):
    """
    Binary early-warning label (H days).

    Preferred source (produced by convert_to_pt.py):
      - last.y_h{H}_resistant_frac_cls  (0/1)

    Backward-compatible fallback:
      - last.y_h{H}_resistant_frac >= threshold
    """

    def __init__(self, horizon: int, threshold: float = 0.15):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"early_outbreak_warning_h{self.horizon}", num_classes=2)
        self.threshold = float(threshold)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []

        y_cls_attr = self._h_attr("resistant_frac_cls")
        y_cont_attr = self._h_attr("resistant_frac")

        for last in lasts:
            y_cls = getattr(last, y_cls_attr, None)
            if y_cls is not None:
                ys.append(y_cls.view(-1).long())
                continue

            y = getattr(last, y_cont_attr, None)
            if y is None:
                raise AttributeError(
                    f"Missing label '{y_cls_attr}' and '{y_cont_attr}'. Expected early-warning label."
                )
            ys.append((y.view(-1) >= self.threshold).long())

        return torch.cat(ys, dim=0)


class EarlyOutbreakWarning_H14(EarlyOutbreakWarning):
    def __init__(self, threshold: float = 0.15):
        super().__init__(horizon=14, threshold=threshold)


class EarlyOutbreakWarningFracRegression(RegressionTask, _HorizonMixin):
    """
    Regression target for future resistant fraction over the next H days.

    Uses:
      - last.y_h{H}_resistant_frac
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"early_outbreak_warning_frac_regression_h{self.horizon}", out_dim=1)
        self.output_activation = "sigmoid"

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("resistant_frac")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class EarlyOutbreakWarningFracRegression_H7(EarlyOutbreakWarningFracRegression):
    def __init__(self):
        super().__init__(horizon=7)


# =============================================================================
# NEW TASK FAMILY 4 — Antibiotic stewardship impact
# =============================================================================

class AntibioticImpactRanking(RegressionTask, _HorizonMixin):
    """
    Predict change in resistant burden if antibiotics are reduced.
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"antibiotic_impact_ranking_h{self.horizon}", out_dim=1)
        self.output_activation = "identity"

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("delta_res_if_abx_reduced")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class AntibioticImpactRanking_H7(AntibioticImpactRanking):
    def __init__(self):
        super().__init__(horizon=7)


# =============================================================================
# NEW TASK FAMILY 5 — Screening optimisation
# =============================================================================

class OptimalScreeningTarget(ClassificationTask, _HorizonMixin):
    """
    Binary: would screening today reveal hidden colonisation?
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"optimal_screening_target_h{self.horizon}", num_classes=2)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("screening_gain")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1))
        return torch.cat(ys, dim=0).long()


class OptimalScreeningTarget_H7(OptimalScreeningTarget):
    def __init__(self):
        super().__init__(horizon=7)


# =============================================================================
# NEW TASK FAMILY 6 — Transmission attribution
# =============================================================================

class TransmissionAttribution(RegressionTask):
    """
    Predict number of true transmission edges attributable
    to top-ranked edges (edge-level evaluation downstream).

    NOTE: This task is intentionally NOT horizon-suffixed by default.
    """

    def __init__(self):
        super().__init__("transmission_attribution", out_dim=1)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        for last in lasts:
            y = getattr(last, "y_true_transmissions", None)
            if y is None:
                raise AttributeError(
                    "Missing label 'y_true_transmissions'. "
                    "Expected ground-truth transmission count."
                )
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


# =============================================================================
# NEW TASK FAMILY 7 — Importation vs endogenous transmission (dynamic census)
# =============================================================================

class EndogenousTransmissionShare(RegressionTask, _HorizonMixin):
    """Regression: share of resistant acquisitions attributable to within-ward transmission
    over the next H days.

    Definition (computed in convert_to_pt.py):
      h{H}_trans_share = h{H}_trans_res / (h{H}_trans_res + h{H}_import_res)

    Note: selection is excluded from the denominator by design (primary endpoint).
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"endogenous_transmission_share_h{self.horizon}", out_dim=1)
        self.output_activation = "sigmoid"

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("trans_share")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class EndogenousTransmissionMajority(ClassificationTask, _HorizonMixin):
    """Classification: whether transmission dominates importation in the next H days.

    Label (computed in convert_to_pt.py):
      y_h{H}_trans_majority = 1 if y_h{H}_trans_share >= 0.5 else 0
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"endogenous_transmission_majority_h{self.horizon}", num_classes=2)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("trans_majority")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1))
        return torch.cat(ys, dim=0).long()


class EndogenousImportationShare(RegressionTask, _HorizonMixin):
    """Regression: share of resistant emergence attributable to endogenous mechanisms
    (transmission + selection) over the next H days.

    Definition (computed in convert_to_pt.py):
      h{H}_endog_res = h{H}_trans_res + h{H}_select_res
      h{H}_endog_share = h{H}_endog_res / (h{H}_endog_res + h{H}_import_res)
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"endogenous_importation_share_h{self.horizon}", out_dim=1)
        self.output_activation = "sigmoid"

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("endog_share")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class EndogenousImportationMajority(ClassificationTask, _HorizonMixin):
    """Classification: whether endogenous emergence (transmission + selection)
    dominates importation in the next H days.

    Label (computed in convert_to_pt.py):
      y_h{H}_endog_majority = 1 if y_h{H}_endog_share >= 0.5 else 0
    """

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"endogenous_importation_majority_h{self.horizon}", num_classes=2)

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("endog_majority")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1))
        return torch.cat(ys, dim=0).long()


class MechanismDecomposition_ImportShare(RegressionTask, _HorizonMixin):
    """Regression: importation share in a 3-way decomposition (importation/transmission/selection)."""

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"mechanism_import_share_h{self.horizon}", out_dim=1)
        self.output_activation = "sigmoid"

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("import_share")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


class MechanismDecomposition_SelectionShare(RegressionTask, _HorizonMixin):
    """Regression: selection share in a 3-way decomposition (importation/transmission/selection)."""

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        super().__init__(f"mechanism_selection_share_h{self.horizon}", out_dim=1)
        self.output_activation = "sigmoid"

    def get_targets(self, graphs, labels_dict):
        lasts = _get_last_graphs(graphs)
        ys = []
        attr = self._h_attr("select_share")
        for last in lasts:
            y = _require_attr(last, attr)
            ys.append(y.view(-1, 1).float())
        return torch.cat(ys, dim=0)


# Wrappers to preserve the original class names / registry keys
class EndogenousTransmissionShare_H7(EndogenousTransmissionShare):
    def __init__(self):
        super().__init__(horizon=7)


class EndogenousTransmissionMajority_H7(EndogenousTransmissionMajority):
    def __init__(self):
        super().__init__(horizon=7)


class EndogenousImportationShare_H7(EndogenousImportationShare):
    def __init__(self):
        super().__init__(horizon=7)


class EndogenousImportationMajority_H7(EndogenousImportationMajority):
    def __init__(self):
        super().__init__(horizon=7)


class EndogenousShare(EndogenousImportationShare):
    """Regression alias with a semantically direct task name for y_h{H}_endog_share."""

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        RegressionTask.__init__(self, f"endogenous_share_h{self.horizon}", out_dim=1)
        self.output_activation = "sigmoid"


class EndogenousMajority(EndogenousImportationMajority):
    """Classification alias with a semantically direct task name for y_h{H}_endog_majority."""

    def __init__(self, horizon: int):
        _HorizonMixin.__init__(self, horizon=horizon)
        ClassificationTask.__init__(self, f"endogenous_majority_h{self.horizon}", num_classes=2)


class EndogenousShare_H7(EndogenousShare):
    def __init__(self):
        super().__init__(horizon=7)


class EndogenousMajority_H7(EndogenousMajority):
    def __init__(self):
        super().__init__(horizon=7)


class MechanismDecomposition_ImportShare_H7(MechanismDecomposition_ImportShare):
    def __init__(self):
        super().__init__(horizon=7)


class MechanismDecomposition_SelectionShare_H7(MechanismDecomposition_SelectionShare):
    def __init__(self):
        super().__init__(horizon=7)


# =============================================================================
# Registry (unchanged keys)
# =============================================================================

TASK_REGISTRY: Dict[str, BaseTask] = {
    # Existing
    "amr_cr_acq_h7": AMR_CR_Acquisitions_H7(),
    "amr_cr_acq_h14": AMR_CR_Acquisitions_H14(),
    "amr_ir_inf_h7": AMR_IR_Infections_H7(),
    "amr_outbreak_cr_h7": AMR_Outbreak_CR_H7(threshold=10),
    "transmission_resistant_burden_h7": TransmissionResistantBurden_H7(),
    "transmission_resistant_burden_gain_h7": TransmissionResistantBurdenGain(horizon=7),
    "transmission_resistant_burden_gain_h14": TransmissionResistantBurdenGain(horizon=14),
    "transmission_importation_resistant_burden_h7": TransmissionImportationResistantBurden(horizon=7),
    "transmission_importation_resistant_burden_h14": TransmissionImportationResistantBurden(horizon=14),
    "transmission_importation_resistant_burden_gain_h7": TransmissionImportationResistantBurdenGain(horizon=7),
    "transmission_importation_resistant_burden_gain_h14": TransmissionImportationResistantBurdenGain(horizon=14),
    "oracle_best_action_h7": OracleBestActionPolicyTask(horizon=7),
    "oracle_best_action_h14": OracleBestActionPolicyTask(horizon=14),

    # New high-impact tasks
    "predict_resistance_emergence_h7": PredictResistanceEmergence_H7(),
    "staff_mediation_effect_h7": StaffMediationEffect_H7(),
    "early_outbreak_warning_h14": EarlyOutbreakWarning_H14(threshold=0.15),
    "early_outbreak_warning_frac_regression_h7": EarlyOutbreakWarningFracRegression_H7(),
    "antibiotic_impact_ranking_h7": AntibioticImpactRanking_H7(),
    "optimal_screening_target_h7": OptimalScreeningTarget_H7(),
    "transmission_attribution": TransmissionAttribution(),

    # Dynamic census: importation vs transmission
    "endogenous_transmission_share_h7": EndogenousTransmissionShare_H7(),
    "endogenous_transmission_majority_h7": EndogenousTransmissionMajority_H7(),

    # Dynamic census: endogenous (transmission + selection) vs importation
    "endogenous_importation_share_h7": EndogenousImportationShare_H7(),
    "endogenous_importation_majority_h7": EndogenousImportationMajority_H7(),

    # Semantically direct aliases for the same endogenous labels
    "endogenous_share_h7": EndogenousShare_H7(),
    "endogenous_majority_h7": EndogenousMajority_H7(),

    "mechanism_import_share_h7": MechanismDecomposition_ImportShare_H7(),
    "mechanism_selection_share_h7": MechanismDecomposition_SelectionShare_H7(),
}


# =============================================================================
# Dynamic task factory (NEW): request amr_cr_acq_h{H}, etc.
# =============================================================================

_HORIZON_BUILDERS: Dict[str, Callable[[int], BaseTask]] = {
    "amr_cr_acq": lambda h: AMR_CR_Acquisitions(horizon=h),
    "amr_ir_inf": lambda h: AMR_IR_Infections(horizon=h),
    "amr_outbreak_cr": lambda h: AMR_Outbreak_CR(horizon=h, threshold=10),
    "transmission_resistant_burden": lambda h: TransmissionResistantBurden(horizon=h),
    "transmission_resistant_burden_gain": lambda h: TransmissionResistantBurdenGain(horizon=h),
    "transmission_importation_resistant_burden": lambda h: TransmissionImportationResistantBurden(horizon=h),
    "transmission_importation_resistant_burden_gain": lambda h: TransmissionImportationResistantBurdenGain(horizon=h),
    "oracle_best_action": lambda h: OracleBestActionPolicyTask(horizon=h),

    "predict_resistance_emergence": lambda h: PredictResistanceEmergence(horizon=h),
    "staff_mediation_effect": lambda h: StaffMediationEffect(horizon=h),
    "early_outbreak_warning": lambda h: EarlyOutbreakWarning(horizon=h, threshold=0.15),
    "early_outbreak_warning_frac_regression": lambda h: EarlyOutbreakWarningFracRegression(horizon=h),
    "antibiotic_impact_ranking": lambda h: AntibioticImpactRanking(horizon=h),
    "optimal_screening_target": lambda h: OptimalScreeningTarget(horizon=h),

    "endogenous_transmission_share": lambda h: EndogenousTransmissionShare(horizon=h),
    "endogenous_transmission_majority": lambda h: EndogenousTransmissionMajority(horizon=h),
    "endogenous_importation_share": lambda h: EndogenousImportationShare(horizon=h),
    "endogenous_importation_majority": lambda h: EndogenousImportationMajority(horizon=h),
    "endogenous_share": lambda h: EndogenousShare(horizon=h),
    "endogenous_majority": lambda h: EndogenousMajority(horizon=h),
    "mechanism_import_share": lambda h: MechanismDecomposition_ImportShare(horizon=h),
    "mechanism_selection_share": lambda h: MechanismDecomposition_SelectionShare(horizon=h),
}



# =============================================================================
# Policy-selection helpers
# =============================================================================

def infer_policy_target_name(task_name: str) -> str:
    """
    Infer the canonical continuous manifest target associated with a policy task.
    Returns an empty string when the task is not a recognized policy-selection task.
    """
    task_norm = str(task_name).strip().lower()
    m = re.search(r"_h(\d+)$", task_norm)
    h = int(m.group(1)) if m else 7

    if task_norm.startswith("transmission_importation_resistant_burden_gain_h"):
        return f"y_h{h}_trans_import_res_gain"
    if task_norm.startswith("transmission_importation_resistant_burden_h"):
        return f"y_h{h}_trans_import_res"
    if task_norm.startswith("transmission_resistant_burden_gain_h"):
        return f"y_h{h}_trans_res_gain"
    if task_norm.startswith("transmission_resistant_burden_h"):
        return f"y_h{h}_trans_res"
    if task_norm.startswith("endogenous_importation_majority_h") or task_norm.startswith("endogenous_importation_share_h"):
        return f"y_h{h}_endog_share"
    if task_norm.startswith("endogenous_transmission_majority_h") or task_norm.startswith("endogenous_transmission_share_h"):
        return f"y_h{h}_trans_share"
    if task_norm.startswith("mechanism_import_share_h"):
        return f"y_h{h}_import_share"
    if task_norm.startswith("mechanism_selection_share_h"):
        return f"y_h{h}_select_share"
    if task_norm.startswith("amr_cr_acq_h"):
        return f"y_h{h}_cr_acq"
    if task_norm.startswith("amr_ir_inf_h"):
        return f"y_h{h}_ir_inf"
    if task_norm.startswith("predict_resistance_emergence_h"):
        return f"y_h{h}_any_res_emergence"
    if task_norm.startswith("early_outbreak_warning_h"):
        return f"y_h{h}_resistant_frac"
    if task_norm.startswith("optimal_screening_target_h"):
        return f"y_h{h}_screening_gain"
    if task_norm.startswith("staff_mediation_effect_h"):
        return f"y_h{h}_transmissions"
    if task_norm.startswith("oracle_best_action_h"):
        return f"y_h{h}_trans_import_res_gain"
    return ""


def infer_policy_target_direction(task_name: str, target_name: str = "") -> str:
    """
    Return whether higher or lower values are better for a policy target.
    """
    task_norm = str(task_name).strip().lower()
    name = str(target_name).strip().lower()

    if name.endswith("_gain"):
        return "maximize"

    if "burden_gain" in task_norm:
        return "maximize"

    return "minimize"


def is_policy_selector_task(task_name: str) -> bool:
    return infer_policy_target_name(task_name) != ""

def get_task(task_name: str) -> BaseTask:
    """
    Returns a task by name.

    Backward compatible:
      - If task_name is in TASK_REGISTRY, returns it.

    New:
      - If task_name matches '{base}_h{H}', dynamically instantiates
        a horizon-parameterised task (if base is supported).
    """
    task_name = str(task_name)

    if task_name in TASK_REGISTRY:
        return TASK_REGISTRY[task_name]

    m = re.match(r"^(?P<base>.+)_h(?P<h>\d+)$", task_name)
    if not m:
        raise KeyError(
            f"Unknown task '{task_name}'. Not in TASK_REGISTRY and not of form '<base>_h<H>'."
        )

    base = m.group("base")
    h = int(m.group("h"))

    builder = _HORIZON_BUILDERS.get(base, None)
    if builder is None:
        raise KeyError(
            f"Unknown horizonised task base '{base}'. "
            f"Supported bases: {sorted(_HORIZON_BUILDERS.keys())}"
        )

    return builder(h)
