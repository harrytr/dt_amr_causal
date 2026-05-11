# AMR Digital Twin (Causal): Intervention-Conditioned Policy Learning and Constrained Decision Support

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch%20Geometric](https://img.shields.io/badge/PyTorch%20Geometric-enabled-orange)
![Pyomo](https://img.shields.io/badge/Pyomo-MILP-green)
![Workflow](https://img.shields.io/badge/Workflow-Reproducible-success)

A reproducible research platform for constructing intervention-conditioned causal-policy datasets from hospital antimicrobial resistance (AMR) simulations, training temporal graph neural networks to score candidate actions from shared observed histories, evaluating learned action ranking against simulator-defined oracle actions, and optimizing feasible intervention choices through a constrained mixed-integer linear programming layer.

This repository integrates four connected layers:

1. **Mechanistic hospital simulation** under surveillance, isolation, seasonality, admission importation, and superspreader effects.
2. **Causal policy dataset construction** from matched stochastic simulator reruns with a common baseline pre-intervention window for each decision state.
3. **Action-conditioned temporal graph learning** for oracle-best-action classification and within-state ranking over baseline-relative improvement targets.
4. **Constrained policy optimization** through a MILP layer operating on learned state-action utilities.

The current recommended reproducibility entrypoint is:

```bash
python experiments_causal.py --run_both_state_modes --emit_latex
```

---

## Contents

- [Overview](#overview)
- [Core capabilities](#core-capabilities)
- [Implemented causal design](#implemented-causal-design)
- [Digital-twin simulation update](#digital-twin-simulation-update)
- [Repository layout](#repository-layout)
- [Environment reproduction](#environment-reproduction)
- [Quick start](#quick-start)
- [Full reproducibility pipeline](#full-reproducibility-pipeline)
- [Resuming and partial reruns](#resuming-and-partial-reruns)
- [Important command-line options](#important-command-line-options)
- [Current default causal configuration](#current-default-causal-configuration)
- [Outputs](#outputs)
- [Overleaf export](#overleaf-export)
- [Reproducibility and design notes](#reproducibility-and-design-notes)
- [Citation](#citation)

---

## Overview

This repository is the causal-policy branch of the AMR digital-twin workflow. It is designed to answer a different question from the predictive benchmark. Instead of asking only what mechanism is likely to dominate future resistant emergence, it asks:

**given the same observed hospital history, which candidate intervention is predicted to produce the best downstream outcome?**

The platform builds causal policy datasets by extracting shared pre-intervention temporal graph windows and pairing each decision state with a finite action library. Each resulting sample contains:

- a common graph-history window of length `T`,
- a specific candidate action descriptor,
- an operational-state summary,
- and simulator-defined post-decision outcomes over horizon `H`.

The current branch-at-decision implementation constructs the action-conditioned futures through matched simulator reruns rather than checkpoint restoration. For each seed and decision day, the baseline pre-intervention window is reused as the supervised input for all actions, while post-decision action effects are activated from the following day.

The repository supports the same two state-observation tracks used in the broader platform:

- `ground_truth`
- `partial_observation`

The reproducibility workflow and CLI are defined by `experiments_causal.py`.

---

## Core capabilities

### 1) Mechanistic AMR hospital simulation for causal-policy data

The causal workflow reuses the hospital AMR simulator under a temporal-heterogeneity regime intended for intervention analysis.

The simulation layer supports:

- patients and staff as distinct node types,
- directed, weighted contact graphs,
- ward assignment, home-ward structure, and multi-ward staffing,
- admission-based importation pressure,
- within-hospital transmission with individual transmission-event tracking,
- bacterial-type inheritance from infectious source agents for successful transmission events,
- competing same-day exposure resolution using antibiotic context,
- infection-state-dependent infectivity through configurable beta multipliers,
- antibiotic exposure and within-host selection effects,
- leaky cross-ward staff hand hygiene for staff-patient contacts outside staff home wards,
- screening and isolation interventions, including symptomatic-only admission screening, active screening on C-to-I progression, and configurable resistant-only versus all-positive isolation triggers,
- observation-limited regimes,
- temporal admission-importation shocks,
- superspreader dynamics,
- and matched stochastic reruns for action-conditioned policy-response samples.

The simulator writes daily GraphML snapshots together with trajectory metadata and event summaries used downstream for policy-target construction.

### 2) Causal policy dataset construction

`build_causal_policy_dataset.py` constructs the intervention-conditioned dataset used by the causal experiments.

For each base simulation seed and each selected decision day, it:

- extracts a temporal window of length `T`,
- treats that baseline window as the observed state at decision time,
- reruns the simulator under the baseline action and each candidate intervention,
- keeps interventions inactive before the decision point in branch-at-decision mode,
- activates intervention effects from day `d + 1` under the current default,
- matches action futures through the same base seed and shared causal noise seed,
- computes horizon-specific policy outcomes,
- and writes a policy manifest linking windows, actions, state summaries, splits, and targets.

This creates repeated state-action samples of the form:

```text
(shared baseline history window, action descriptor, operational summary) -> post-decision outcome
```

The current causal workflow is centered on horizon-7 baseline-relative improvement in future transmission-plus-importation resistant burden:

```text
y_h7_trans_import_res_gain
```

The implemented burden is:

```text
new_trans_cr_total + new_import_cr_total + new_import_ir_total
```

The gain is computed as:

```text
baseline burden - action burden
```

Positive gain therefore means that an action improves on the baseline action for the implemented burden definition. This target does not include selection burden or `new_ir_inf_total`.

### 3) Graph conversion and policy-manifest temporal datasets

`convert_to_pt.py` converts GraphML outputs into `.pt` graph objects for temporal graph learning.

`temporal_graph_dataset.py` supports the policy-manifest mode used by the causal branch, allowing the trainer to build temporal windows from manifest-defined state-action samples rather than only from standard forecasting trajectories.

Converted samples can store:

- node features,
- edge features,
- graph-level metadata,
- action descriptors,
- operational-state summaries,
- stable node identifiers for traceability,
- policy targets and auxiliary policy labels,
- and split information inherited from seed-level partitioning.

This preserves the causal design principle that all action-conditioned samples derived from the same underlying seed remain in the same split.

### 4) Action-conditioned temporal graph learning

`train_amr_dygformer.py` is used in action-conditioning mode to train the causal policy model.

The current causal stack combines:

- a daily graph encoder built around GraphSAGE-style message passing,
- a Transformer-based temporal encoder over windows of length `T = 7`,
- action conditioning through projected action descriptors,
- explicit operational-state features,
- a nonlinear state-action interaction head,
- a state-wise oracle-best-action loss,
- and an active within-state pairwise/listwise ranking regularizer on the continuous gain target.

The current causal driver is aligned to:

```text
task = oracle_best_action_h7
use_action_conditioning = true
oracle metric = y_h7_trans_import_res_gain
```

The model is trained to score candidate interventions from the same state so that the highest-scoring action should match the simulator-defined oracle-best action.

### 5) Held-out policy evaluation

`evaluate_policy_selector.py` evaluates the learned action-conditioned predictor on held-out decision states.

It compares the model-selected action against the simulator-defined oracle action using metrics such as:

- policy accuracy,
- top-k policy accuracy,
- regret,
- baseline improvement of selected actions,
- predicted-versus-oracle action summaries,
- and exported action-score tables.

The oracle-best action is defined from the manifest/simulator target, not from model predictions. Under the default configuration, the oracle is the complete-action-set member with the largest finite `y_h7_trans_import_res_gain` value.

### 6) Constrained MILP decision support

`optimize_policy_milp.py` turns held-out action scores into explicit feasible decisions.

The decision layer supports:

- one action per decision state,
- baseline-relative utility transforms,
- action-specific costs,
- optional budget constraints,
- maximum-use limits,
- cooldown constraints,
- and solver-based exact optimization through Pyomo.

The checked-in default MILP configuration uses `score_transform = baseline_delta` and enforces the configured action-family cooldown rules. Action costs are represented in `milp_policy_config.json`, but the default `budget_total` is `null`, so costs are not binding unless an explicit budget is supplied through the config or CLI.

The MILP layer is a downstream constrained selector over learned state-action utilities. It does not embed the mechanistic simulator inside the optimization loop.

### 7) End-to-end causal reproducibility workflow

`experiments_causal.py` orchestrates the full causal workflow, including:

- causal dataset building,
- action-conditioned model training,
- held-out policy evaluation,
- MILP policy optimization,
- dual-track execution across `ground_truth` and `partial_observation`,
- archive generation,
- and Overleaf-ready manuscript export.

---

## Implemented causal design

The current experiment design is summarized as follows.

| Step | Component | Implemented behavior |
|---|---|---|
| C1 | Base trajectory generation | Generates 60-day hospital trajectories under disjoint train, validation, and test seed ranges with patient-level CS/CR/IS/IR day-0 seeding, shock-modulated CS/CR/IS/IR admission importation, weekly baseline screening with a 3-day reporting delay, symptomatic admission testing, no universal admission screening, baseline isolation multiplier 0.8 for 3 days, and a superspreader episode from day 14 to day 21. |
| C2 | Decision-state extraction | Extracts pre-intervention windows of length `T = 7` for valid selected decision days. With `decision_days = 7:60:7`, `num_days = 60`, and `H = 7`, the valid decision days are 7, 14, 21, 28, 35, 42, and 49. |
| C3 | Action-conditioned counterfactual generation | Reruns the simulator under the baseline action and four nonbaseline screening-response actions. Each action future is matched by base seed and shared causal noise seed. The supervised input window for every action is fixed to the corresponding baseline pre-intervention window, and intervention effects are applied from day `d + 1`. |
| C4 | Outcome construction | Computes horizon-7 outcomes over days `d + 1` through `d + 7`. The main oracle metric is `y_h7_trans_import_res_gain`, defined as baseline burden minus action burden for `new_trans_cr_total + new_import_cr_total + new_import_ir_total`. |
| C5 | Split assignment without leakage | Assigns all action-conditioned samples derived from the same base seed to the same train, validation, or test split. |
| C6 | Intervention-conditioned prediction | Trains an action-conditioned temporal graph model using `oracle_best_action_h7`, state-wise oracle-best-action loss, and a within-state ranking regularizer on `y_h7_trans_import_res_gain`. |
| C7 | Policy-response evaluation | Evaluates predictions on held-out test decision states against simulator-defined oracle responses. Only complete action sets with finite oracle values are retained. |
| C8 | Constrained decision layer | Passes predicted action utilities to a MILP layer using baseline-relative utilities and configured feasibility rules. Cooldown rules are active by default; action costs become binding only when a budget is supplied. |

---

## Digital-twin simulation update

This version includes updated transmission, screening, and isolation mechanics in the causal-policy simulator. These changes affect policy-manifest outcomes and therefore require rebuilding the causal policy dataset from C1 before training or evaluating the policy selector. Old manifests or cached `.pt` files generated under earlier simulator semantics should not be reused for the current experiments.

### Transmission mechanics

- Successful infectious contacts are tracked as individual transmission events.
- When an agent acquires infection or carriage through infectious contact, the acquiring agent inherits the bacterial type of the transmitting source: resistant or sensitive.
- If multiple successful infectious contacts occur within a day, the final infecting type is resolved according to antibiotic context:
  - no antibiotics: sensitive exposure is favoured;
  - antibiotics: resistant exposure is favoured.
- Infectivity is infection-state dependent through configurable beta multipliers. The resistant global multiplier remains neutral (`etaR = 1`).
- Leaky hand hygiene is applied to cross-ward staff-patient contacts when staff interact outside their home ward.

### Screening and isolation semantics

- Baseline admission screening is symptomatic-only by default (`IS` or `IR`).
- Universal admission screening is disabled in the baseline simulator settings used by `experiments_causal.py`.
- Active screening is triggered when patients progress from colonisation to symptomatic infection.
- Isolation can be configured with two main trigger modes:
  - `resistant_only`,
  - `all_positives`.
- Isolation policies can be combined with different efficacies, durations, screening frequencies, and screening-result delays through `candidate_interventions.json`.

### Candidate intervention menu

The default action set contains the baseline action plus four nonbaseline hospital-wide screening-response actions.

The four nonbaseline actions are:

| Action family | Screening cadence | Result delay | Admission screening | Isolation trigger | Isolation multiplier | Isolation duration |
|---|---:|---:|---:|---|---:|---:|
| `set_screening_frequency` | 2 days | 0 days | enabled | `resistant_only` | 0.75 | 3 days |
| `set_screening_frequency` | 7 days | 2 days | enabled | `resistant_only` | 0.75 | 3 days |
| `set_screening_frequency` | 2 days | 0 days | enabled | `all_positives` | 0.75 | 3 days |
| `set_screening_frequency` | 7 days | 2 days | enabled | `all_positives` | 0.75 | 3 days |

This is not a full factorial over all cadence, delay, and trigger combinations. It is the checked-in four-action menu defined by `candidate_interventions.json`, plus the included baseline action.

### Explicitly not modelled in this version

- Bed/environment contamination is not included.
- Isolation does not physically transfer patients to a separate isolation ward; it is implemented as a contact/transmission modifier.
- The branch-at-decision workflow does not restore a saved simulator checkpoint at the decision state. It uses matched stochastic reruns with the intervention inactive before the decision point and with the baseline pre-intervention window used as the shared supervised input.

---

## Repository layout

### Core pipeline

- `experiments_causal.py` — reproducibility driver for the full causal workflow.
- `build_causal_policy_dataset.py` — intervention-conditioned dataset builder.
- `convert_to_pt.py` — GraphML-to-PyTorch-Geometric conversion.
- `train_amr_dygformer.py` — temporal graph learning trainer with action conditioning support.
- `temporal_graph_dataset.py` — temporal dataset construction including policy-manifest mode.
- `models_amr.py` — model definitions for the temporal graph architecture.
- `tasks.py` — registered forecasting and policy-learning targets.
- `evaluate_policy_selector.py` — held-out policy evaluation.
- `optimize_policy_milp.py` — constrained MILP decision layer.
- `candidate_interventions.json` — finite intervention library used for action-conditioned reruns.
- `milp_policy_config.json` — MILP configuration for action costs, cooldowns, and related constraints.
- `best_config.json` — checked-in training configuration override used by the driver when present.

### Utilities and support scripts

Additional utilities include:

- `causal_interventions.py` — action specification and intervention application logic.
- `causal_estimands.py` — causal target helpers.
- `counterfactual_rollout.py` — counterfactual rollout support utilities.
- `run_policy_feature_ablations.py` — policy-feature ablation runner.
- `train_simple_policy_baseline.py` — simple policy baseline training.
- `tune_policy_trainer.py` — policy-training tuning support.
- `verify_causal_branching.py` — diagnostic utility for older/alternative manifest layouts with explicit graph-path fields.

---

## Environment reproduction

The codebase expects a Python environment with PyTorch, PyTorch Geometric, and Pyomo support.

At minimum, confirm the availability of PyTorch and PyTorch Geometric:

```bash
python -c "import torch, torch_geometric; print('torch', torch.__version__); print('pyg', torch_geometric.__version__)"
```

For MILP support, also confirm Pyomo availability:

```bash
python -c "import pyomo; print('pyomo ok')"
```

If your environment uses sparse PyG extensions, verify them explicitly:

```bash
python -c "import torch, torch_geometric, torch_sparse, torch_scatter; print('torch', torch.__version__); print('pyg', torch_geometric.__version__); print('torch_sparse', torch_sparse.__version__); print('torch_scatter', torch_scatter.__version__)"
```

If you use CPLEX, ensure it is installed and accessible from the environment used to run `optimize_policy_milp.py`. The checked-in driver points to a local CPLEX executable path; adjust `CONFIG["MILP"]["cplex_executable"]` in `experiments_causal.py` or use a solver available on your system.

A good practice is to export the full conda environment used for the causal experiments and store it alongside the repository for exact reconstruction.

---

## Quick start

### Recommended end-to-end run

```bash
python experiments_causal.py --run_both_state_modes --emit_latex
```

To inspect the commands without executing them:

```bash
python experiments_causal.py --run_both_state_modes --dry_run
```

### Manual decomposition

The commands below show the direct stage-level workflow. The end-to-end driver remains the canonical source for the exact current configuration.

#### 1) Build a causal policy dataset

```bash
python build_causal_policy_dataset.py \
  --out_root causal_demo \
  --state_mode ground_truth \
  --candidate_interventions_json candidate_interventions.json \
  --include_baseline \
  --train_seeds 4000:4019 \
  --val_seeds 5000:5003 \
  --test_seeds 6000:6009 \
  --window_T 7 \
  --horizons 7 \
  --decision_days 7:60:7 \
  --decision_stride 1 \
  --action_start_mode branch_at_decision_day \
  --decision_applies_from next_day \
  --jobs 4 \
  --force \
  -- \
  --num_days 60 \
  --num_regions 1 \
  --num_wards 5 \
  --num_patients 50 \
  --num_staff 100 \
  --staff_wards_per_staff 2 \
  --isolation_mult 0.8 \
  --isolation_days 3 \
  --p_import_cs 0.3 \
  --p_import_cr 0.1 \
  --p_import_is 0.3 \
  --p_import_ir 0.1 \
  --p_admit_import_cs 0.3 \
  --p_admit_import_cr 0.4 \
  --p_admit_import_is 0.3 \
  --p_admit_import_ir 0.4 \
  --daily_discharge_frac 0.30 \
  --daily_discharge_min_per_ward 3 \
  --screen_every_k_days 7 \
  --screen_on_admission 0 \
  --screen_result_delay_days 3 \
  --persist_observations 1 \
  --beta_res_mult 1.0 \
  --beta_state_mult_cs 0.5 \
  --beta_state_mult_cr 0.5 \
  --beta_state_mult_is 1.0 \
  --beta_state_mult_ir 1.0 \
  --crossward_hand_hygiene_enabled 1 \
  --crossward_hand_hygiene_mult 0.55 \
  --passive_screen_capacity_per_week 0 \
  --active_symptom_screening 1 \
  --admission_screen_symptomatic_only 1 \
  --isolation_trigger_policy resistant_only \
  --admit_import_seasonality shock \
  --admit_import_shock_min_days 20 \
  --admit_import_shock_max_days 30 \
  --admit_import_shock_mult_min 2.0 \
  --admit_import_shock_mult_max 3.0 \
  --admit_import_pmax_cs 1.0 \
  --admit_import_pmax_cr 1.0 \
  --admit_import_pmax_is 1.0 \
  --admit_import_pmax_ir 1.0 \
  --superspreader_staff s1 \
  --superspreader_state CR \
  --superspreader_start_day 14 \
  --superspreader_end_day 21 \
  --superspreader_patient_frac_mult 2 \
  --superspreader_patient_min_add 1 \
  --superspreader_staff_contacts 30 \
  --superspreader_edge_weight_mult 2
```

#### 2) Convert GraphML outputs to `.pt`

```bash
python convert_to_pt.py --graphml_dir causal_demo
```

#### 3) Train the action-conditioned model

```bash
python train_amr_dygformer.py \
  --data_folder causal_demo \
  --task oracle_best_action_h7 \
  --T 7 \
  --sliding_step 1 \
  --hidden 128 \
  --heads 2 \
  --dropout 0.2 \
  --transformer_layers 2 \
  --sage_layers 4 \
  --batch_size 8 \
  --epochs 100 \
  --lr 1e-4 \
  --neighbor_sampling false \
  --max_neighbors 0 \
  --use_action_conditioning true \
  --action_hidden_dim 256 \
  --action_interaction_hidden_dim 256 \
  --action_interaction_dropout 0.1 \
  --require_complete_action_set true \
  --oracle_best_action_loss true \
  --oracle_best_action_loss_weight 1 \
  --aux_policy_loss false \
  --aux_policy_target_name y_h7_trans_import_res_gain \
  --aux_policy_loss_weight 2 \
  --pairwise_policy_ranking_loss true \
  --pairwise_policy_ranking_weight 1 \
  --pairwise_policy_margin 0.02 \
  --pairwise_policy_min_target_gap 0.01 \
  --emit_translational_figures false \
  --out_dir policy_training_large_ns
```

#### 4) Evaluate the held-out policy selector

```bash
python evaluate_policy_selector.py \
  --data_folder causal_demo \
  --trained_dir policy_training_large_ns \
  --split test \
  --oracle_metric y_h7_trans_import_res_gain \
  --selection_direction auto \
  --emit_action_scores_csv true \
  --candidate_interventions_json candidate_interventions.json \
  --out_dir policy_selector_eval_large
```

#### 5) Run the MILP decision layer

```bash
python optimize_policy_milp.py \
  --eval_dir policy_selector_eval_large \
  --config_json milp_policy_config.json \
  --split test \
  --score_transform baseline_delta \
  --solver cplex \
  --time_limit_sec 300 \
  --mip_gap 0.0 \
  --out_dir policy_milp_large
```

If no CPLEX installation is available, use another Pyomo-supported solver accepted by `optimize_policy_milp.py`.

---

## Full reproducibility pipeline

The recommended full causal run is:

```bash
python experiments_causal.py --run_both_state_modes --emit_latex
```

This pipeline is designed to be:

- deterministic in folder layout,
- resumable by stage number,
- track-aware across `ground_truth` and `partial_observation`,
- suitable for manuscript-oriented archive generation,
- compatible with held-out policy evaluation,
- and compatible with downstream MILP optimization.

### Dual-track execution

When `--run_both_state_modes` is used, the driver runs the two state modes as isolated track-specific workflows and writes separate outputs per track.

### Canonical causal workflow structure

The current causal pipeline is organized around four main stages:

- **Stage 1 / C1**: build the causal policy dataset,
- **Stage 2 / C2**: train the action-conditioned temporal GNN,
- **Stage 3 / C3**: evaluate policy selection on held-out decision states,
- **Stage 4 / C4**: optimize a feasible policy with MILP over the scored candidate actions.

The command-line interface uses numeric `--start` and `--stop` values.

### Single-track runs

Ground-truth mode only:

```bash
DT_STATE_MODE=ground_truth python experiments_causal.py --emit_latex
```

Partial-observation mode only:

```bash
DT_STATE_MODE=partial_observation python experiments_causal.py --emit_latex
```

---

## Resuming and partial reruns

The pipeline is resumable by numeric stage.

### Example: rerun training through optimization

```bash
python experiments_causal.py --run_both_state_modes --start 2 --stop 4 --emit_latex
```

### Example: rebuild only the dataset

```bash
python experiments_causal.py --run_both_state_modes --start 1 --stop 1
```

### Example: reevaluate and rerun MILP from existing trained outputs

```bash
python experiments_causal.py --run_both_state_modes --start 3 --stop 4 --emit_latex
```

Supported stage values are:

```text
1 = dataset construction
2 = model training
3 = held-out policy evaluation
4 = MILP optimization
```

---

## Important command-line options

The checked-in `experiments_causal.py` exposes:

```bash
--dry_run
--run_both_state_modes
--results_parent
--emit_latex
--emit_latex_only
--overleaf_dir
--start
--stop
```

Use:

```bash
python experiments_causal.py --help
```

for the exact CLI of the repository version being run.

Stage-specific scripts expose their own options:

```bash
python build_causal_policy_dataset.py --help
python train_amr_dygformer.py --help
python evaluate_policy_selector.py --help
python optimize_policy_milp.py --help
```

---

## Current default causal configuration

The default values below reflect the checked-in `experiments_causal.py`, `best_config.json`, `candidate_interventions.json`, and `milp_policy_config.json`.

### Dataset and split settings

```text
out_name = causal_policy_monthly_shock_superspreader
state modes = ground_truth, partial_observation
include_baseline = true
train_seeds = 4000:4019
val_seeds = 5000:5003
test_seeds = 6000:6009
window T = 7
horizon H = 7
decision_days = 7:60:7
valid decision days = 7, 14, 21, 28, 35, 42, 49
decision_stride = 1
action_start_mode = branch_at_decision_day
decision_applies_from = next_day
shared_noise_seed_offset = 910000
jobs = 4
```

### Policy-learning target

```text
task = oracle_best_action_h7
oracle metric = y_h7_trans_import_res_gain
continuous ranking target = y_h7_trans_import_res_gain
selection direction = larger is better under the default gain target
implemented burden = new_trans_cr_total + new_import_cr_total + new_import_ir_total
gain = baseline burden - action burden
```

### Model settings

```text
use_action_conditioning = true
require_complete_action_set = true
hidden = 128
heads = 2
dropout = 0.2
transformer_layers = 2
sage_layers = 4
batch_size = 8
epochs = 100
lr = 1e-4
neighbor_sampling = false
max_neighbors = 0
seed_count = 256
seed_strategy = random
seed_batch_size = 64
max_sub_batches = 4
emit_translational_figures = false
action_hidden_dim = 256
action_interaction_hidden_dim = 256
action_interaction_dropout = 0.1
```

### Policy-aligned losses

```text
oracle_best_action_loss = true
oracle_best_action_loss_weight = 1
aux_policy_loss = false
aux_policy_target_name = y_h7_trans_import_res_gain
aux_policy_loss_weight = 2
pairwise_policy_ranking_loss = true
pairwise_policy_ranking_weight = 1
pairwise_policy_margin = 0.02
pairwise_policy_min_target_gap = 0.01
```

The auxiliary policy loss is configured with the same gain target name but disabled by default. The within-state ranking regularizer is active.

### Causal simulation regime

```text
num_days = 60
num_regions = 1
num_wards = 5
num_patients = 50
num_staff = 100
staff_wards_per_staff = 2
baseline screen_every_k_days = 7
baseline screen_result_delay_days = 3
baseline screen_on_admission = 0
baseline admission_screen_symptomatic_only = 1
baseline active_symptom_screening = 1
baseline passive_screen_capacity_per_week = 0
baseline isolation_trigger_policy = resistant_only
baseline isolation_mult = 0.8
baseline isolation_days = 3
admission importation seasonality = shock
admission importation shock duration = 20 to 30 days
admission importation shock multiplier = 2.0 to 3.0
superspreader_staff = s1
superspreader_state = CR
superspreader_start_day = 14
superspreader_end_day = 21
```

### Intervention library

The action set contains:

```text
baseline
screen every 2 days, delay 0 days, admission screening enabled, resistant-only isolation
screen every 7 days, delay 2 days, admission screening enabled, resistant-only isolation
screen every 2 days, delay 0 days, admission screening enabled, all-positive isolation
screen every 7 days, delay 2 days, admission screening enabled, all-positive isolation
```

All four nonbaseline actions use:

```text
isolation_mult = 0.75
isolation_days = 3
target_type = global
target_id = hospital
action family = set_screening_frequency
```

### Evaluation settings

```text
split = test
oracle_metric = y_h7_trans_import_res_gain
selection_direction = auto
emit_action_scores_csv = true
out_name = policy_selector_eval_large
```

Evaluation retains only decision states with complete action sets and finite oracle values.

### MILP layer

```text
config_json = milp_policy_config.json
solver = cplex
score_transform = baseline_delta
split = test
time_limit_sec = 300
mip_gap = 0.0
out_name = policy_milp_large
```

The checked-in MILP config contains:

```text
default_action_cost = 0.0
default_cooldown_days = 0
budget_total = null
baseline cost = 0.0
baseline cooldown_days = 0
set_screening_frequency cost = 1.0
set_screening_frequency cooldown_days = 7
set_screening_frequency max_uses = null
```

Because `budget_total = null`, action costs are recorded but do not create a binding budget constraint unless a budget is supplied.

---

## Outputs

A typical full run writes to a results root such as:

```text
experiments_causal_results/
├── run_report.txt
├── TRACK_ground_truth/
│   └── work/
│       ├── causal_policy_monthly_shock_superspreader/
│       ├── policy_training_large_ns/
│       ├── policy_selector_eval_large/
│       ├── policy_milp_large/
│       └── repro_artifacts_causal/
└── TRACK_partial_observation/
    └── work/
        ├── causal_policy_monthly_shock_superspreader/
        ├── policy_training_large_ns/
        ├── policy_selector_eval_large/
        ├── policy_milp_large/
        └── repro_artifacts_causal/
```

Outputs may include:

- policy manifests,
- converted `.pt` graph datasets,
- dataset summaries,
- trained model checkpoints,
- validation and test metrics,
- action-score CSV exports,
- regret and policy-accuracy summaries,
- MILP solution summaries,
- archived artifacts for manuscript integration,
- and causal figures and tables prepared for Overleaf export.

---

## Overleaf export

When `--emit_latex` is enabled, the driver assembles an Overleaf-ready package by collecting archived figures and generating reusable LaTeX blocks.

Example:

```bash
python experiments_causal.py --run_both_state_modes --emit_latex
```

Typical export location:

```text
experiments_causal_results/overleaf_package2/
```

Typical contents:

- `figures/`,
- `latex.txt`.

This package may include:

- causal evaluation figures,
- policy tables,
- action-ranking summaries,
- MILP results summaries,
- and appendix-ready parameter tables.

To rebuild only the manuscript package from existing results:

```bash
python experiments_causal.py --emit_latex_only
```

---

## Reproducibility and design notes

This repository is structured for repeatable causal-policy experimentation.

Key safeguards in the current workflow include:

- deterministic per-track working directories,
- explicit seed-level split assignment,
- prevention of leakage across action-conditioned samples from the same seed,
- manifest-based temporal dataset construction,
- explicit baseline inclusion in the intervention library,
- held-out policy evaluation against simulator-defined oracle actions,
- complete-action-set filtering during training/evaluation when requested,
- separation between learned action scoring and downstream optimization,
- support for dual-track execution across `ground_truth` and `partial_observation`,
- and archive generation for manuscript integration.

Important implementation details:

1. The branch-at-decision workflow uses matched stochastic reruns rather than checkpoint-restored simulator branching. The pre-intervention supervised history is fixed to the baseline window for all actions associated with a seed and decision day.
2. The causal workflow is distinct from the predictive benchmark workflow and should be reproduced with `experiments_causal.py` rather than `experiments_pbc7.py`.
3. The MILP layer is a downstream constrained selector over learned state-action utilities; it is not a fully simulator-embedded hospital-control optimizer.
4. The checked-in `verify_causal_branching.py` expects explicit graph-path fields that are not populated by the current branch-at-decision manifest layout. Treat it as a diagnostic utility for compatible manifests rather than as the primary validation script for the current run path.
5. Candidate-action `display_name` fields are present in `candidate_interventions.json`; the manifest action identifiers and action features should be used for reliable action disambiguation.

---

## Citation

If you use this repository, please cite the associated manuscript and the causal-policy extension of the AMR digital-twin framework.
