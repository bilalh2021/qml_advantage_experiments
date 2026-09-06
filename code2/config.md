# Experiment configuration (exact values used for all reported results)

## Environment
- Python 3.12.3, NumPy 2.4.4, SciPy 1.17.1, scikit-learn 1.8.0

## Extended benchmark (Section: Extended Eight-Dataset Benchmark)
- Qubit counts: 2 and 4 (set `N_QUBITS` in `run_experiment.py`)
- Datasets (8): structured, easy, moons, circles, blobs_overlap,
  label_noise_moons, high_dim_redundant, breast_cancer (`datasets.py`)
- Dataset generation seed: `RNG_DATA = 12345` (fixed, `datasets.py`)
- Samples per dataset: 26 (18 train / 8 test split per config)
- Architectures (4): shallow_hea, deep_hea, strong_entangle, reupload
- Hardware conditions (5): none@0.0, depolarizing@0.15, depolarizing@0.30,
  amplitude_damping@0.15, phase_damping@0.15
- QML training seeds: `SEEDS = [0, 1, 2, 3, 4]`
- Optimizer: COBYLA, `MAXITER = 25` (25 for 4-qubit runs; 35 for the
  earlier 2-qubit-only pass — see `run_experiment.py` history)
- Outcome threshold: `Y = 1[test_acc > 0.60]` (fixed before any LODO or
  bootstrap analysis; sensitivity checked at {0.55, 0.60, 0.65, 0.70})
- Total configurations: 800 at each qubit count (8 datasets x 4
  architectures x 5 hardware conditions x 5 seeds)

## Classical baselines (`run_classical.py`)
- logistic regression, linear SVM, RBF SVM, random forest (n_estimators=100,
  random_state=0), gradient boosting (random_state=0), MLP
  (hidden_layer_sizes=(8,), max_iter=2000, random_state=0)
- Evaluated via 5-fold `cross_val_score` on the same 4-qubit or 2-qubit
  feature representation as the QML runs for that qubit count.

## Model comparison / LODO (`analyze.py`)
- M1: single-axis logistic regression (T, D, A, or H alone)
- M2: additive logistic regression (T + D + A + H, standardized)
- M3: full pairwise-interaction logistic regression (M2's 4 terms + all 6
  pairwise products)
- Within-distribution CV: 5-fold `StratifiedKFold`, `shuffle=True`,
  `random_state=0`, refit per fold (`StandardScaler` + `LogisticRegression`,
  `max_iter=2000`, no additional hyperparameter tuning)
- LODO: train on 7 (2q: also excludes 2 degenerate datasets from
  evaluation, leaving 6 evaluable) or 7 (4q, all 8 evaluable) datasets,
  evaluate on the held-out dataset, repeated for every dataset

## Bootstrap significance (`bootstrap_tests.py`)
- LODO comparisons: paired bootstrap over held-out datasets, 5000 resamples,
  resampling per-dataset AUROC values with replacement
- CV comparisons: row-level bootstrap over configurations, 300 resamples,
  refitting the full 5-fold CV pipeline (new `StratifiedKFold` seed per
  resample) on each resample
- All comparisons, the outcome threshold, and the model specifications were
  fixed before these tests were run; no model or threshold was chosen after
  inspecting the bootstrap results.
