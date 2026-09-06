# QML Task-Data-Architecture-Hardware Alignment Benchmark

Code accompanying "A Diagnostic Framework for Task-Data-Architecture-Hardware
Alignment in Quantum Machine Learning."

## Contents

- `config.md` — every fixed experimental setting (seeds, thresholds, grid
  sizes, model specifications) in one place, for exact reproducibility.
- `significance_report.json` — bootstrap test results (Table: Statistical
  significance of the key comparisons).
- `outputs_2q/`, `outputs_4q/` — the actual generated result files (QML
  training results, master feature tables, model-comparison reports,
  classical-baseline results) for the 2-qubit and 4-qubit benchmarks,
  respectively, so the paper's numbers can be checked without re-running
  anything.
- `figures/` — the four figures used in the paper (source PNGs).

### `pilot_raw_logs/` — original pilot study (Sections: Results through Conflicting Axes)
Raw per-configuration run logs from the authors' original variational-circuit
implementation (not the simulator below). Six JSONL files:
- `fullgrid_seed_results.jsonl`, `sweep_results.jsonl` — the primary 2x2x6
  (dataset x depth x noise) grid on structured/easy, 24 configs each.
- `seed_results.jsonl` — additional seeds at the noise extremes.
- `mi_results.jsonl` — mutual-information mediator experiment (coarser noise
  grid, collected separately).
- `newdatasets_results.jsonl` — moons/circles out-of-sample extension.
- `geometry_test_results.jsonl` — linear/radial controlled-geometry probes.

Each record includes dataset, circuit depth (`n_layers`), noise level
(`noise_p`), accuracy, and mediator quantities (entanglement entropy, purity,
noise-only control entropy, and mutual information where computed). Every
accuracy and mediator figure in the pilot study section of the paper can be
recomputed directly from these logs; see the paper's "A note on implementation
provenance" section for what was independently re-verified.

### Extended benchmark code (Section: Extended Eight-Dataset Benchmark)
- `qsim.py` — density-matrix statevector simulator (arbitrary qubit count,
  depolarizing/amplitude-damping/phase-damping noise channels, 4 ansatz
  families).
- `datasets.py` — dataset construction (8 datasets spanning simple/nonlinear/
  noisy/high-dimensional/real-world regimes), task-hardness axis, and
  geometric-difference computation.
- `run_experiment.py` — trains the full architecture x noise x seed grid for
  a given qubit count (resumable/checkpointed; set `N_QUBITS` and `SEEDS`).
- `run_classical.py` — six classical baselines (logistic regression, linear/
  RBF SVM, random forest, gradient boosting, MLP) per dataset.
- `build_master.py` — joins QML results with the T/D/A/H axis table.
- `analyze.py` — fits M1 (single-axis)/M2 (additive)/M3 (interaction) models,
  evaluates via 5-fold CV and leave-one-dataset-out (LODO), runs ablations.
- `bootstrap_tests.py` — bootstrap significance tests for the CV-vs-LODO and
  M2-vs-M3 comparisons reported in the paper.

## Reproducing the paper's numbers
```
pip install -r requirements.txt
python run_experiment.py        # edit N_QUBITS (2 or 4) and SEEDS at the top
python run_classical.py
python build_master.py
python analyze.py
python bootstrap_tests.py
```
All random seeds are fixed in the source (dataset generation: `RNG_DATA =
12345`; QML training seeds: `SEEDS = [0,1,2,3,4]`).

## Known limitations (see paper Section: Limitations)
- No quantum SDK (Qiskit/PennyLane) is used; `qsim.py` is a from-scratch
  exact density-matrix simulator implemented in NumPy.
- Small sample sizes (26 samples/dataset) and small qubit counts (2, 4) —
  see paper for discussion.
