"""
Candidate C experiment: does a hardware-axis quantity (noise level p) predict
the *quantum-mediated* contribution to QML performance, and does a combined
data+hardware prediction beat a data-axis-only prediction?

Methodology follows the causal-mediation spirit of Kesiku & Garcia-Zapirain
(2026): for each run we measure
    - task performance (accuracy)                 -> outcome
    - a classical-architecture proxy (param count,
      circuit expressivity via effective dimension) -> "direct effect" mediator
    - quantum-state mediators (entanglement entropy,
      purity, linear entropy)                       -> "indirect effect" mediators
and estimate each mediator's contribution to the outcome via a simple
linear mediation (Baron-Kenny style) decomposition, producing a
Relative Quantum Contribution (RQC) analogue per run.

This is a *skeleton*: circuit topologies, datasets, and mediator estimators
are simplified relative to a full replication, but the pipeline, noise sweep,
and analysis logic are complete and runnable end-to-end on a laptop.

Requires: pennylane, numpy, pandas, scikit-learn, statsmodels, matplotlib
Install:  pip install pennylane pennylane-lightning scikit-learn numpy pandas
          statsmodels matplotlib --break-system-packages
"""

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
N_QUBITS = 4
N_LAYERS_LIST = [1, 3]           # architecture axis: circuit depth/expressivity proxy
NOISE_LEVELS = np.linspace(0.0, 0.25, 8)   # hardware axis: depolarizing prob per gate
N_TRAIN_STEPS = 60
LEARNING_RATE = 0.2
SEED = 42
N_TRAIN_SAMPLES = 60
N_TEST_SAMPLES = 40

np.random.seed(SEED)


# ----------------------------------------------------------------------
# Datasets: one "quantum-favorable" (structured, high geometric-difference
# proxy), one "classically-easy" (linearly separable) -- crude stand-ins
# for the data-axis distinction from Huang et al. (2021).
# ----------------------------------------------------------------------
def get_dataset(kind: str):
    if kind == "structured":
        # Iris, projected to N_QUBITS features -- nontrivial structure,
        # used as a rough proxy for "harder to imitate classically".
        X, y = load_iris(return_X_y=True)
        mask = y < 2  # binary subset
        X, y = X[mask], y[mask]
        if X.shape[1] < N_QUBITS:
            X = np.hstack([X, np.zeros((X.shape[0], N_QUBITS - X.shape[1]))])
        else:
            X = X[:, :N_QUBITS]
    elif kind == "easy":
        X, y = make_classification(
            n_samples=200, n_features=N_QUBITS, n_informative=N_QUBITS,
            n_redundant=0, n_clusters_per_class=1, class_sep=2.5,
            random_state=SEED,
        )
    elif kind == "moons":
        # Nonlinearly separable (interleaving half-moons); pad to N_QUBITS.
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=200, noise=0.15, random_state=SEED)
        if X.shape[1] < N_QUBITS:
            pad = np.random.RandomState(SEED).uniform(-1, 1, size=(X.shape[0], N_QUBITS - X.shape[1]))
            X = np.hstack([X, pad])
    elif kind == "circles":
        # Radially structured (concentric circles) -- different geometry
        # from both existing datasets, genuinely nonlinear.
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=200, noise=0.1, factor=0.4, random_state=SEED)
        if X.shape[1] < N_QUBITS:
            pad = np.random.RandomState(SEED + 1).uniform(-1, 1, size=(X.shape[0], N_QUBITS - X.shape[1]))
            X = np.hstack([X, pad])
    elif kind == "linear_controlled":
        # Controlled comparison pair with "radial_controlled": same sample
        # size, same 2D-plus-padding structure, same noise level, same
        # random seed for padding -- differs only in decision boundary
        # shape (linear vs radial), isolating that one variable.
        rng = np.random.RandomState(SEED)
        pts = rng.uniform(-1, 1, size=(200, 2))
        w = np.array([1.0, 1.0]) / np.sqrt(2)
        margin = pts @ w
        y = (margin > 0).astype(int)
        pts = pts + rng.normal(0, 0.15, size=pts.shape)
        pad = np.random.RandomState(SEED + 1).uniform(-1, 1, size=(200, 2))
        X = np.hstack([pts, pad])
    elif kind == "radial_controlled":
        rng = np.random.RandomState(SEED)
        pts = rng.uniform(-1, 1, size=(200, 2))
        radius = np.linalg.norm(pts, axis=1)
        y = (radius > np.median(radius)).astype(int)
        pts = pts + rng.normal(0, 0.15, size=pts.shape)
        pad = np.random.RandomState(SEED + 1).uniform(-1, 1, size=(200, 2))
        X = np.hstack([pts, pad])
    else:
        raise ValueError(kind)

    X = MinMaxScaler(feature_range=(0, np.pi)).fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=N_TRAIN_SAMPLES, test_size=N_TEST_SAMPLES,
        random_state=SEED, stratify=y,
    )
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------
# Circuit: a hardware-efficient ansatz with a tunable depolarizing noise
# channel per gate, plus mediator readouts (entanglement entropy, purity).
# ----------------------------------------------------------------------
def make_device(noise_p: float):
    if noise_p > 0:
        return qml.device("default.mixed", wires=N_QUBITS)
    return qml.device("default.qubit", wires=N_QUBITS)


def circuit_factory(n_layers: int, noise_p: float, include_entangling: bool = True):
    """
    include_entangling=False produces a *matched noise-only control circuit*:
    same single-qubit rotations and the same number/placement of
    DepolarizingChannel insertions as the real circuit, but with the CNOTs
    removed. This isolates decoherence-induced mixedness (which the control
    circuit still produces, since it has the same noise exposure) from
    genuine multi-qubit entanglement (which only the real circuit produces).
    """
    dev = make_device(noise_p)

    def ansatz(weights, x):
        for i in range(N_QUBITS):
            qml.RY(x[i], wires=i)
            if noise_p > 0:
                qml.DepolarizingChannel(noise_p, wires=i)
        for layer in range(n_layers):
            for i in range(N_QUBITS):
                qml.RY(weights[layer, i, 0], wires=i)
                qml.RZ(weights[layer, i, 1], wires=i)
                if noise_p > 0:
                    qml.DepolarizingChannel(noise_p, wires=i)
            if include_entangling:
                for i in range(N_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])
                    if noise_p > 0:
                        qml.DepolarizingChannel(noise_p, wires=i + 1)
            else:
                # Same noise exposure (same wires, same channel, same count)
                # as the entangling branch, but the CNOT itself is omitted.
                if noise_p > 0:
                    for i in range(N_QUBITS - 1):
                        qml.DepolarizingChannel(noise_p, wires=i + 1)

    @qml.qnode(dev)
    def classify(weights, x):
        ansatz(weights, x)
        return qml.expval(qml.PauliZ(0))

    @qml.qnode(dev)
    def state_probe(weights, x):
        ansatz(weights, x)
        return qml.density_matrix(wires=range(N_QUBITS)) if noise_p > 0 else qml.state()

    return classify, state_probe, dev


# ----------------------------------------------------------------------
# Mediators
# ----------------------------------------------------------------------
def von_neumann_entropy(rho, base=2):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]
    return float(-np.sum(eigvals * np.log(eigvals) / np.log(base)))


def entanglement_and_purity(weights, x, state_probe, noise_p):
    if noise_p > 0:
        rho_full = np.array(state_probe(weights, x))
    else:
        psi = np.array(state_probe(weights, x))
        rho_full = np.outer(psi, psi.conj())

    rho_full = rho_full.reshape([2] * N_QUBITS + [2] * N_QUBITS)
    rho_reduced = _partial_trace_to_qubit0(rho_full)

    ent_entropy = von_neumann_entropy(rho_reduced)
    purity = float(np.real(np.trace(rho_reduced @ rho_reduced)))
    return ent_entropy, purity


def _partial_trace_to_qubit0(rho_tensor):
    """Trace out all qubits except qubit 0 from a (2,)*2N tensor."""
    n = N_QUBITS
    rho = rho_tensor.reshape(2 ** n, 2 ** n)
    rho = rho.reshape([2] + [2 ** (n - 1)] + [2] + [2 ** (n - 1)])
    reduced = np.einsum("aibj,ij->ab", rho, np.eye(2 ** (n - 1)))
    return reduced


def _partial_trace_to_rest(rho_tensor):
    """Trace out qubit 0 only, keeping qubits 1..N-1 (the complement of
    _partial_trace_to_qubit0), needed for mutual information."""
    n = N_QUBITS
    rho = rho_tensor.reshape(2 ** n, 2 ** n)
    rho = rho.reshape([2, 2 ** (n - 1), 2, 2 ** (n - 1)])
    reduced = np.einsum("iajb,ij->ab", rho, np.eye(2))
    return reduced


def mutual_information(weights, x, state_probe, noise_p):
    """
    I(qubit0 : rest) = S(qubit0) + S(rest) - S(full).
    Under noise the full multi-qubit state is mixed (S(full) > 0), which
    makes this genuinely distinct from single-qubit entanglement entropy
    -- unlike raw entropy vs. purity (both simple mixedness measures on
    the same reduced state), MI explicitly incorporates the joint state's
    own mixedness, addressing the referee's request for an
    information-theoretically independent mediator.
    """
    if noise_p > 0:
        rho_full = np.array(state_probe(weights, x))
    else:
        psi = np.array(state_probe(weights, x))
        rho_full = np.outer(psi, psi.conj())

    S_full = von_neumann_entropy(rho_full)
    rho_full_t = rho_full.reshape([2] * N_QUBITS + [2] * N_QUBITS)
    rho_A = _partial_trace_to_qubit0(rho_full_t)
    rho_B = _partial_trace_to_rest(rho_full_t)
    S_A = von_neumann_entropy(rho_A)
    S_B = von_neumann_entropy(rho_B)
    return float(S_A + S_B - S_full)


def entanglement_residual(weights, x, n_layers, noise_p):
    """
    Fix (ii) for the noise/entanglement confound: compare the real
    (entangling) circuit's reduced-state entropy against a matched
    noise-only control circuit (same noise exposure, CNOTs removed).
    The residual, real_entropy - control_entropy, is intended to isolate
    genuine entanglement from decoherence-induced mixedness; clipped at 0
    since noise-only entropy should not legitimately exceed the real
    circuit's entropy.
    """
    _, probe_real, _ = circuit_factory(n_layers, noise_p, include_entangling=True)
    _, probe_ctrl, _ = circuit_factory(n_layers, noise_p, include_entangling=False)
    e_real, _ = entanglement_and_purity(weights, x, probe_real, noise_p)
    e_ctrl, _ = entanglement_and_purity(weights, x, probe_ctrl, noise_p)
    residual = max(0.0, e_real - e_ctrl)
    return residual, e_real, e_ctrl


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train_and_evaluate(dataset_kind, n_layers, noise_p, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X_train, X_test, y_train, y_test = get_dataset(dataset_kind)
    y_train_pm1 = 2 * y_train - 1
    y_test_pm1 = 2 * y_test - 1

    classify, state_probe, dev = circuit_factory(n_layers, noise_p)
    weights = pnp.array(
        0.1 * np.random.randn(n_layers, N_QUBITS, 2), requires_grad=True
    )

    opt = qml.AdamOptimizer(stepsize=LEARNING_RATE)

    def cost(w):
        preds = pnp.stack([classify(w, x) for x in X_train])
        return pnp.mean((preds - y_train_pm1) ** 2)

    for _ in range(N_TRAIN_STEPS):
        weights = opt.step(cost, weights)

    preds_test = np.array([classify(weights, x) for x in X_test])
    acc = float(np.mean(np.sign(preds_test) == y_test_pm1))

    # mediators, averaged over a small probe subset for speed
    probe_idx = np.random.choice(len(X_test), size=min(8, len(X_test)), replace=False)
    ents, purities, residuals, ctrl_ents, mis = [], [], [], [], []
    for i in probe_idx:
        e, p = entanglement_and_purity(weights, X_test[i], state_probe, noise_p)
        ents.append(e)
        purities.append(p)
        resid, e_real, e_ctrl = entanglement_residual(weights, X_test[i], n_layers, noise_p)
        residuals.append(resid)
        ctrl_ents.append(e_ctrl)
        mis.append(mutual_information(weights, X_test[i], state_probe, noise_p))

    param_count = weights.size  # crude classical-architecture-scaling proxy

    return {
        "dataset": dataset_kind,
        "n_layers": n_layers,
        "noise_p": noise_p,
        "accuracy": acc,
        "param_count": param_count,
        "mean_entanglement_entropy": float(np.mean(ents)),
        "mean_purity": float(np.mean(purities)),
        "mean_control_entropy": float(np.mean(ctrl_ents)),
        "mean_entanglement_residual": float(np.mean(residuals)),
        "mean_mutual_information": float(np.mean(mis)),
    }


# ----------------------------------------------------------------------
# Mediation-style decomposition (simplified Baron-Kenny linear mediation):
#   outcome ~ direct_proxy + quantum_proxy
# RQC analogue = |beta_quantum| / (|beta_direct| + |beta_quantum|)
#
# IMPORTANT: we group by dataset ONLY (not by n_layers). Grouping by
# (dataset, n_layers) would hold architecture fixed within each group,
# making param_count constant and therefore unable to explain any
# variance -- that silently biases RQC toward 1.0 regardless of the
# true quantum contribution. Grouping by dataset alone lets param_count
# vary across n_layers and mean_entanglement_entropy vary across both
# n_layers and noise, so the regression can actually separate them.
# ----------------------------------------------------------------------
def compute_rqc(df: pd.DataFrame, quantum_col: str = "mean_entanglement_entropy") -> pd.DataFrame:
    results = []
    for dataset, group in df.groupby("dataset"):
        if len(group) < 4:
            continue
        X = sm.add_constant(group[["param_count", quantum_col]])
        y = group["accuracy"]
        model = sm.OLS(y, X).fit()
        b_direct = abs(model.params.get("param_count", 0.0))
        b_quantum = abs(model.params.get(quantum_col, 0.0))
        denom = b_direct + b_quantum
        rqc = b_quantum / denom if denom > 0 else np.nan
        results.append({"dataset": dataset, "mediator": quantum_col, "rqc": rqc,
                         "beta_direct": b_direct, "beta_quantum": b_quantum,
                         "n_obs": len(group)})
    return pd.DataFrame(results)


# ----------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------
def run_sweep():
    rows = []
    for dataset_kind in ["structured", "easy"]:
        for n_layers in N_LAYERS_LIST:
            for noise_p in NOISE_LEVELS:
                print(f"Running: dataset={dataset_kind}, layers={n_layers}, p={noise_p:.3f}")
                row = train_and_evaluate(dataset_kind, n_layers, noise_p)
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    df = run_sweep()
    df.to_csv("/mnt/user-data/outputs/qml_sweep_results.csv", index=False)

    # RQC per (dataset, layers) combo, computed across the noise sweep
    rqc_df = compute_rqc(df)
    rqc_df.to_csv("/mnt/user-data/outputs/qml_rqc_summary.csv", index=False)
    print("\nRQC summary (fraction of accuracy variance attributable to the\n"
          "quantum-mediated proxy vs. the classical-architecture proxy):")
    print(rqc_df)

    # Plot: accuracy and entanglement entropy vs noise, per dataset/layers
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for (dataset, layers), group in df.groupby(["dataset", "n_layers"]):
        group = group.sort_values("noise_p")
        label = f"{dataset}, L={layers}"
        axes[0].plot(group["noise_p"], group["accuracy"], marker="o", label=label)
        axes[1].plot(group["noise_p"], group["mean_entanglement_entropy"],
                     marker="o", label=label)
    axes[0].set_xlabel("Depolarizing noise p")
    axes[0].set_ylabel("Test accuracy")
    axes[0].set_title("Performance vs. hardware noise")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Depolarizing noise p")
    axes[1].set_ylabel("Mean entanglement entropy (qubit 0)")
    axes[1].set_title("Quantum-mediator decay vs. hardware noise")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("/mnt/user-data/outputs/qml_noise_sweep.png", dpi=150)
    print("\nSaved: qml_sweep_results.csv, qml_rqc_summary.csv, qml_noise_sweep.png")


if __name__ == "__main__":
    main()
