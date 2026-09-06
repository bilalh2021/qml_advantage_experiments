"""
Geometric-difference (g_CQ) computation, following Huang, Broughton, Mohseni,
Babbush, Boixo, Neven, McClean, "Power of data in quantum machine learning"
(Nature Communications 12, 2631, 2021).

Definition: given an N x N quantum kernel matrix K_Q and classical kernel
matrix K_C (both PSD, evaluated on the same N training points), the
geometric difference is

    g(K_C || K_Q) = sqrt( || sqrt(K_C) (K_Q + lambda I)^{-1} sqrt(K_C) ||_inf )

where ||.||_inf is the spectral (largest-eigenvalue) norm and lambda is a
small regularizer for invertibility. A large g_CQ (growing with N, up to
the trivial bound sqrt(N)) indicates the classical kernel would need much
higher effective model complexity than the quantum kernel to represent
functions the quantum kernel represents cheaply -- necessary (not
sufficient -- label complexity s_Q must also be checked) condition for a
potential quantum learning advantage on that data distribution.

This script computes g_CQ for the two datasets used in
qml_advantage_experiment.py, to test (exploratory, not a proven claim)
whether the data axis's favorability correlates with the noise-robustness
difference found in the main sweep (Finding 1: structured dataset stayed
accurate under noise, easy dataset collapsed).
"""

import numpy as np
import pennylane as qml
from scipy.linalg import sqrtm
from sklearn.metrics.pairwise import rbf_kernel

import qml_advantage_experiment as base

N_QUBITS = base.N_QUBITS
N_SAMPLES = 20  # kept small: kernel matrix costs O(N^2) circuit evaluations
REG = 1e-6


def quantum_kernel_matrix(X, n_layers=1, weights=None):
    """
    Fidelity-type quantum kernel: K_Q[i,j] = |<0| U(x_j)^dagger U(x_i) |0>|^2,
    using the same data-encoding structure (RY encoding + entangling layer)
    as the main experiment's ansatz, at a fixed set of weights (or identity
    if weights is None) and zero noise -- g_CQ is a noiseless-kernel notion.
    """
    dev = qml.device("default.qubit", wires=N_QUBITS)

    def encode(x):
        for i in range(N_QUBITS):
            qml.RY(x[i], wires=i)
        if weights is not None:
            for layer in range(n_layers):
                for i in range(N_QUBITS):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                for i in range(N_QUBITS - 1):
                    qml.CNOT(wires=[i, i + 1])

    @qml.qnode(dev)
    def overlap(x1, x2):
        encode(x1)
        qml.adjoint(lambda: encode(x2))()
        return qml.probs(wires=range(N_QUBITS))

    N = len(X)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            K[i, j] = overlap(X[i], X[j])[0]  # prob of all-zeros = |<0|...|0>|^2
    return K


def geometric_difference(K_from, K_to, reg=REG):
    """g(K_from || K_to): large value means K_to (e.g. quantum) can express
    functions cheaply that K_from (e.g. classical) cannot."""
    N = K_from.shape[0]
    K_to_reg = K_to + reg * np.eye(N)
    K_to_inv = np.linalg.inv(K_to_reg)
    sqrt_K_from = sqrtm(K_from + reg * np.eye(N)).real
    M = sqrt_K_from @ K_to_inv @ sqrt_K_from
    eigvals = np.linalg.eigvalsh(M)
    return float(np.sqrt(max(eigvals.max(), 0.0)))


def model_complexity(K, y, reg=REG):
    """s_K = sqrt(y^T (K + lambda I)^{-1} y), Huang et al.'s label-complexity
    term. Large s_K means the specific labeling y requires a high-complexity
    model in the RKHS of K -- i.e. even if g_QC is large (some functions
    favor quantum), THIS labeling might not be one of them unless s_Q is
    small (or at least not large) relative to s_C."""
    N = K.shape[0]
    K_reg = K + reg * np.eye(N)
    K_inv = np.linalg.inv(K_reg)
    val = float(y.T @ K_inv @ y)
    return float(np.sqrt(max(val, 0.0)))


def run_for_dataset(dataset_kind, reg=REG):
    X_train, X_test, y_train, y_test = base.get_dataset(dataset_kind)
    X = X_train[:N_SAMPLES]
    y = y_train[:N_SAMPLES].astype(float)
    y_pm1 = 2 * y - 1  # match the +-1 label convention used in training

    K_Q = quantum_kernel_matrix(X)  # untrained encoding kernel (weights=None)
    # classical kernel: RBF with median-heuristic bandwidth
    dists = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    median_dist = np.median(dists[dists > 0])
    gamma = 1.0 / (2 * median_dist ** 2) if median_dist > 0 else 1.0
    K_C = rbf_kernel(X, gamma=gamma)

    g_CQ = geometric_difference(K_C, K_Q, reg=reg)  # classical vs quantum
    g_QC = geometric_difference(K_Q, K_C, reg=reg)  # quantum vs classical
    sqrtN = np.sqrt(N_SAMPLES)

    s_Q = model_complexity(K_Q, y_pm1, reg=reg)
    s_C = model_complexity(K_C, y_pm1, reg=reg)

    return {
        "dataset": dataset_kind,
        "N": N_SAMPLES,
        "reg": reg,
        "g_CQ": g_CQ,
        "g_QC": g_QC,
        "sqrt_N": sqrtN,
        "g_CQ_normalized": g_CQ / sqrtN,
        "g_QC_normalized": g_QC / sqrtN,
        "s_Q": s_Q,
        "s_C": s_C,
        "s_Q_over_s_C": s_Q / s_C if s_C > 0 else np.nan,
    }


def main():
    results = []
    for dataset_kind in ["structured", "easy"]:
        print(f"Computing kernels for dataset={dataset_kind} ...")
        r = run_for_dataset(dataset_kind)
        print(r)
        results.append(r)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv("/mnt/user-data/outputs/geometric_difference_results.csv", index=False)
    print("\nSummary:")
    print(df.to_string())
    print(
        "\nInterpretation: higher g_CQ_normalized (closer to 1, its trivial "
        "upper bound) indicates the classical kernel needs disproportionately "
        "higher effective complexity than the quantum kernel to represent "
        "functions on that dataset -- the data-axis signal for potential "
        "quantum advantage. Compare against Finding 1 (structured dataset "
        "was noise-robust, easy dataset was noise-fragile) to see whether "
        "data-axis favorability lines up with the observed noise robustness."
    )


if __name__ == "__main__":
    main()
