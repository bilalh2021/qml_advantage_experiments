"""
Minimal density-matrix statevector simulator for small (2-qubit) variational
quantum classifiers with single-qubit noise channels (depolarizing, amplitude
damping, phase damping). Pure numpy, no external quantum SDK (none available
in this offline environment).
"""
import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)


def kron_list(ops):
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def rz(theta):
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex)


def embed_1q(gate, qubit, n):
    ops = [I2] * n
    ops[qubit] = gate
    return kron_list(ops)


def cnot(control, target, n):
    dim = 2 ** n
    U = np.zeros((dim, dim), dtype=complex)
    for basis in range(dim):
        bits = [(basis >> (n - 1 - q)) & 1 for q in range(n)]
        if bits[control] == 1:
            bits[target] ^= 1
        new_basis = 0
        for q in range(n):
            new_basis = (new_basis << 1) | bits[q]
        U[new_basis, basis] = 1.0
    return U


def kraus_depolarizing(p):
    # single-qubit depolarizing channel
    k0 = np.sqrt(1 - 3 * p / 4) * I2
    k1 = np.sqrt(p / 4) * X
    k2 = np.sqrt(p / 4) * Y
    k3 = np.sqrt(p / 4) * Z
    return [k0, k1, k2, k3]


def kraus_amplitude_damping(gamma):
    k0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    k1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [k0, k1]


def kraus_phase_damping(lam):
    k0 = np.array([[1, 0], [0, np.sqrt(1 - lam)]], dtype=complex)
    k1 = np.array([[0, 0], [0, np.sqrt(lam)]], dtype=complex)
    return [k0, k1]


NOISE_KRAUS = {
    "depolarizing": kraus_depolarizing,
    "amplitude_damping": kraus_amplitude_damping,
    "phase_damping": kraus_phase_damping,
    "none": lambda p: [I2],
}


def apply_channel_all_qubits(rho, n, noise_model, p):
    if noise_model == "none" or p == 0:
        return rho
    kraus_fn = NOISE_KRAUS[noise_model]
    ks = kraus_fn(p)
    for q in range(n):
        new_rho = np.zeros_like(rho)
        for k in ks:
            K = embed_1q(k, q, n)
            new_rho += K @ rho @ K.conj().T
        rho = new_rho
    return rho


class Circuit:
    """
    n-qubit variational circuit with configurable architecture family.
    architecture in {'shallow_hea','deep_hea','strong_entangle','reupload'}
    """

    def __init__(self, n_qubits, architecture, n_features):
        self.n = n_qubits
        self.arch = architecture
        self.n_features = n_features
        if architecture == "shallow_hea":
            self.n_layers, self.reupload, self.entangle = 1, False, "ring"
        elif architecture == "deep_hea":
            self.n_layers, self.reupload, self.entangle = 3, False, "ring"
        elif architecture == "strong_entangle":
            self.n_layers, self.reupload, self.entangle = 2, False, "all2all"
        elif architecture == "reupload":
            self.n_layers, self.reupload, self.entangle = 2, True, "ring"
        else:
            raise ValueError(architecture)
        # 2 rotation params per qubit per layer
        self.n_params = self.n_layers * self.n * 2

    def _entangling_layer(self, n):
        U = np.eye(2 ** n, dtype=complex)
        if self.entangle == "ring":
            pairs = [(q, (q + 1) % n) for q in range(n)] if n > 1 else []
        else:  # all2all
            pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        for (c, t) in pairs:
            U = cnot(c, t, n) @ U
        return U

    def _encode(self, x, n):
        U = np.eye(2 ** n, dtype=complex)
        for q in range(n):
            val = x[q % len(x)]
            U = embed_1q(ry(val), q, n) @ U
        return U

    def run(self, x, params, noise_model="none", noise_p=0.0):
        n = self.n
        rho = np.zeros((2 ** n, 2 ** n), dtype=complex)
        rho[0, 0] = 1.0
        # initial data encoding
        rho = self._encode(x, n) @ rho @ self._encode(x, n).conj().T
        rho = apply_channel_all_qubits(rho, n, noise_model, noise_p)
        idx = 0
        for layer in range(self.n_layers):
            if self.reupload and layer > 0:
                Ue = self._encode(x, n)
                rho = Ue @ rho @ Ue.conj().T
                rho = apply_channel_all_qubits(rho, n, noise_model, noise_p)
            U = np.eye(2 ** n, dtype=complex)
            for q in range(n):
                th_y = params[idx]; idx += 1
                th_z = params[idx]; idx += 1
                U = embed_1q(rz(th_z), q, n) @ embed_1q(ry(th_y), q, n) @ U
            rho = U @ rho @ U.conj().T
            Uent = self._entangling_layer(n)
            rho = Uent @ rho @ Uent.conj().T
            rho = apply_channel_all_qubits(rho, n, noise_model, noise_p)
        Zop = embed_1q(Z, 0, n)
        exp_z = np.real(np.trace(Zop @ rho))
        return exp_z, rho

    def predict_proba(self, X, params, noise_model="none", noise_p=0.0):
        probs = []
        for x in X:
            ez, _ = self.run(x, params, noise_model, noise_p)
            probs.append((1 - ez) / 2)  # map <Z>=+1(|0>)->class0, -1->class1
        return np.clip(np.array(probs), 1e-6, 1 - 1e-6)
