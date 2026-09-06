import numpy as np, json, csv, time, sys
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from qsim import Circuit
from datasets import build_datasets, geometric_difference, task_hardness_features

N_QUBITS = 4
ARCHITECTURES = ['shallow_hea', 'deep_hea', 'strong_entangle', 'reupload']
HARDWARE_CONDITIONS = [
    ('none', 0.0),
    ('depolarizing', 0.15),
    ('depolarizing', 0.30),
    ('amplitude_damping', 0.15),
    ('phase_damping', 0.15),
]
SEEDS = [0, 1, 2, 3, 4]
MAXITER = 25

OUT_CSV = 'results_qml_4q.csv'
FIELDS = ['dataset', 'regime', 'architecture', 'noise_model', 'noise_p', 'seed',
          'train_acc', 'test_acc', 'test_loss', 'n_params', 'wall_time']


def train_one(circuit, Xtr, ytr, seed):
    rng = np.random.RandomState(seed)
    x0 = rng.uniform(-0.5, 0.5, circuit.n_params)

    def loss_fn(params):
        p = circuit.predict_proba(Xtr, params, circuit._noise_model, circuit._noise_p)
        eps = 1e-9
        return -np.mean(ytr * np.log(p + eps) + (1 - ytr) * np.log(1 - p + eps))

    res = minimize(loss_fn, x0, method='COBYLA', options={'maxiter': MAXITER})
    return res.x, res.fun


def main():
    import os
    datasets = build_datasets(n_qubits=N_QUBITS)
    done_keys = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline='') as f:
            reader = csv.DictReader(f)
            for r in reader:
                done_keys.add((r['dataset'], r['architecture'], r['noise_model'], r['noise_p'], r['seed']))
    else:
        with open(OUT_CSV, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
    print(f'resuming: {len(done_keys)} already done', flush=True)

    t_start = time.time()
    n_total = len(datasets) * len(ARCHITECTURES) * len(HARDWARE_CONDITIONS) * len(SEEDS)
    n_done = 0

    for dname, (X, y, meta) in datasets.items():
        for arch in ARCHITECTURES:
            for (noise_model, noise_p) in HARDWARE_CONDITIONS:
                for seed in SEEDS:
                    key = (dname, arch, noise_model, str(noise_p), str(seed))
                    if key in done_keys:
                        n_done += 1
                        continue
                    t0 = time.time()
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=8, train_size=18, random_state=seed)
                    tr_idx, te_idx = next(sss.split(X, y))
                    Xtr, ytr = X[tr_idx], y[tr_idx]
                    Xte, yte = X[te_idx], y[te_idx]

                    circuit = Circuit(N_QUBITS, arch, N_QUBITS)
                    circuit._noise_model = noise_model
                    circuit._noise_p = noise_p

                    params, train_loss = train_one(circuit, Xtr, ytr, seed)

                    ptr = circuit.predict_proba(Xtr, params, noise_model, noise_p)
                    pte = circuit.predict_proba(Xte, params, noise_model, noise_p)
                    train_acc = accuracy_score(ytr, (ptr > 0.5).astype(int))
                    test_acc = accuracy_score(yte, (pte > 0.5).astype(int))
                    eps = 1e-9
                    test_loss = -np.mean(yte * np.log(pte + eps) + (1 - yte) * np.log(1 - pte + eps))

                    row = dict(dataset=dname, regime=meta['regime'], architecture=arch,
                               noise_model=noise_model, noise_p=noise_p, seed=seed,
                               train_acc=train_acc, test_acc=test_acc, test_loss=test_loss,
                               n_params=circuit.n_params, wall_time=time.time() - t0)
                    with open(OUT_CSV, 'a', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=FIELDS)
                        writer.writerow(row)

                    n_done += 1
                    if n_done % 40 == 0:
                        elapsed = time.time() - t_start
                        print(f'{n_done}/{n_total} done, elapsed {elapsed:.0f}s, '
                              f'est total {elapsed / n_done * n_total:.0f}s', flush=True)

    print('DONE', time.time() - t_start, 's total')


if __name__ == '__main__':
    main()
