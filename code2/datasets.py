import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs, make_classification, load_breast_cancer, load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

RNG_DATA = 12345  # fixed seed for dataset generation itself (not QML training seeds)


def _to_angle_range(X):
    X = StandardScaler().fit_transform(X)
    X = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit_transform(X)
    return X


def _resize_features(X, n_qubits, rng):
    """Make X exactly n_qubits-dimensional: PCA down if wider, cyclic-pad if narrower."""
    d = X.shape[1]
    if d == n_qubits:
        return X
    if d > n_qubits:
        return PCA(n_components=n_qubits, random_state=RNG_DATA).fit_transform(X)
    cols = [X[:, i % d] for i in range(n_qubits)]
    return np.column_stack(cols)


def _add_nuisance_dims(X, n_qubits, rng, sigma=0.6):
    """Embed an intrinsically 2D manifold (moons/circles) in n_qubits dimensions
    by appending label-independent Gaussian nuisance features. This tests
    whether extra qubits/entangling capacity help, hurt, or are neutral when
    they carry no additional class information."""
    d = X.shape[1]
    if n_qubits <= d:
        return X
    extra = rng.normal(0, sigma, size=(X.shape[0], n_qubits - d))
    return np.hstack([X, extra])


def build_datasets(n_samples=26, n_qubits=4):
    rng = np.random.RandomState(RNG_DATA)
    ds = {}

    # 1. structured: Iris, 2 classes, native 4 features
    iris = load_iris()
    mask = iris.target != 2
    Xi, yi = iris.data[mask], iris.target[mask]
    Xi = _resize_features(Xi, n_qubits, rng)
    idx = rng.choice(len(Xi), n_samples, replace=False)
    ds['structured'] = (_to_angle_range(Xi[idx]), yi[idx], dict(regime='simple', noise_injected=0.0))

    # 2. easy: linearly separable synthetic, generated directly at n_qubits features
    Xe, ye = make_classification(n_samples=n_samples, n_features=n_qubits, n_informative=n_qubits,
                                  n_redundant=0, n_clusters_per_class=1, class_sep=2.5, random_state=RNG_DATA)
    ds['easy'] = (_to_angle_range(Xe), ye, dict(regime='simple', noise_injected=0.0))

    # 3. moons (nonlinear, intrinsically 2D) + nuisance dims to reach n_qubits
    Xm, ym = make_moons(n_samples=n_samples, noise=0.15, random_state=RNG_DATA)
    Xm = _add_nuisance_dims(Xm, n_qubits, rng)
    ds['moons'] = (_to_angle_range(Xm), ym, dict(regime='nonlinear', noise_injected=0.0))

    # 4. circles (nonlinear, intrinsically 2D) + nuisance dims
    Xc, yc = make_circles(n_samples=n_samples, noise=0.1, factor=0.4, random_state=RNG_DATA)
    Xc = _add_nuisance_dims(Xc, n_qubits, rng)
    ds['circles'] = (_to_angle_range(Xc), yc, dict(regime='nonlinear', noise_injected=0.0))

    # 5. blobs_overlap: natively supports arbitrary n_features
    Xb, yb = make_blobs(n_samples=n_samples, centers=2, n_features=n_qubits, cluster_std=2.2, random_state=RNG_DATA)
    ds['blobs_overlap'] = (_to_angle_range(Xb), yb, dict(regime='simple', noise_injected=0.0))

    # 6. label_noise_moons: moons (+nuisance dims) with 15% injected label noise
    Xln, yln = make_moons(n_samples=n_samples, noise=0.1, random_state=RNG_DATA + 1)
    Xln = _add_nuisance_dims(Xln, n_qubits, rng)
    flip_frac = 0.15
    n_flip = int(flip_frac * n_samples)
    flip_idx = rng.choice(n_samples, n_flip, replace=False)
    yln = yln.copy()
    yln[flip_idx] = 1 - yln[flip_idx]
    ds['label_noise_moons'] = (_to_angle_range(Xln), yln, dict(regime='noisy', noise_injected=flip_frac))

    # 7. high_dim_redundant: many redundant/irrelevant features, PCA-reduced to n_qubits
    Xh, yh = make_classification(n_samples=n_samples, n_features=12, n_informative=3, n_redundant=6,
                                  n_repeated=0, n_clusters_per_class=2, class_sep=1.2, random_state=RNG_DATA)
    Xh = _resize_features(Xh, n_qubits, rng)
    ds['high_dim_redundant'] = (_to_angle_range(Xh), yh, dict(regime='high_dim', noise_injected=0.0))

    # 8. breast_cancer: real-world dataset, native 30 dims, PCA to n_qubits
    bc = load_breast_cancer()
    idxb = rng.choice(len(bc.data), n_samples, replace=False)
    Xr = _resize_features(bc.data[idxb], n_qubits, rng)
    yr = bc.target[idxb]
    ds['breast_cancer'] = (_to_angle_range(Xr), yr, dict(regime='real_world', noise_injected=0.0))

    return ds


def geometric_difference(X, y, lam=1e-2):
    """Huang et al. style geometric difference between a fidelity-type quantum
    kernel (product over all encoded dimensions, matching the circuit's angle
    encoding) and a classical RBF kernel."""
    n = len(X)
    KQ = np.ones((n, n))
    for d in range(X.shape[1]):
        diff = X[:, d][:, None] - X[:, d][None, :]
        KQ *= np.cos(diff / 2) ** 2
    gamma = 1.0 / X.shape[1]
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    KC = np.exp(-gamma * sq)
    KC_reg = KC + lam * np.eye(n)
    KC_inv = np.linalg.inv(KC_reg)
    sqrtKQ = _sqrtm_psd(KQ)
    M = sqrtKQ @ KC_inv @ sqrtKQ
    eigvals = np.linalg.eigvalsh(M)
    g_qc = np.sqrt(max(eigvals.max(), 0))
    y_pm = 2 * y - 1
    sQ = np.sqrt(max(y_pm @ np.linalg.solve(KQ + lam * np.eye(n), y_pm), 0))
    sC = np.sqrt(max(y_pm @ np.linalg.solve(KC_reg, y_pm), 0))
    return g_qc, sQ, sC


def _sqrtm_psd(A):
    w, v = np.linalg.eigh(A)
    w = np.clip(w, 0, None)
    return (v * np.sqrt(w)) @ v.T


def task_hardness_features(X, y, noise_injected):
    knn = KNeighborsClassifier(n_neighbors=5)
    overlap_acc = cross_val_score(knn, X, y, cv=5).mean()
    t_overlap = 1 - overlap_acc

    lin = SVC(kernel='linear', C=1.0)
    lin.fit(X, y)
    w_norm = np.linalg.norm(lin.coef_)
    margin = 1.0 / w_norm if w_norm > 0 else 0.0
    t_margin_raw = margin

    lin_acc = cross_val_score(SVC(kernel='linear'), X, y, cv=5).mean()
    rbf_acc = cross_val_score(SVC(kernel='rbf'), X, y, cv=5).mean()
    t_nonlinearity = max(0.0, rbf_acc - lin_acc)

    t_noise = noise_injected

    return dict(t_overlap=t_overlap, t_margin_raw=t_margin_raw,
                t_nonlinearity=t_nonlinearity, t_noise=t_noise)
