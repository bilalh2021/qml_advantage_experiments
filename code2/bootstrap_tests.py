import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
N_BOOT = 5000

# ---------- Part 1: LODO paired-dataset bootstrap ----------

def lodo_bootstrap(report_path, model_a, model_b, label):
    r = json.load(open(report_path))

    def get_pd(name):
        if name in r['M1_single_axis']:
            return r['M1_single_axis'][name]['lodo']['per_dataset']
        return r[name]['lodo']['per_dataset']

    pa = get_pd(model_a)
    pb = get_pd(model_b)
    common = sorted([d for d in pa if pa[d] is not None and pb[d] is not None and d in pb])
    a = np.array([pa[d] for d in common])
    b = np.array([pb[d] for d in common])
    diffs = a - b
    obs_diff = diffs.mean()

    boot_diffs = []
    n = len(diffs)
    for _ in range(N_BOOT):
        idx = np.random.randint(0, n, n)
        boot_diffs.append(diffs[idx].mean())
    boot_diffs = np.array(boot_diffs)
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])
    p_two_sided = 2 * min((boot_diffs > 0).mean(), (boot_diffs < 0).mean())
    p_two_sided = min(p_two_sided, 1.0)
    print(f'{label}: n_datasets={n}, mean diff={obs_diff:+.3f}, 95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}], '
          f'two-sided bootstrap p={p_two_sided:.3f}')
    return dict(label=label, n=n, mean_diff=obs_diff, ci=(ci_lo, ci_hi), p=p_two_sided)


# ---------- Part 2: CV row-level bootstrap ----------

def make_features(df, axes):
    X = df[axes].values.astype(float)
    inter_cols, inter_names = [], []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            inter_cols.append(X[:, i] * X[:, j])
            inter_names.append(f'{axes[i]}*{axes[j]}')
    Xi = np.column_stack([X] + inter_cols) if inter_cols else X
    return Xi, axes + inter_names


def cv_auroc_once(df, feature_cols, y_col, seed):
    X = df[feature_cols].values.astype(float)
    y = df[y_col].values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
        p = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return np.mean(aucs) if aucs else np.nan


def cv_bootstrap(master_path, label, n_boot=300):
    df = pd.read_csv(master_path)
    df['Y_0.6'] = (df['test_acc'] > 0.60).astype(int)
    base_axes = ['T_hard', 'D_axis', 'A_axis', 'H_axis']

    Xi, names = make_features(df, base_axes)
    for k, name in enumerate(names):
        df[f'feat_{k}'] = Xi[:, k]
    feat_cols = [f'feat_{k}' for k in range(len(names))]

    diffs = []
    n = len(df)
    for b in range(n_boot):
        idx = np.random.randint(0, n, n)
        dfb = df.iloc[idx].reset_index(drop=True)
        auc_m2 = cv_auroc_once(dfb, base_axes, 'Y_0.6', seed=b)
        auc_m3 = cv_auroc_once(dfb, feat_cols, 'Y_0.6', seed=b)
        if not (np.isnan(auc_m2) or np.isnan(auc_m3)):
            diffs.append(auc_m3 - auc_m2)
    diffs = np.array(diffs)
    obs_diff = diffs.mean()
    ci_lo, ci_hi = np.percentile(diffs, [2.5, 97.5])
    p_two_sided = min(2 * min((diffs > 0).mean(), (diffs < 0).mean()), 1.0)
    print(f'{label}: CV M3-M2 bootstrap ({len(diffs)} draws): mean diff={obs_diff:+.3f}, '
          f'95% CI=[{ci_lo:+.3f}, {ci_hi:+.3f}], two-sided p={p_two_sided:.3f}')
    return dict(label=label, mean_diff=obs_diff, ci=(ci_lo, ci_hi), p=p_two_sided, n_boot=len(diffs))


if __name__ == '__main__':
    results = {}

    print('=== LODO paired-dataset bootstrap (2 qubits) ===')
    results['lodo_2q_M3_vs_M2'] = lodo_bootstrap('model_report.json', 'M3_interaction', 'M2_additive', '2q: M3 - M2 (LODO)')
    results['lodo_2q_M3_vs_H'] = lodo_bootstrap('model_report.json', 'M3_interaction', 'H_axis', '2q: M3 - H_only (LODO)')
    results['lodo_2q_M2_vs_H'] = lodo_bootstrap('model_report.json', 'M2_additive', 'H_axis', '2q: M2 - H_only (LODO)')

    print()
    print('=== LODO paired-dataset bootstrap (4 qubits) ===')
    results['lodo_4q_M3_vs_M2'] = lodo_bootstrap('model_report_4q.json', 'M3_interaction', 'M2_additive', '4q: M3 - M2 (LODO)')
    results['lodo_4q_M3_vs_A'] = lodo_bootstrap('model_report_4q.json', 'M3_interaction', 'A_axis', '4q: M3 - A_only (LODO)')
    results['lodo_4q_M2_vs_A'] = lodo_bootstrap('model_report_4q.json', 'M2_additive', 'A_axis', '4q: M2 - A_only (LODO)')

    print()
    print('=== CV row-level bootstrap ===')
    results['cv_2q_M3_vs_M2'] = cv_bootstrap('master_table.csv', '2q')
    results['cv_4q_M3_vs_M2'] = cv_bootstrap('master_table_4q.csv', '4q')

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, tuple):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        return o

    with open('significance_report.json', 'w') as f:
        json.dump(clean(results), f, indent=2)
