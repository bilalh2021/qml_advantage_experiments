import numpy as np, pandas as pd, json
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from datasets import build_datasets, geometric_difference, task_hardness_features

ARCH_COMPLEXITY = {  # pre-specified structural complexity score (layers x entangling-pair count), not fit to outcomes
    'shallow_hea': 1 * 2,      # 1 layer, ring (2 pairs on 2 qubits -> 2 directed, use 2)
    'reupload': 2 * 2,         # 2 layers ring, with reuploading
    'deep_hea': 3 * 2,         # 3 layers ring
    'strong_entangle': 2 * 2,  # 2 layers all-to-all (2 qubits: all2all==ring, so same connectivity, more expressive rotations)
}
# reupload gets an extra structural increment for the repeated encoding
ARCH_COMPLEXITY_RAW = {'shallow_hea': 2, 'reupload': 5, 'deep_hea': 6, 'strong_entangle': 4}

NOISE_MAX = 0.30  # normalization ceiling shared across channels (severity assumed comparable; see limitations)


def build_axis_table(n_qubits=4):
    datasets = build_datasets(n_qubits=n_qubits)
    rows = []
    for dname, (X, y, meta) in datasets.items():
        g_qc, sQ, sC = geometric_difference(X, y)
        th = task_hardness_features(X, y, meta['noise_injected'])
        rows.append(dict(dataset=dname, regime=meta['regime'], g_qc=g_qc, sQ=sQ, sC=sC, **th))
    return pd.DataFrame(rows)


def main():
    axis_df = build_axis_table()

    # normalize task-hardness components to [0,1] across datasets, then combine with
    # PRE-SPECIFIED equal weights (not fit to any QML outcome)
    for col in ['t_overlap', 't_margin_raw', 't_nonlinearity', 't_noise']:
        mn, mx = axis_df[col].min(), axis_df[col].max()
        axis_df[col + '_n'] = 0.0 if mx == mn else (axis_df[col] - mn) / (mx - mn)
    axis_df['t_margin_n'] = 1 - axis_df['t_margin_raw_n']  # invert: smaller margin -> harder
    axis_df['T_hard'] = axis_df[['t_overlap_n', 't_margin_n', 't_nonlinearity_n', 't_noise_n']].mean(axis=1)

    g_mn, g_mx = axis_df['g_qc'].min(), axis_df['g_qc'].max()
    axis_df['D_axis'] = (axis_df['g_qc'] - g_mn) / (g_mx - g_mn) if g_mx > g_mn else 0.0

    qml = pd.read_csv('results_qml_4q.csv')
    qml['A_axis'] = qml['architecture'].map(ARCH_COMPLEXITY_RAW)
    a_mn, a_mx = min(ARCH_COMPLEXITY_RAW.values()), max(ARCH_COMPLEXITY_RAW.values())
    qml['A_axis'] = (qml['A_axis'] - a_mn) / (a_mx - a_mn)
    qml['H_axis'] = 1 - (qml['noise_p'] / NOISE_MAX)  # 1 = clean hardware, 0 = worst noise tested

    df = qml.merge(axis_df[['dataset', 'T_hard', 'D_axis', 'regime']], on='dataset', how='left')

    df.to_csv('master_table_4q.csv', index=False)
    print('rows:', len(df))
    print(df[['dataset', 'architecture', 'noise_model', 'noise_p', 'T_hard', 'D_axis', 'A_axis', 'H_axis', 'test_acc']].head(8))

    results = {}
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        df[f'Y_{thresh}'] = (df['test_acc'] > thresh).astype(int)
        results[f'base_rate_{thresh}'] = float(df[f'Y_{thresh}'].mean())

    with open('axis_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

    axis_df.to_csv('axis_table_4q.csv', index=False)


if __name__ == '__main__':
    main()
