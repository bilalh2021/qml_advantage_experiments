import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

np.random.seed(0)
N_BOOT = 2000
BASE_AXES = ['T_hard', 'D_axis', 'A_axis', 'H_axis']


def make_features(df, axes):
    X = df[axes].values.astype(float)
    inter_cols, inter_names = [], []
    for i in range(len(axes)):
        for j in range(i + 1, len(axes)):
            inter_cols.append(X[:, i] * X[:, j])
            inter_names.append(f'{axes[i]}*{axes[j]}')
    Xi = np.column_stack([X] + inter_cols) if inter_cols else X
    return Xi, axes + inter_names


def per_fold_lodo(master_path, y_thresh, label):
    df = pd.read_csv(master_path)
    df['Y'] = (df['test_acc'] > y_thresh).astype(int)

    Xi, names = make_features(df, BASE_AXES)
    for k, name in enumerate(names):
        df[f'feat_{k}'] = Xi[:, k]
    feat_cols = [f'feat_{k}' for k in range(len(names))]

    rows = []
    for held_out in sorted(df['dataset'].unique()):
        train_df = df[df['dataset'] != held_out]
        test_df = df[df['dataset'] == held_out]
        n_test = len(test_df)
        y_te = test_df['Y'].values
        base_rate = y_te.mean()

        row = dict(dataset=held_out, n_configs=n_test, base_rate=base_rate)

        if len(np.unique(y_te)) < 2:
            row['M2_auroc'] = None
            row['M3_auroc'] = None
            row['degenerate'] = True
            rows.append(row)
            continue
        row['degenerate'] = False

        for model_name, cols in [('M2', BASE_AXES), ('M3', feat_cols)]:
            Xtr = train_df[cols].values.astype(float)
            ytr = train_df['Y'].values
            Xte = test_df[cols].values.astype(float)

            scaler = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
            p_te = clf.predict_proba(scaler.transform(Xte))[:, 1]
            point_auc = roc_auc_score(y_te, p_te)

            # bootstrap CI: resample the held-out fold's test rows (fixed trained classifier)
            boot_aucs = []
            n = len(y_te)
            for _ in range(N_BOOT):
                idx = np.random.randint(0, n, n)
                if len(np.unique(y_te[idx])) < 2:
                    continue
                boot_aucs.append(roc_auc_score(y_te[idx], p_te[idx]))
            if boot_aucs:
                lo, hi = np.percentile(boot_aucs, [2.5, 97.5])
            else:
                lo, hi = (None, None)

            row[f'{model_name}_auroc'] = point_auc
            row[f'{model_name}_ci_lo'] = lo
            row[f'{model_name}_ci_hi'] = hi

        rows.append(row)

    result_df = pd.DataFrame(rows)
    print(f'=== {label} ===')
    print(result_df.to_string(index=False))
    return result_df


if __name__ == '__main__':
    df2q = per_fold_lodo('master_table.csv', 0.60, '2 qubits (n=800, 8 datasets)')
    df2q.to_csv('per_fold_lodo_2q.csv', index=False)
    print()
    df4q = per_fold_lodo('master_table_4q.csv', 0.60, '4 qubits (n=800, 8 datasets)')
    df4q.to_csv('per_fold_lodo_4q.csv', index=False)
