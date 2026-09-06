import numpy as np, pandas as pd, json
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler

np.random.seed(0)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1] if i < n_bins - 1 else y_prob <= bins[i + 1])
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_prob)) * abs(acc - conf)
    return ece


def make_features(df, axes):
    X = df[axes].values.astype(float)
    if len(axes) > 1:
        # add pairwise interaction terms
        inter_cols = []
        inter_names = []
        for i in range(len(axes)):
            for j in range(i + 1, len(axes)):
                inter_cols.append(X[:, i] * X[:, j])
                inter_names.append(f'{axes[i]}*{axes[j]}')
        Xi = np.column_stack([X] + inter_cols) if inter_cols else X
        return Xi, axes + inter_names
    return X, axes


def cv_metrics(df, feature_cols, y_col, n_splits=5, seed=0):
    X = df[feature_cols].values.astype(float)
    y = df[y_col].values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aurocs, auprcs, briers, eces = [], [], [], []
    for tr, te in skf.split(X, y):
        if len(np.unique(y[tr])) < 2:
            continue
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(X[tr]), y[tr])
        p = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        if len(np.unique(y[te])) < 2:
            continue
        aurocs.append(roc_auc_score(y[te], p))
        auprcs.append(average_precision_score(y[te], p))
        briers.append(brier_score_loss(y[te], p))
        eces.append(expected_calibration_error(y[te], p))
    return dict(auroc=(np.mean(aurocs), np.std(aurocs)),
                auprc=(np.mean(auprcs), np.std(auprcs)),
                brier=(np.mean(briers), np.std(briers)),
                ece=(np.mean(eces), np.std(eces)), n_folds=len(aurocs))


def lodo_metrics(df, feature_cols, y_col):
    """Leave-one-DATASET-out: train on 7 datasets, test on the held-out one, per dataset."""
    datasets = df['dataset'].unique()
    aurocs, auprcs, briers = [], [], []
    per_dataset = {}
    for held_out in datasets:
        train_df = df[df['dataset'] != held_out]
        test_df = df[df['dataset'] == held_out]
        Xtr = train_df[feature_cols].values.astype(float)
        ytr = train_df[y_col].values
        Xte = test_df[feature_cols].values.astype(float)
        yte = test_df[y_col].values
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            per_dataset[held_out] = None
            continue
        scaler = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000).fit(scaler.transform(Xtr), ytr)
        p = clf.predict_proba(scaler.transform(Xte))[:, 1]
        auc = roc_auc_score(yte, p)
        aurocs.append(auc)
        auprcs.append(average_precision_score(yte, p))
        briers.append(brier_score_loss(yte, p))
        per_dataset[held_out] = auc
    return dict(auroc=(np.mean(aurocs), np.std(aurocs)) if aurocs else (None, None),
                auprc=(np.mean(auprcs), np.std(auprcs)) if auprcs else (None, None),
                brier=(np.mean(briers), np.std(briers)) if briers else (None, None),
                n_datasets_evaluable=len(aurocs), per_dataset=per_dataset)


def main():
    df = pd.read_csv('master_table_4q.csv')
    y_col = 'Y_0.6'
    df[y_col] = (df['test_acc'] > 0.60).astype(int)

    base_axes = ['T_hard', 'D_axis', 'A_axis', 'H_axis']

    report = {}

    # M1: single-axis models
    report['M1_single_axis'] = {}
    for ax in base_axes:
        cols, names = make_features(df, [ax])
        df_tmp = df.copy()
        cv = cv_metrics(df_tmp, [ax], y_col)
        lodo = lodo_metrics(df_tmp, [ax], y_col)
        report['M1_single_axis'][ax] = dict(cv=cv, lodo=lodo)

    # M2: additive (main effects only)
    cv2 = cv_metrics(df, base_axes, y_col)
    lodo2 = lodo_metrics(df, base_axes, y_col)
    report['M2_additive'] = dict(cv=cv2, lodo=lodo2)

    # M3: full pairwise interaction model
    Xi, names = make_features(df, base_axes)
    for k, name in enumerate(names):
        df[f'feat_{k}'] = Xi[:, k]
    feat_cols = [f'feat_{k}' for k in range(len(names))]
    cv3 = cv_metrics(df, feat_cols, y_col)
    lodo3 = lodo_metrics(df, feat_cols, y_col)
    report['M3_interaction'] = dict(cv=cv3, lodo=lodo3, feature_names=names)

    # Ablations: drop one axis group at a time from M3
    report['ablations'] = {}
    for drop_ax in base_axes:
        keep_axes = [a for a in base_axes if a != drop_ax]
        Xi_a, names_a = make_features(df, keep_axes)
        for k, name in enumerate(names_a):
            df[f'abl_{k}'] = Xi_a[:, k]
        feat_cols_a = [f'abl_{k}' for k in range(len(names_a))]
        cv_a = cv_metrics(df, feat_cols_a, y_col)
        lodo_a = lodo_metrics(df, feat_cols_a, y_col)
        report['ablations'][f'drop_{drop_ax}'] = dict(cv=cv_a, lodo=lodo_a)

    # threshold sensitivity for M2 vs M3 (LODO AUROC)
    report['threshold_sensitivity'] = {}
    for thresh in [0.55, 0.60, 0.65, 0.70]:
        yc = f'Y_{thresh}'
        df[yc] = (df['test_acc'] > thresh).astype(int)
        l2 = lodo_metrics(df, base_axes, yc)
        l3 = lodo_metrics(df, feat_cols, yc)
        report['threshold_sensitivity'][str(thresh)] = dict(M2_lodo_auroc=l2['auroc'], M3_lodo_auroc=l3['auroc'])

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if o is None:
            return None
        return o

    with open('model_report_4q.json', 'w') as f:
        json.dump(clean(report), f, indent=2)

    print(json.dumps(clean(report), indent=2)[:4000])


if __name__ == '__main__':
    main()
