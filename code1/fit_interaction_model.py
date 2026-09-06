"""
First real attempt at Equation 2 (the interaction-based logistic model)
from the paper, tested against single-axis baselines on the actual
experimental data collected in qml_advantage_experiment.py's sweeps.

This directly operationalizes referee comments 3, 4, 5, 12: fit the
combined/interaction model, compare against single-axis baselines, and
report whether D x H carries predictive information beyond D and H alone.

Axis scores used (necessarily simplified -- see caveats printed at the
end):
  D (data axis)  : g_QC_normalized from geometric_difference.py, at
                   reg=1e-2 (a middle regularization value), min-max
                   scaled to [0,1] across the two datasets.
  H (hardware axis): 1 - (noise_p / max_noise_p), so higher = more
                   favorable hardware condition (less noise).
  A (architecture axis): n_layers / max_layers (raw, direction not
                   asserted -- this is a placeholder, not the
                   trainability-dequantizability-derived tau(A) from
                   the paper).
  T (task axis)  : not available from this experiment (no task-hardness
                   proxy was computed); held out of the model, noted as
                   a limitation.

Outcome Y: binarized accuracy, Y=1 if accuracy > 0.6 (chosen because the
observed "collapse" regime lands at accuracy=0.5, chance level for this
binary task, and the robust regime sits at 0.7-1.0 -- 0.6 cleanly
separates the two clusters in the data without being fit post-hoc to
any single point).
"""

import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss

# ---------------------------------------------------------------------
# Load and combine all experimental runs
# ---------------------------------------------------------------------
df1 = pd.read_json("sweep_results.jsonl", lines=True); df1["seed"] = 42
df2 = pd.read_json("fullgrid_seed_results.jsonl", lines=True)
df3 = pd.read_json("seed_results.jsonl", lines=True)
df = pd.concat([df1, df2, df3], ignore_index=True)

# ---------------------------------------------------------------------
# Axis scores
# ---------------------------------------------------------------------
# D: from the geometric-difference results computed earlier (reg=1e-2)
G_QC = {"structured": 2.81, "easy": 1.49}  # from geometric_difference_results.csv
g_min, g_max = min(G_QC.values()), max(G_QC.values())
D_SCORE = {k: (v - g_min) / (g_max - g_min) for k, v in G_QC.items()}
print("D scores:", D_SCORE)

df["D"] = df["dataset"].map(D_SCORE)
df["H"] = 1 - (df["noise_p"] / df["noise_p"].max())
df["A"] = (df["n_layers"] - df["n_layers"].min()) / (df["n_layers"].max() - df["n_layers"].min())
df["Y"] = (df["accuracy"] > 0.6).astype(int)

print(df[["dataset", "n_layers", "noise_p", "accuracy", "D", "H", "A", "Y"]].to_string())
print("\nOutcome balance:", df["Y"].value_counts().to_dict())

# ---------------------------------------------------------------------
# Fit models: single-axis baselines vs. the interaction model
# ---------------------------------------------------------------------
def fit_and_eval(df, cols, label):
    X = sm.add_constant(df[cols])
    y = df["Y"]
    try:
        model = sm.Logit(y, X).fit(disp=0)
        pred_prob = model.predict(X)
        auc = roc_auc_score(y, pred_prob)
        acc = accuracy_score(y, (pred_prob > 0.5).astype(int))
        ll = log_loss(y, pred_prob)
        pseudo_r2 = model.prsquared
        print(f"\n--- {label} ---")
        print(model.summary().tables[1])
        print(f"AUROC={auc:.3f}  Accuracy={acc:.3f}  LogLoss={ll:.3f}  PseudoR2={pseudo_r2:.3f}")
        return {"model": label, "auroc": auc, "accuracy": acc, "log_loss": ll, "pseudo_r2": pseudo_r2, "n_params": len(cols)+1}
    except Exception as e:
        print(f"\n--- {label} FAILED: {e} ---")
        return {"model": label, "auroc": np.nan, "accuracy": np.nan, "log_loss": np.nan, "pseudo_r2": np.nan, "n_params": len(cols)+1}

results = []
results.append(fit_and_eval(df, ["D"], "Single-axis: D only"))
results.append(fit_and_eval(df, ["H"], "Single-axis: H only"))
results.append(fit_and_eval(df, ["A"], "Single-axis: A only"))
results.append(fit_and_eval(df, ["D", "H", "A"], "Additive: D + H + A"))

df["DH"] = df["D"] * df["H"]
df["AH"] = df["A"] * df["H"]
df["DA"] = df["D"] * df["A"]
results.append(fit_and_eval(df, ["D", "H", "A", "DH"], "Interaction: D + H + A + D*H"))
results.append(fit_and_eval(df, ["D", "H", "A", "DH", "AH", "DA"], "Full interaction: all pairwise"))

results_df = pd.DataFrame(results)
results_df.to_csv("/mnt/user-data/outputs/interaction_model_comparison_UNREGULARIZED_DO_NOT_USE.csv", index=False)
print("\n\n=== IN-SAMPLE UNREGULARIZED SUMMARY (see warnings above -- likely separation artifacts) ===")
print(results_df.to_string())

# -----------------------------------------------------------------
# The unregularized fits above show AUROC=1.0 with huge, non-converged
# coefficients -- the signature of complete/quasi-complete separation,
# not genuine predictive power, given only 10/64 outcomes are Y=0.
# Redo properly: L2-regularized logistic regression, evaluated by
# stratified k-fold cross-validation (honest out-of-sample performance,
# not in-sample fit).
# -----------------------------------------------------------------
print("\n\n" + "=" * 70)
print("PROPER (regularized, cross-validated) COMPARISON")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

def cv_eval(df, cols, label, n_splits=5):
    X = df[cols].values
    y = df["Y"].values
    if len(np.unique(y)) < 2:
        return {"model": label, "cv_auroc_mean": np.nan, "cv_auroc_std": np.nan, "n_params": len(cols)}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=1000)
    try:
        aurocs = cross_val_score(clf, X, y, cv=skf, scoring="roc_auc")
        print(f"{label}: CV AUROC = {aurocs.mean():.3f} +/- {aurocs.std():.3f}  (folds: {np.round(aurocs,3)})")
        return {"model": label, "cv_auroc_mean": aurocs.mean(), "cv_auroc_std": aurocs.std(), "n_params": len(cols)}
    except Exception as e:
        print(f"{label}: FAILED ({e})")
        return {"model": label, "cv_auroc_mean": np.nan, "cv_auroc_std": np.nan, "n_params": len(cols)}

cv_results = []
cv_results.append(cv_eval(df, ["D"], "Single-axis: D only"))
cv_results.append(cv_eval(df, ["H"], "Single-axis: H only"))
cv_results.append(cv_eval(df, ["A"], "Single-axis: A only"))
cv_results.append(cv_eval(df, ["D", "H", "A"], "Additive: D + H + A"))
cv_results.append(cv_eval(df, ["D", "H", "A", "DH"], "Interaction: D + H + A + D*H"))
cv_results.append(cv_eval(df, ["D", "H", "A", "DH", "AH", "DA"], "Full interaction: all pairwise"))

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv("/mnt/user-data/outputs/interaction_model_comparison_CV.csv", index=False)
print("\n=== CROSS-VALIDATED SUMMARY (honest generalization estimate) ===")
print(cv_df.to_string())
