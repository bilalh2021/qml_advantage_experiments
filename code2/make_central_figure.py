import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Aggregate numbers (mean, std) from model_report.json / model_report_4q.json
data = {
    '2 qubits': {
        'CV':   {'M2': (0.716, 0.046), 'M3': (0.729, 0.050)},
        'LODO': {'M2': (0.597, 0.199), 'M3': (0.530, 0.180)},
    },
    '4 qubits': {
        'CV':   {'M2': (0.691, 0.022), 'M3': (0.697, 0.022)},
        'LODO': {'M2': (0.589, 0.094), 'M3': (0.555, 0.087)},
    },
}

per_fold_2q = pd.read_csv('per_fold_lodo_2q.csv')
per_fold_4q = pd.read_csv('per_fold_lodo_4q.csv')

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

bar_width = 0.32
x = np.array([0, 1])  # CV, LODO
colors = {'M2': '#4C72B0', 'M3': '#C44E52'}

for ax, (qkey, per_fold_df) in zip(axes, [('2 qubits', per_fold_2q), ('4 qubits', per_fold_4q)]):
    for i, model in enumerate(['M2', 'M3']):
        means = [data[qkey]['CV'][model][0], data[qkey]['LODO'][model][0]]
        stds = [data[qkey]['CV'][model][1], data[qkey]['LODO'][model][1]]
        offset = (i - 0.5) * bar_width
        bars = ax.bar(x + offset, means, bar_width, yerr=stds, capsize=4,
                       label=f'{model} ({"additive" if model=="M2" else "interaction"})',
                       color=colors[model], alpha=0.85, edgecolor='black', linewidth=0.6)

    # overlay per-dataset LODO points, jittered, for both models
    rng = np.random.RandomState(0)
    for i, model in enumerate(['M2', 'M3']):
        vals = per_fold_df[f'{model}_auroc'].dropna().values
        offset = (i - 0.5) * bar_width
        jitter = rng.uniform(-bar_width * 0.32, bar_width * 0.32, size=len(vals))
        ax.scatter(np.full(len(vals), x[1] + offset) + jitter, vals,
                    color='black', s=18, zorder=5, alpha=0.65,
                    label='per-dataset LODO fold' if (i == 0) else None)

    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(['Within-distribution\nCV (5-fold)', 'Leave-one-dataset-out\n(LODO)'])
    ax.set_title(qkey)
    ax.set_ylim(0.15, 1.0)

axes[0].set_ylabel('AUROC')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)
fig.suptitle('')
plt.tight_layout()
plt.savefig('qml_cv_vs_lodo.png', dpi=200, bbox_inches='tight')
print('saved qml_cv_vs_lodo.png')
