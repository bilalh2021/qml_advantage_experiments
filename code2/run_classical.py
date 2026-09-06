import numpy as np, csv
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from datasets import build_datasets

MODELS = {
    'logreg': LogisticRegression(max_iter=1000),
    'svm_linear': SVC(kernel='linear'),
    'svm_rbf': SVC(kernel='rbf'),
    'random_forest': RandomForestClassifier(n_estimators=100, random_state=0),
    'gradient_boosting': GradientBoostingClassifier(random_state=0),
    'mlp': MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=0),
}


def main():
    datasets = build_datasets(n_qubits=4)
    rows = []
    for dname, (X, y, meta) in datasets.items():
        best_acc, best_model = 0, None
        for mname, model in MODELS.items():
            acc = cross_val_score(model, X, y, cv=5).mean()
            rows.append(dict(dataset=dname, model=mname, cv_acc=acc))
            if acc > best_acc:
                best_acc, best_model = acc, mname
        rows.append(dict(dataset=dname, model='BEST', cv_acc=best_acc, best_model=best_model))

    with open('results_classical_4q.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'model', 'cv_acc', 'best_model'])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print('wrote results_classical.csv')
    for r in rows:
        if r['model'] == 'BEST':
            print(r['dataset'], 'best classical acc =', round(r['cv_acc'], 3), 'via', r['best_model'])


if __name__ == '__main__':
    main()
