import sys, json, itertools
import qml_advantage_experiment as m

m.N_TRAIN_STEPS = 25
m.N_TRAIN_SAMPLES = 30
m.N_TEST_SAMPLES = 20

DATASETS = ["moons", "circles"]
LAYERS = [1, 3]
NOISE = [0.0, 0.10, 0.20, 0.30]

GRID = list(itertools.product(DATASETS, LAYERS, NOISE))

def run_indices(indices, out_path):
    results = []
    for idx in indices:
        dataset, layers, noise_p = GRID[idx]
        row = m.train_and_evaluate(dataset, layers, noise_p)
        results.append(row)
        print(f"[{idx}] {row}")
    with open(out_path, "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

if __name__ == "__main__":
    start, end = int(sys.argv[1]), int(sys.argv[2])
    run_indices(list(range(start, end)), "/home/claude/newdatasets_results.jsonl")
