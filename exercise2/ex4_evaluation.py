import numpy as np
import pickle
import matplotlib.pyplot as plt

result = pickle.load(open("results_random_search.pkl", "rb"))
inc_runs = result.get_all_runs()

accs = []
best_accs = []
for r in inc_runs:
    acc = r.info['validation_accuracy']
    accs.append(acc)

    # best accuracy after x iterations
    best_accs.append(np.max(accs))

# plot best accuracies per iteration
plt.plot(best_accs)
plt.xlabel("random search iteration")
plt.ylabel("best validation accuracy")
plt.savefig("iteration_history.png")
