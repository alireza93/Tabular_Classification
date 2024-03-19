import time
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy.stats as stats


def print_report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")




def grid_search(model, param_grid, train_features, train_labels):
    grid_search = GridSearchCV(
        model,
        param_grid=param_grid,
    )

    start = time.time()
    grid_search.fit(train_features, train_labels)
    print(
        "GridSearchCV took %.2f seconds for %d candidate parameter settings."
        % (time.time() - start, len(grid_search.cv_results_["params"]))
    )
    print_report(grid_search.cv_results_, n_top=10)
