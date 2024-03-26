import seaborn as sns
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    cohen_kappa_score,
    make_scorer,
)


def explain_permutation(model, features, labels, repeats=20, topn=20):
    result = permutation_importance(
        model,
        features,
        labels,
        n_repeats=repeats,
        n_jobs=-1,
        scoring=make_scorer(cohen_kappa_score),
    )

    fdf = pd.DataFrame({"features": features.columns, "importance": result.importances_mean})
    fdf.sort_values(by="importance", ascending=False, inplace=True)

    sns.barplot(fdf[:topn], y="features", x="importance", hue="features", legend=False)
