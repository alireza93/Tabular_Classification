import seaborn as sns
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    cohen_kappa_score,
    make_scorer,
)
import shap


def explain_permutation(model, features, labels, repeats=20, topn=20):
    result = permutation_importance(
        model,
        features,
        labels,
        n_repeats=repeats,
        n_jobs=-1,
        scoring=make_scorer(cohen_kappa_score),
    )

    fdf = pd.DataFrame(
        {"features": features.columns, "importance": result.importances_mean}
    )
    fdf.sort_values(by="importance", ascending=False, inplace=True)

    sns.barplot(fdf[:topn], y="features", x="importance", hue="features", legend=False)


def shap_init(clsf, X, Y):
    expl = shap.TreeExplainer(clsf)
    shap_values = expl.shap_values(X, Y)
    return expl, shap_values


def shap_summary_pc(shap_values, X, class_idx):
    shap.summary_plot(shap_values[class_idx], X, max_display=20, plot_size=[12, 10])

def shap_summary_bar(shap_values, X, class_names):
    shap.summary_plot(shap_values, X, plot_type="bar", class_names=class_names)


def shap_dependence(
    shap_values,
    data,
    class_idx,
    feature1,
    feature2="auto",
):
    shap.dependence_plot(
        feature1,
        shap_values[class_idx],
        data,
        display_features=data,
        interaction_index=feature2,
    )


def shap_exp_sample(expl, shap_values, data, sample_idx, class_idx):
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[class_idx][sample_idx],
            base_values=expl.expected_value[class_idx],
            data=data.iloc[sample_idx],
            feature_names=data.columns.tolist(),
        )
    )
