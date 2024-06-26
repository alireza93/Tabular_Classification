from sklearn.base import accuracy_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
import Models, Preprocess, Explain
from Parameter_tuning import grid_search

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_validate, StratifiedKFold


RANDOM_STATE = 50
OUTPUT_FOLDER = "./out"

def evaluate(model, test_features, test_labels):
    preds = model.predict(test_features)
    probs = model.predict_proba(test_features)
    acc = accuracy_score(test_labels, preds)
    kappa = cohen_kappa_score(test_labels, preds, weights="quadratic")
    roc_auc = roc_auc_score(
        test_labels,
        probs,
        multi_class="ovr",
        average=None,
    )
    cfmat = confusion_matrix(test_labels, preds)
    print(
        f"Accuracy: {acc:.4f},\nKappa: {kappa:.4f},\nROC AUC One-vs-Rest: {np.array2string(roc_auc, precision=2, floatmode='fixed')}"
    )
    print(f"Confusion Matrix:\n{cfmat}")
    display_conf_mat(cfmat)


def display_conf_mat(cfmat):
    plt.title("Confusion matrix")
    sns.heatmap(cfmat, cmap="Blues", annot=True, fmt="g", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.show()


def cross_validation(train_features, train_labels,kfold=5):

    stratified_cv = StratifiedKFold(n_splits=kfold, shuffle=True)
    cross_val_result = cross_validate(
        model,
        train_features,
        train_labels,
        cv=stratified_cv,
        scoring="accuracy",
        n_jobs=-1,
        return_estimator=True,
    )
    print(
        f"{kfold}fold cross validation accuracy average: {cross_val_result['test_score'].mean()}"
    )
    model = cross_val_result["estimator"][0]
    return model


def main():
    target_label_name = "y_label"
    df = pd.read_csv("./data/dataset.csv")
    feature_names = df.drop(columns=[target_label_name]).columns
    labels = df[target_label_name]
    classes = np.unique(labels)

    cat_cols = pd.read_csv("./data/categorical_cols.csv", index_col=0)[
        "name"
    ].to_numpy()
    cat_col_names = [x for x in feature_names if x in cat_cols]
    num_col_names = [x for x in feature_names if x not in cat_cols]

    train_features, test_features, train_labels, test_labels = train_test_split(
        df,
        labels,
        test_size=0.25,
        shuffle=True,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    prep = Preprocess.create_prep_pipeline(num_col_names,cat_col_names,"all",RANDOM_STATE)
    model = Models.create_model_pipeline(prep,Models.clsf_rf)

    model.fit(train_features, train_labels)

    evaluate(model,test_features,test_labels)

    param_grid = {
        "preprocessor__feature_selector__k": np.linspace(50, 200, num=4).astype(int),
        "classifier__n_estimators": np.linspace(50, 750, num=25).astype(int),
        "classifier__criterion": ["gini", "entropy"],
    }

    grid_search(model, param_grid, train_features, train_labels)
    Explain.explain_permutation(model,test_features,test_labels,20,10)

    # Saving data sets and the fitted model:
    with open(os.path.join(OUTPUT_FOLDER, "model.pkl"), "wb") as f:
        pickle.dump(
            [train_features, train_labels, test_features, test_labels, model],
            f,
        )

if __name__ == "main":
    main()
