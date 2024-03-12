from sklearn.base import accuracy_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score
import Models, Preprocess

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt


RANDOM_STATE = 50


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


if __name__ == "main":
    main()
