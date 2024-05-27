import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
import numpy as np
import pandas as pd
import os
import pickle

OUTPUT_FOLDER = "./out/"

with open(os.path.join(OUTPUT_FOLDER, "model.pkl"), "rb") as f:
    [train_features, train_labels, test_features, test_labels, model] = (
        pickle.load(f)
    )


def vis_class_imbalance(labels):
    autopct = "%.1f"
    labels_df = pd.DataFrame({"target": labels})
    labels_df.value_counts().plot.pie(autopct=autopct)
    plt.show()

def feature_target_relation(df, feature_sel, target):
    sns.swarmplot(
    df,
    y=feature_sel,
    x=target,
    size=1.5,
    orient="v",
    warn_thresh=0.1,
    )
