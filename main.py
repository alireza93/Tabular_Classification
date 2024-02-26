import Models, Preprocess

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 50

target_label_name = "y_label"
df = pd.read_csv("./data/dataset.csv")
feature_names = df.drop(columns=[target_label_name]).columns
labels = df[target_label_name]
classes = np.unique(labels)

cat_cols = pd.read_csv("./data/categorical_cols.csv", index_col=0)["name"].to_numpy()
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


