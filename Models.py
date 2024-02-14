from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 50

clsf_mlp = MLPClassifier(
    solver="sgd",
    alpha=1e-5,
    hidden_layer_sizes=(50,),
    max_iter=1500,
)
clsf_rf = RandomForestClassifier(
    n_estimators=200,
    min_samples_split=2,
    n_jobs=-1,
    criterion="gini",
    random_state=RANDOM_STATE,
)
clsf_hgb = HistGradientBoostingClassifier(
    min_samples_leaf=1, learning_rate=0.12, random_state=RANDOM_STATE
)
