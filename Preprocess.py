from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
import sklearn
from sklearn.preprocessing import OneHotEncoder

from imblearn.combine import SMOTETomek

def create_prep_pipeline(numeric_features,categorical_features, feature_topk = "all", random_state = None):
    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    col_transfer = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    resampler = SMOTETomek(random_state=random_state)
    feature_selector = SelectKBest(chi2, k=feature_topk)

    preprocesssor = Pipeline(
        steps=[
            ("column_transfer", col_transfer),
            ("resampler", resampler),
            ("feature_selector", feature_selector),
        ]
    )
    
    return preprocesssor
