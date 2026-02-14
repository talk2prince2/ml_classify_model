import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from src.preprocess import build_preprocessing_pipeline


def train_models(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    preprocessor = build_preprocessing_pipeline(df, target_column)

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier(),
        "knn": KNeighborsClassifier(),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10,random_state=42),
        "xgboost": XGBClassifier(eval_metric='logloss')
    }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    os.makedirs("../models", exist_ok=True)

    trained_models = {}

    for name, model in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        joblib.dump(pipeline, f"../models/{name}.pkl")

        trained_models[name] = pipeline

        print(f"{name} trained and saved.")

    return trained_models, X_test, y_test
