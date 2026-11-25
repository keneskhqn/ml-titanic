import tempfile
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


mlflow.set_tracking_uri("sqlite:///mlflow.db")
EXPERIMENT_NAME = "titanic-survival"
REGISTERED_MODEL_NAME = "titanic-model"
mlflow.set_experiment(EXPERIMENT_NAME)  

TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv('data/Titanic-Dataset.csv')
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(df.head())

categorical_features = ["Sex", "Embarked"]
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

X = df[categorical_features + numeric_features]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


def build_model(n_estimators, max_depth):
    """Preprocessing + Random Forest in one pipeline (same as before)."""
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_features),
            ("num", numeric_pipeline, numeric_features),
        ]
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=RANDOM_STATE)),
    ])


experiments = [
    {"n_estimators": 100, "max_depth": 3},
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
]

# Model used by the API (model.joblib). depth 5 is chosen over depth 10:
# almost the same test scores, but much less overfitting (train 0.86 vs 0.95).
API_MAX_DEPTH = 5
chosen = None

for params in experiments:
    run_name = f"rf_depth_{params['max_depth']}"
    with mlflow.start_run(run_name=run_name):
        model = build_model(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_params(params)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "train_accuracy": accuracy_score(y_train, model.predict(X_train)),
        }
        mlflow.log_metrics(metrics)


        model_info = mlflow.sklearn.log_model(
            model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            input_example=X_train.head(3),
            skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
        )

        with tempfile.TemporaryDirectory() as tmp:
            joblib_path = Path(tmp) / "model.joblib"
            joblib.dump(model, joblib_path)
            mlflow.log_artifact(str(joblib_path))

        print(f"\n=== {run_name} ===")
        print(f"Registered '{REGISTERED_MODEL_NAME}' version {model_info.registered_model_version}")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))

        if params["max_depth"] == API_MAX_DEPTH:
            chosen = {"model": model, "run_name": run_name, "f1": metrics["f1_score"],
                      "version": model_info.registered_model_version}


joblib.dump(chosen["model"], "model.joblib")
print(f"\nModel for the API: {chosen['run_name']} (F1 = {chosen['f1']:.4f}, "
      f"registry version {chosen['version']}) -> saved to model.joblib")
