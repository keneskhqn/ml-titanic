import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

df = pd.read_csv('data/Titanic-Dataset.csv')
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
print(df.head())

categorical_features = ["Sex", "Embarked"]
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

X = df[categorical_features + numeric_features]
y = df["Survived"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)),
])
model.fit(X_train, y_train)


print(preprocessor.get_feature_names_out())
print("Age median learned from train:",
      preprocessor.named_transformers_["num"].named_steps["imputer"].statistics_[1])
print("Embarked value used for missing:",
      preprocessor.named_transformers_["cat"].named_steps["imputer"].statistics_[1])
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Died", "Survived"]))


joblib.dump(model, "model.joblib")
print("Model saved to model.joblib")