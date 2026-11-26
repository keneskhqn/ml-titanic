# Titanic Survival Prediction System

A machine learning system that predicts whether a Titanic passenger survived. The model is served as a REST API with **FastAPI**, packaged with **Docker**, used through a **Streamlit** web interface, and tracked and versioned with **MLflow**.

## Architecture

```
Training:   train.py ──► MLflow (experiments, metrics, Model Registry "titanic-model")
                │
                └──► model.joblib
                          │
Serving:    User ──► Streamlit (frontend.py, :8501) ──HTTP──► FastAPI in Docker (main.py, :8000) ──► model.joblib
```

| Component | File | Role |
|---|---|---|
| Training + MLflow | `train.py` | trains the models, logs experiments, registers versions, saves the chosen model |
| Model | `model.joblib` | preprocessing + Random Forest in one pipeline |
| API | `main.py` | FastAPI service with `GET /` and `POST /predict` |
| Container | `Dockerfile` | runs the API in Docker |
| Frontend | `frontend.py` | Streamlit page where users enter passenger data and see the prediction |

The frontend does not load the model itself: it sends requests to the API, so the interface and the model service are independent parts.

## Project structure

```
ml-titanic/
├── data/
│   └── Titanic-Dataset.csv   # Kaggle Titanic training data (891 passengers, 12 columns)
├── train.py                  # trains 3 models, logs them to MLflow, saves the chosen one to model.joblib
├── model.joblib              # model used by the API (preprocessing + classifier)
├── main.py                   # FastAPI application
├── frontend.py               # Streamlit web interface
├── requirements.txt          # Python dependencies
├── Dockerfile                # container instructions
├── .dockerignore
└── README.md
```

`mlflow.db` (the local MLflow database) is created by `train.py` and is not stored in Git.

## Dataset

The Kaggle Titanic dataset (`train.csv`), 891 passengers.

| Role | Columns |
|---|---|
| Target | `Survived` (0 = died, 1 = survived) |
| Features | `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked` |
| Dropped | `PassengerId`, `Name`, `Ticket`, `Cabin` (IDs, free text, or mostly missing) |

Missing values: `Age` (177), `Embarked` (2).

## Model

The model is a scikit-learn `Pipeline`, so all preprocessing is saved together with the classifier in `model.joblib`:

1. **Train/test split:** 80% / 20%, stratified by `Survived`, `random_state=42`.
2. **Categorical features** (`Sex`, `Embarked`): fill missing values with the most frequent value, then one-hot encode.
3. **Numeric features** (`Pclass`, `Age`, `SibSp`, `Parch`, `Fare`): fill missing values with the median.
4. **Classifier:** `RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)`.

The imputers are fitted on the training data only, which avoids data leakage. Because preprocessing is part of the pipeline, the API can accept raw values such as `"sex": "female"` or a missing `age`.

`max_depth=5` was chosen after comparing three experiments in MLflow (see [Experiment tracking](#experiment-tracking-mlflow)).

### Results of the API model (test set, 179 passengers)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Died | 0.79 | 0.94 | 0.86 |
| Survived | 0.86 | 0.61 | 0.71 |

**Accuracy: 0.81**

The model recognises non-survivors very well but misses some survivors, mainly because the data is imbalanced (more passengers died than survived).

## Experiment tracking (MLflow)

`train.py` uses MLflow to track experiments and register models. Data is stored locally in `mlflow.db` (SQLite).

### Experiment

All runs are logged to the experiment **`titanic-survival`**. Three Random Forest models with different `max_depth` are trained; each one is a separate run.

For every run MLflow logs:

| What | Values |
|---|---|
| Parameters | `model_type`, `n_estimators`, `max_depth`, `test_size`, `random_state` |
| Metrics | `accuracy`, `f1_score`, `precision`, `recall` (test set), `train_accuracy` |
| Artifacts | the model in MLflow format (`model`) and the saved model file `model.joblib` |

### Results

| Run | max_depth | Accuracy | F1-score | Precision | Recall | Train accuracy | Registry version |
|---|---|---|---|---|---|---|---|
| rf_depth_3 | 3 | 0.804 | 0.720 | 0.804 | 0.652 | 0.819 | 1 |
| **rf_depth_5** | **5** | **0.810** | **0.712** | **0.857** | **0.609** | **0.855** | **2** |
| rf_depth_10 | 10 | 0.810 | 0.726 | 0.818 | 0.652 | 0.949 | 3 |

**Chosen model: `rf_depth_5` (version 2).** Depth 10 has a slightly higher F1-score, but its train accuracy (0.95) is much higher than its test accuracy (0.81), which is a sign of overfitting. Depth 5 has the same test accuracy with a much smaller gap, so it is used by the API (`model.joblib`).

### Model Registry

Each run registers its model in the MLflow Model Registry under the name **`titanic-model`**, so the registry contains **versions 1, 2 and 3**. A version can be loaded by name and number, for example:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/titanic-model/2")
```

### How to run

Train the models and log them to MLflow:

```bash
python train.py
```

Open the MLflow UI:

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Then open http://127.0.0.1:5000.

Running `train.py` again adds three new runs and three new registry versions. To start from a clean state, delete `mlflow.db` first.

## Run locally

Install the dependencies:

```bash
pip install -r requirements.txt
```

(Optional) retrain the models and log them to MLflow:

```bash
python train.py
```

Start the API:

```bash
python -m uvicorn main:app --reload
```

Then open:

- http://127.0.0.1:8000/ for the health check
- http://127.0.0.1:8000/docs for the Swagger UI

## Run with Docker

Build the image:

```bash
docker build -t titanic-api .
```

Run the container:

```bash
docker run -p 8000:8000 titanic-api
```

The API is then available at http://localhost:8000 and http://localhost:8000/docs.

## Frontend (Streamlit)

`frontend.py` is a simple web interface for the model. The user fills in a form, and the page sends the data to the FastAPI `POST /predict` endpoint and shows the result.

**Features:**

- form with all passenger features: ticket class, sex, age, siblings/spouses, parents/children, fare, port of embarkation
- "Age unknown" and "Unknown" port options (sent as missing values, filled by the model pipeline)
- result as **Survived** / **Died** with the probability of survival
- the exact request and response can be shown under "Request and response"
- API status in the sidebar, and a clear error message if the API is not running

### How to run

The frontend needs the API to be running, so use two terminals.

**Terminal 1: start the API** (locally or in Docker):

```bash
python -m uvicorn main:app --reload
```

or

```bash
docker run -p 8000:8000 titanic-api
```

**Terminal 2: start the frontend:**

```bash
python -m streamlit run frontend.py
```

Then open http://localhost:8501.

By default the frontend calls the API at `http://127.0.0.1:8000`. A different address can be set with the `API_URL` environment variable, for example in PowerShell:

```powershell
$env:API_URL = "http://localhost:9000"
python -m streamlit run frontend.py
```

### Example

| Input | Result | Probability of survival |
|---|---|---|
| 3rd class, male, 25, fare 7.25, Southampton | Died | 8.6% |
| 1st class, female, 29, fare 100, Southampton | Survived | 96.6% |

## API endpoints

### `GET /`

Health check.

```json
{"message": "ML API is running"}
```

### `POST /predict`

Predicts survival for one passenger.

**Request body:**

| Field | Type | Required | Allowed values |
|---|---|---|---|
| `pclass` | int | yes | 1, 2, 3 |
| `sex` | string | yes | `"male"`, `"female"` |
| `age` | float | no | 0–100 |
| `sibsp` | int | yes | ≥ 0 |
| `parch` | int | yes | ≥ 0 |
| `fare` | float | yes | ≥ 0 |
| `embarked` | string | no | `"C"`, `"Q"`, `"S"` |

Missing `age` or `embarked` values are filled automatically by the model pipeline. Invalid values (for example `"pclass": 5`) return a **422** validation error.

**Example 1:** 1st-class woman

```json
{"pclass": 1, "sex": "female", "age": 29, "sibsp": 0, "parch": 0, "fare": 100.0, "embarked": "S"}
```

```json
{"survived": 1, "prediction": "Survived", "survival_probability": 0.966}
```

**Example 2:** 3rd-class man

```json
{"pclass": 3, "sex": "male", "age": 25, "sibsp": 0, "parch": 0, "fare": 7.25, "embarked": "S"}
```

```json
{"survived": 0, "prediction": "Died", "survival_probability": 0.086}
```

**Example with curl:**

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"pclass": 3, "sex": "male", "age": 25, "sibsp": 0, "parch": 0, "fare": 7.25, "embarked": "S"}'
```

## Tech stack

Python 3.14 · pandas · scikit-learn · joblib · FastAPI · Uvicorn · Docker · MLflow · Streamlit · requests
