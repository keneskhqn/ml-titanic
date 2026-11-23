from pathlib import Path
from typing import Literal, Optional
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).parent / "model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Titanic Survival API",
    description="Predicts whether a Titanic passenger survived, using a Random Forest model.",
)


class Passenger(BaseModel):
    """Input features for one passenger. FastAPI validates every request against this."""

    pclass: Literal[1, 2, 3] = Field(description="Ticket class: 1, 2 or 3")
    sex: Literal["male", "female"]
    age: Optional[float] = Field(default=None, ge=0, le=100, description="Age in years (optional)")
    sibsp: int = Field(ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(ge=0, description="Number of parents/children aboard")
    fare: float = Field(ge=0, description="Ticket price")
    embarked: Optional[Literal["C", "Q", "S"]] = Field(
        default=None, description="Port: C = Cherbourg, Q = Queenstown, S = Southampton (optional)"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"pclass": 1, "sex": "female", "age": 29, "sibsp": 0,
                 "parch": 0, "fare": 100.0, "embarked": "S"}
            ]
        }
    }


@app.get("/")
def root():
    return {"message": "ML API is running"}


@app.post("/predict")
def predict(passenger: Passenger):
    row = pd.DataFrame([{
        "Sex": passenger.sex,
        "Embarked": passenger.embarked if passenger.embarked is not None else np.nan,
        "Pclass": passenger.pclass,
        "Age": passenger.age if passenger.age is not None else np.nan,
        "SibSp": passenger.sibsp,
        "Parch": passenger.parch,
        "Fare": passenger.fare,
    }])

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0][1]) 

    return {
        "survived": prediction,
        "prediction": "Survived" if prediction == 1 else "Died",
        "survival_probability": round(probability, 3),
    }
