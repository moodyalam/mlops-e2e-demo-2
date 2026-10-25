"""A FastAPI web service that serves predictions from the Iris model."""
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from model import train_model

# The Iris model uses exactly 4 measurements per flower.
N_FEATURES = 4


class InputData(BaseModel):
    # Exactly 4 numbers. Anything else is rejected with HTTP 422 before our code runs.
    x: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)


app = FastAPI(title="Iris prediction service")

# The model is trained once, when the app starts. This is fine for a demonstration;
# a production system would load a model that was trained and saved earlier.
_, _, model = train_model()


@app.get("/health")
def health():
    """Tells callers (and the tests) that the service is up and the model is loaded."""
    return {"status": "ok"}


@app.post("/predict")
def predict(data: InputData):
    x = np.array(data.x).reshape(1, -1)
    prediction = int(model.predict(x)[0])
    print(f"Received request: {data.x} -> Prediction: {prediction}")
    return {"prediction": prediction}
