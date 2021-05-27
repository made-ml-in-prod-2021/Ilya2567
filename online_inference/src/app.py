import logging
import os
from typing import Any, List

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from starlette.responses import PlainTextResponse

from .utils import pickle_load
from .entities import DiagnosisResponse, DiagnosisRequest

DEFAULT_VALIDATION_ERROR_CODE = 400

logging.basicConfig(filename="app.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)
model: Any = None
transformer: Any = None
app = FastAPI(title="Heart Disease Prediction")


@app.get("/")
async def root():
    return {"message": app.title}


@app.on_event("startup")
def load_model():
    logger.info(f"Loading model...")
    global model, transformer
    path = os.getenv("MODEL_PATH")
    if path is None:
        path = os.path.join('model', 'model.pkl')
    model = pickle_load(path)

    path = os.getenv("TRANSFORMER_PATH")
    if path is None:
        path = os.path.join('model', 'transformer.pkl')
    transformer = pickle_load(path)

    logger.info(f"Model loaded.")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return PlainTextResponse(str(exc), status_code=DEFAULT_VALIDATION_ERROR_CODE)


@app.post("/predict", response_model=List[DiagnosisResponse])
def predict(request: DiagnosisRequest):
    return get_predict(request.data, request.features)


def get_predict(data: List, features: List[str]) -> List[DiagnosisResponse]:
    global model, transformer
    data = pd.DataFrame(data, columns=features)
    transformed = transformer.transform(data)
    predicts = model.predict(transformed)

    result = [
        DiagnosisResponse(id=tp[0], diagnosis=tp[1])
        for tp in enumerate(predicts)
    ]
    return result


if __name__ == "__main__":
    uvicorn.run("src.app:app", host="localhost", port=os.getenv("PORT", 8000))
