import logging
import os

from fastapi import FastAPI
import uvicorn

logger = logging.getLogger(__name__)
app = FastAPI(title="Heart Disease Prediction")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id):
    return {"item_id": item_id}


if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=os.getenv("PORT", 8000))
