FROM python:3.8-slim

COPY model/ ./model
COPY src/ ./src
COPY requirements_docker.txt ./

RUN pip install -r requirements_docker.txt

EXPOSE 8000

ENV MODEL_PATH="/model/model.pkl"

WORKDIR .
CMD ["uvicorn", "src.app:app"]
