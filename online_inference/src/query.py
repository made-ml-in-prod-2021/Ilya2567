import logging

# import numpy as np
import pandas as pd
import click
import requests

logging.basicConfig(filename=f'query.log', level=logging.DEBUG)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--query", default="query.csv")
@click.option("--host", default="localhost")
@click.option("--port", default=8000)
def predict(query, host, port):
    logger.info("Query start...")

    data = pd.read_csv(query)

    logger.info(f'Request {len(data)} lines')
    response = requests.post(
        f"http://{host}:{port}/predict",
        json={"data": data.to_numpy().tolist(), "features": data.columns.tolist()},
    )
    logger.info(f'Response code: {response.status_code}')
    if response.ok:
        print(pd.DataFrame(response.json()))

    logger.info("Query end")


if __name__ == "__main__":
    predict()
