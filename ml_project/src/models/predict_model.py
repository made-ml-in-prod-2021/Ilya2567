import os
import logging

import hydra
import pandas as pd
from sklearn.compose import ColumnTransformer

from ..entities import PredictParams
from .utils import pickle_load

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def predict(cfg: PredictParams):
    logger.info("Prediction started")
    logger.debug('cwd: %s', hydra.utils.get_original_cwd())

    # Входные пути пути заданы жестко, поскольку не предполагают изменения и служат
    # передаточным звеном между модулями проекта
    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'raw', cfg.data_filename)
    data_raw = pd.read_csv(path)
    logger.debug('raw_data_shape: %s', data_raw.shape)

    path = os.path.join(hydra.utils.get_original_cwd(), 'models', cfg.transformer_filename)
    transformer: ColumnTransformer = pickle_load(path)

    cols = set.union(*[set(tp[2]) for tp in transformer.transformers_])
    cols = [col for col in data_raw.columns if col in cols]
    data = transformer.transform(data_raw[cols])
    logger.debug('processed_data_shape: %s', data.shape)

    path = os.path.join(hydra.utils.get_original_cwd(), 'models', cfg.model_filename)
    model = pickle_load(path)
    logger.debug('model: %s', model)

    y_pred = model.predict(data)
    logger.debug('predict_shape: %s', y_pred.shape)
    pd.Series(y_pred, name='predict').to_csv('predict.csv', index=False)

    logger.info("Prediction finished")


@hydra.main(
    config_path=os.path.join('..', '..', 'configs'),
    config_name="predict_params.yaml"
)
def train_pipeline_command(cfg: PredictParams = None):
    logger.debug("cfg: %s", cfg)
    predict(cfg)


# python -m src.models.train_model --config-name training_params.yaml
if __name__ == "__main__":
    train_pipeline_command()
