import os
# import pickle
import logging

import hydra
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import yaml

from ..entities import TrainingParams, ProcessedData
from .utils import pickle_dump

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def train(cfg: TrainingParams):
    logger.info("Started train pipeline")
    logger.debug('cwd: %s', hydra.utils.get_original_cwd())

    # Входные пути пути заданы жестко, поскольку не предполагают изменения и служат
    # передаточным звеном между модулями проекта
    logger.info("Loading processed data")
    data = ProcessedData()
    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'processed', 'x_train.csv')
    data.x_train = pd.read_csv(path)
    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'processed', 'x_test.csv')
    data.x_test = pd.read_csv(path)
    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'processed', 'target_train.csv')
    data.target_train = pd.read_csv(path).iloc[:, 0]
    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'processed', 'target_test.csv')
    data.target_test = pd.read_csv(path).iloc[:, 0]

    metrics = {name: df.shape for name, df in data.__dict__.items()}
    logger.debug('shapes: %s', metrics)

    model = hydra.utils.instantiate(cfg.models)
    model.fit(data.x_train, data.target_train)
    pickle_dump(model, 'model.pkl')

    y_pred = model.predict(data.x_test)
    metrics = {
        'score': float(model.score(data.x_train, data.target_train)),
        'f1_metric': float(f1_score(y_pred, data.target_test)),
        'conf_matrix': confusion_matrix(y_pred, data.target_test).tolist(),
    }
    logger.debug('score: %.4f', metrics['score'])
    logger.info('f1_metric: %.4f', metrics['f1_metric'])
    conf_matrix_str = '\n'.join(map(str, metrics['conf_matrix']))
    logger.info('confusion_matrix: \n%s', conf_matrix_str)

    # path = os.path.join('reports', 'model.yml')
    with open('metrics.yaml', "w") as fin:
        yaml_report = yaml.dump(metrics)
        fin.writelines(yaml_report)

    logger.info("Finished train pipeline")


@hydra.main(
    config_path=os.path.join('..', '..', 'configs'),
    config_name="training_params.yaml"
)
def train_pipeline_command(cfg: TrainingParams = None):
    logger.debug("cfg: %s", cfg)
    train(cfg)


# python -m src.models.train_model --config-name training_params.yaml
if __name__ == "__main__":
    train_pipeline_command()
