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

from ..entities import TrainingParams
from .utils import pickle_dump

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# def pickle_load(path):
#     assert os.path.exists(path), f'{path} not exists!'
#     with open(path, 'rb') as fin:
#         return pickle.load(fin)
#
#
# def pickle_dump(obj, path):
#     # assert not os.path.exists(path), f'{path} already exists!'
#     with open(path, 'wb') as fout:
#         pickle.dump(obj, fout)
#     logger.debug(f'Object saved.')


def train(cfg: TrainingParams):
    logger.info("Started train pipeline")
    logger.debug('cwd: %s', hydra.utils.get_original_cwd())

    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'raw', 'heart.csv')
    heart = pd.read_csv(path)
    assert heart.isna().sum().sum() == 0

    target = heart.pop(cfg.target_column)

    heart_train, heart_test, target_train, target_test = train_test_split(
        heart, target, **cfg.splits
    )

    cat_enc = hydra.utils.instantiate(cfg.categorical_encoders)
    num_enc = hydra.utils.instantiate(cfg.numerical_encoders)
    cat_pipe = Pipeline([
        ('cat_scaler', cat_enc)
    ])
    num_pipe = Pipeline([
        ('num_scaler', num_enc)
    ])
    col_transformer = ColumnTransformer([
        ('cat_pipe', cat_pipe, list(cfg.categorical_columns)),
        ('num_pipe', num_pipe, list(cfg.numerical_columns)),
    ])
    x_train = col_transformer.fit_transform(heart_train)
    x_test = col_transformer.transform(heart_test)

    model = hydra.utils.instantiate(cfg.models)
    model.fit(x_train, target_train)
    pickle_dump(model, 'model.pkl')

    y_pred = model.predict(x_test)
    metrics_dict = {
        'score': float(model.score(x_train, target_train)),
        'f1_metric': float(f1_score(y_pred, target_test)),
        'conf_matrix': confusion_matrix(y_pred, target_test).tolist(),
    }
    logger.debug('score: %.4f', metrics_dict['score'])
    logger.info('f1_metric: %.4f', metrics_dict['f1_metric'])
    conf_matrix_str = '\n'.join(map(str, metrics_dict['conf_matrix']))
    logger.info('confusion_matrix: \n%s', conf_matrix_str)

    # path = os.path.join('reports', 'model.yml')
    with open('metrics.yaml', "w") as fin:
        yaml_report = yaml.dump(metrics_dict)
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
