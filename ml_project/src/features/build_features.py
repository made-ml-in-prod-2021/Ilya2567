import os
# import pickle
import logging

import hydra
import pandas as pd
from sklearn.compose import ColumnTransformer
# from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
# import yaml

# from ..entities import TrainingParams
from ..entities import FeaturesParams, ProcessedData
from ..models import pickle_dump
from .binary_encoder import BinaryEncoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def build_features(cfg: FeaturesParams):
    logger.info("Started feature engineering")
    logger.debug('cwd: %s', hydra.utils.get_original_cwd())

    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'raw', cfg.data_filename)
    heart = pd.read_csv(path)
    assert heart.isna().sum().sum() == 0

    target = heart.pop(cfg.target_column)

    heart_train, heart_test, target_train, target_test = train_test_split(
        heart, target, **cfg.splits
    )

    cat_enc = hydra.utils.instantiate(cfg.categorical_encoders)
    num_enc = hydra.utils.instantiate(cfg.numerical_encoders)
    bin_pipe = Pipeline([
        ('bin', BinaryEncoder())
    ])
    cat_pipe = Pipeline([
        ('cat_scaler', cat_enc)
    ])
    num_pipe = Pipeline([
        ('num_scaler', num_enc)
    ])

    col_transformer = ColumnTransformer([
        ('bin_pipe', bin_pipe, list(cfg.binary_columns)),
        ('cat_pipe', cat_pipe, list(cfg.categorical_columns)),
        ('num_pipe', num_pipe, list(cfg.numerical_columns)),
    ])
    data = ProcessedData()
    data.x_train = col_transformer.fit_transform(heart_train)
    data.x_test = col_transformer.transform(heart_test)
    data.target_train = target_train
    data.target_test = target_test

    # Выходные пути заданы жестко, поскольку не предполагают изменения и служат
    # передаточным звеном между модулями проекта
    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'processed', 'x_train.csv')
    pd.DataFrame(data.x_train).to_csv(path, index=False)
    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'processed', 'x_test.csv')
    pd.DataFrame(data.x_test).to_csv(path, index=False)
    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'processed', 'target_train.csv')
    data.target_train.to_csv(path, index=False)
    path = os.path.join(hydra.utils.get_original_cwd(), 'data', 'processed', 'target_test.csv')
    data.target_test.to_csv(path, index=False)

    metrics = {name: df.shape for name, df in data.__dict__.items()}
    logger.debug('shapes: %s', metrics)

    path = os.path.join(hydra.utils.get_original_cwd(), 'models', cfg.transformer_filename)
    pickle_dump(col_transformer, path)

    logger.info("Finished feature engineering")


@hydra.main(
    config_path=os.path.join('..', '..', 'configs'),
    config_name=os.path.join('features', 'features_param.yaml')
)
def train_pipeline_command(cfg: FeaturesParams = None):
    logger.debug("cfg: %s", cfg)
    build_features(cfg)


# python -m src.models.train_model --config-name training_params.yaml
if __name__ == "__main__":
    train_pipeline_command()
