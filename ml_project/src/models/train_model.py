import os
import pickle

import hydra
import pandas as pd
from omegaconf import OmegaConf
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..entities import TrainingParams


def pickle_load(path):
    assert os.path.exists(path), f'{path} not exists!'
    with open(path, 'rb') as fin:
        return pickle.load(fin)


def pickle_dump(obj, path):
    # assert not os.path.exists(path), f'{path} already exists!'
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)
    print(f'Object saved.')


def train():
    path = os.path.join('data', 'raw', 'heart.csv')
    heart = pd.read_csv(path)
    assert heart.isna().sum().sum() == 0

    target = heart.pop('target')

    bin_cols = ['sex', 'fbs', 'exang', ]
    cat_cols = ['cp', 'restecg', 'slope', 'thal']
    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']
    assert set(bin_cols) | set(cat_cols) | set(num_cols) == set(heart.columns)

    heart_train, heart_test, target_train, target_test = train_test_split(
        heart, target, test_size=0.2, random_state=42
    )
    cat_pipe = Pipeline([
        ('ohe', OneHotEncoder(drop='if_binary'))
    ])
    num_pipe = Pipeline([
        ('scaler', StandardScaler())
    ])
    col_transformer = ColumnTransformer([
        ('categorical_pipeline', cat_pipe, bin_cols + cat_cols),
        ('num_pipe', num_pipe, num_cols),
    ])
    x_train = col_transformer.fit_transform(heart_train)
    x_test = col_transformer.transform(heart_test)

    model = RandomForestClassifier(max_depth=5)
    model.fit(x_train, target_train)
    # model.score(x_train, target_train)

    path = os.path.join('models', 'model.pkl')
    pickle_dump(model, path)

    y_pred = model.predict(x_test)
    f1_metric = f1_score(y_pred, target_test)

    report_dict = {
        'model_type': 'RandomForestClassifier',
        'f1_metric': str(f1_metric),
    }
    report = OmegaConf.create(report_dict)
    yaml_report = OmegaConf.to_yaml(report)

    path = os.path.join('reports', 'model.yml')
    with open(path, "w") as fin:
        fin.writelines(yaml_report)


@hydra.main(
    config_path=os.path.join("..", "..",),
    # config_name="training_params.yml"
            )
def train_pipeline_command(cfg: TrainingParams = None):
    # params = read_training_pipeline_params(config_path)
    # train_pipeline(params)
    print(cfg)


# python -m src.models.train_model --config-name ./configs/training_params.yml
if __name__ == "__main__":
    train_pipeline_command()
