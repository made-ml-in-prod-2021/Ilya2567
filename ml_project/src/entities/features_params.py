# from dataclasses import dataclass, field
from dataclasses import dataclass
from typing import List

from omegaconf import MISSING
import pandas as pd

from .models import ObjectConfig
from .splits import SplitConfig


@dataclass()
class FeaturesParams:
    data_filename: str = MISSING
    splits: SplitConfig = MISSING
    categorical_encoders: ObjectConfig = MISSING
    numerical_encoders: ObjectConfig = MISSING
    target_column: str = MISSING
    categorical_columns: List[str] = MISSING
    numerical_columns: List[str] = MISSING
    transformer_filename: str = MISSING


@dataclass()
class ProcessedData:
    x_train: pd.DataFrame = None
    x_test: pd.DataFrame = None
    target_train: pd.Series = None
    target_test: pd.Series = None
