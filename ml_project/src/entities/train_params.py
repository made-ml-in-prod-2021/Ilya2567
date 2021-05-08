# from dataclasses import dataclass, field
from dataclasses import dataclass
from typing import List

from omegaconf import MISSING

from .models import ObjectConfig
from .splits import SplitConfig


@dataclass()
class TrainingParams:
    splits: SplitConfig = MISSING
    categorical_encoders: ObjectConfig = MISSING
    numerical_encoders: ObjectConfig = MISSING
    models: ObjectConfig = MISSING
    target_column: str = MISSING
    categorical_columns: List[str] = MISSING
    numerical_columns: List[str] = MISSING
    # random_state: int = MISSING
