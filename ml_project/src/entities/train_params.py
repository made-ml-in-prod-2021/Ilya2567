# from dataclasses import dataclass, field
from dataclasses import dataclass

from omegaconf import MISSING

from .models import ModelConfig
from .splits import SplitConfig


@dataclass()
class TrainingParams:
    models: ModelConfig = MISSING  # field(default="RandomForestClassifier")
    splits: SplitConfig = MISSING
    # random_state: int = MISSING
