# from dataclasses import dataclass, field
from dataclasses import dataclass

from omegaconf import MISSING

from .models import ModelConfig


@dataclass()
class TrainingParams:
    models: ModelConfig = MISSING  # field(default="RandomForestClassifier")
    random_state: int = MISSING
