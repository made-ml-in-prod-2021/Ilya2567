from dataclasses import dataclass

from omegaconf import MISSING

from .models import ObjectConfig


@dataclass()
class TrainingParams:
    models: ObjectConfig = MISSING
    model_filename: str = MISSING
