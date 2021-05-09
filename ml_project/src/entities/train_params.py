# from dataclasses import dataclass, field
from dataclasses import dataclass
from typing import List

from omegaconf import MISSING

from .models import ObjectConfig
from .splits import SplitConfig


@dataclass()
class TrainingParams:
    models: ObjectConfig = MISSING
    model_filename: str = MISSING
