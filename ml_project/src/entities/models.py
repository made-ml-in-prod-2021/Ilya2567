from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ModelConfig:
    _target_: str = MISSING


@dataclass
class RandomForestClassifierConfig(ModelConfig):
    max_depth: int = MISSING


@dataclass
class LogisticRegressionConfig(ModelConfig):
    C: float = MISSING
    random_state: int = MISSING
