from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ObjectConfig:
    _target_: str = MISSING


@dataclass
class RandomForestClassifierConfig(ObjectConfig):
    max_depth: int = MISSING


@dataclass
class LogisticRegressionConfig(ObjectConfig):
    C: float = MISSING
    random_state: int = MISSING
