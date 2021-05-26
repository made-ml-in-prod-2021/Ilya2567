from dataclasses import dataclass

from omegaconf import MISSING


@dataclass()
class PredictParams:
    data_filename: str = MISSING
    transformer_filename: str = MISSING
    model_filename: str = MISSING
