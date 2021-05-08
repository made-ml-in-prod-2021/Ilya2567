from dataclasses import dataclass

from omegaconf import MISSING


@dataclass
class ColumnsPipeline:
    _target_: str = MISSING
