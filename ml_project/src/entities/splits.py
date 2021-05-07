from dataclasses import dataclass

# from omegaconf import MISSING


@dataclass
class SplitConfig:
    # max_depth: int = MISSING
    test_size: float = 0.2
    random_state: int = 42
