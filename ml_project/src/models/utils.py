import os
import pickle
import logging
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pickle_load(path: str):
    assert os.path.exists(path), f'{path} not exists!'
    with open(path, 'rb') as f_in:
        return pickle.load(f_in)
    logger.debug('Object loaded.')


def pickle_dump(obj: Any, path: str):
    # assert not os.path.exists(path), f'{path} already exists!'
    with open(path, 'wb') as f_out:
        pickle.dump(obj, f_out)
    logger.debug('Object saved.')
