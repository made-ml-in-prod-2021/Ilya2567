import os
import pickle
import logging
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def pickle_load(path: str):
    assert os.path.exists(path), f'{path} not exists!'
    try:
        with open(path, 'rb') as fin:
            result = pickle.load(fin)
    except AssertionError as err:
        logger.error('%s, %s', err, path)
        raise RuntimeError(err)
    logger.debug('Object loaded.')
    return result


def pickle_dump(obj: Any, path: str):
    # assert not os.path.exists(path), f'{path} already exists!'
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)
    logger.debug('Object saved.')
