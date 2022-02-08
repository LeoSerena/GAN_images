import os
import logging

logger = logging.getLogger(__name__)

def make_dir_if_not_exists(path : str) -> bool:
    """
    verifies whether the given path is a dir and if not creates it
    """
    if not os.path.isdir(path):
        logger.info('creating directory: ./{}'.format(path))
        os.makedirs(path)
        return True
    else:
        return False