import logging
import logging.handlers
import os


def get_logger(name: str):
    dirname = os.path.dirname(name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # By default, logs all messages
    fh = logging.FileHandler("{0}.log".format(name))
    fh.setLevel(logging.DEBUG)
    fh_format = logging.Formatter('%(asctime)s  - %(levelname)-8s - %(message)s')
    fh.setFormatter(fh_format)
    logger.addHandler(fh)
    return logger
