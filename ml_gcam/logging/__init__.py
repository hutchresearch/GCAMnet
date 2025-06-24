from pathlib import Path

from rich.logging import RichHandler

from .. import config


def init_logging():
    import logging

    fmt = "[%(levelname)-8s %(asctime)s %(filename)15s:%(lineno)-4d] %(message)s"
    datefmt = "%d-%b-%y %H:%M:%S"
    level = logging.DEBUG if bool(int(config.debug)) else logging.WARNING

    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    c_handler = RichHandler()
    c_handler.setLevel(level)
    logger.addHandler(c_handler)

    f_handler = logging.FileHandler(Path(config.paths.repo) / "error.log")
    f_handler.setLevel(logging.ERROR)
    f_format = logging.Formatter(fmt=fmt, datefmt=datefmt)
    f_handler.setFormatter(f_format)
    logger.addHandler(f_handler)

    return logger


logger = init_logging()
