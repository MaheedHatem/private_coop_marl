import logging.config


class LevelOnlyFilter:
    def __init__(self, level):
        self.level = level

    def filter(self, record):
        return record.levelno == self.level

def init_logging(log_file):
    LOGGING_CONFIG = {
        "version": 1,
        "loggers": {
            "": {  # root logger
                "level": "INFO",
                "propagate": False,
                "handlers": ["stream_handler", "file_handler"],
            },
        },
        "handlers": {
            "stream_handler": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "level": "INFO",
                "filters": ["only_info"],
                "formatter": "default_formatter",
            },
            "file_handler": {
                "class": "logging.FileHandler",
                "filename": log_file,
                "mode": 'w',
                "level": "INFO",
                "formatter": "default_formatter",
            },
        },
        "filters": {
            "only_info": {
                "()": LevelOnlyFilter,
                "level": logging.INFO,
            },
        },
        "formatters": {
            "default_formatter": {
                "format": "%(asctime)s-%(levelname)s:: %(message)s",
            },
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)