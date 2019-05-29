import logging
import logging.handlers
import coloredlogs
import os

logger = None
logging_to_file = False


def save_logs():
    global logging_to_file

    if not logging_to_file:
        fmt = logging.Formatter('[%(asctime)s] %(message)s', "%m-%d %H:%M:%S")
        os.makedirs('logs', exist_ok=True)
        fh = logging.handlers.TimedRotatingFileHandler('logs/log.log', when='D')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        logging_to_file = True


def init_logger():
    global logger

    logger = logging.getLogger('npy')
    logger.propagate = False

    logger.setLevel(logging.DEBUG)
    level_styles = {
                    'critical': {'color': 'red', 'bold': True},
                    'error': {'color': 'yellow'},
                    'warning': {'color': 'green', 'bold': True},
                    'info': {},
                    'debug': {'color': 'black', 'bright': True},
                    }
    field_styles = {
        'asctime': {'color': 'blue'},
        'levelname': {'color': 'yellow', 'faint': True}
    }
    coloredlogs.install(level='DEBUG', logger=logger,
                        fmt='[%(asctime)s] %(message)s',
                        datefmt="%m-%d %H:%M:%S",
                        field_styles=field_styles,
                        level_styles=level_styles)


if logger is None:
    init_logger()
