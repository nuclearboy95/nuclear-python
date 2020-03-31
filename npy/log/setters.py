import queue
import logging
import logging.handlers
import coloredlogs
import os

_logger = None
_sh = None
LOGGING_TO_FILE = False
LOGGING_TO_TELEGRAM = False


__all__ = ['verbosity', 'save', 'telegram']


def _init():
    global _logger
    global _sh

    _logger = logging.getLogger('npy')
    _logger.propagate = False

    _logger.setLevel(logging.DEBUG)
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
    coloredlogs.install(level='DEBUG', logger=_logger,
                        fmt='[%(asctime)s] %(message)s',
                        datefmt="%d %b %H:%M:%S",
                        field_styles=field_styles,
                        level_styles=level_styles)
    _sh = _logger.handlers[0]


def save():
    global LOGGING_TO_FILE

    if not LOGGING_TO_FILE:
        fmt = logging.Formatter('P%(process)05d L%(levelno).1s [%(asctime)s] %(message)s', "%m-%d %H:%M:%S")
        os.makedirs('log', exist_ok=True)
        fh = logging.handlers.TimedRotatingFileHandler('log/log.log', when='D')
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        _logger.addHandler(fh)
        LOGGING_TO_FILE = True


def int2level(level):
    if level <= 1:
        return logging.DEBUG

    elif level <= 2:
        return logging.INFO

    elif level <= 3:
        return logging.WARNING

    elif level <= 4:
        return logging.ERROR

    else:
        return logging.CRITICAL


def verbosity(level=2):
    """

    :param int level:
    :return:
    """
    if not isinstance(level, int):
        return

    if _logger is None:
        _init()

    level = int2level(level)
    _sh.setLevel(level)


def telegram(level=3):
    from .telegram_handler import TelegramHandler
    global LOGGING_TO_TELEGRAM
    if not LOGGING_TO_TELEGRAM:

        fmt = logging.Formatter('%(message)s')
        th = TelegramHandler()
        th.setFormatter(fmt)
        level = int2level(level)
        th.setLevel(level)

        que = queue.Queue(-1)
        qh = logging.handlers.QueueHandler(que)
        qh.setLevel(level)
        listener = logging.handlers.QueueListener(que, th)

        _logger.addHandler(qh)

        listener.start()

        LOGGING_TO_TELEGRAM = True


if _logger is None:
    _init()
