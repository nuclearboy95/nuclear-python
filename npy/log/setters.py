import logging
import logging.handlers
import coloredlogs
import os

_logger = None
_sh = None
logging_to_file = False


__all__ = ['verbosity', 'save']


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
                        datefmt="%m-%d %H:%M:%S",
                        field_styles=field_styles,
                        level_styles=level_styles)
    _sh = _logger.handlers[0]


def save():
    global logging_to_file

    if not logging_to_file:
        fmt = logging.Formatter('P%(process)05d L%(levelno).1s [%(asctime)s] %(message)s', "%m-%d %H:%M:%S")
        os.makedirs('log', exist_ok=True)
        fh = logging.handlers.TimedRotatingFileHandler('log/log.log', when='D')
        fh.setFormatter(fmt)
        fh.setLevel(logging.DEBUG)
        _logger.addHandler(fh)
        logging_to_file = True


def verbosity(level=2):
    """

    :param int level:
    :return:
    """
    if not isinstance(level, int):
        return

    if _logger is None:
        _init()

    if level <= 1:
        _sh.setLevel(logging.DEBUG)

    elif level <= 2:
        _sh.setLevel(logging.INFO)

    elif level <= 3:
        _sh.setLevel(logging.WARNING)

    elif level <= 4:
        _sh.setLevel(logging.ERROR)

    else:
        _sh.setLevel(logging.CRITICAL)


if _logger is None:
    _init()
