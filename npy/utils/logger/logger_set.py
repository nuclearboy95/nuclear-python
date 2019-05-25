import logging
import coloredlogs

logger = None


def init_logger():
    global logger

    logger = logging.getLogger('npy')
    logger.propagate = False
    # fmt = logging.Formatter('%(levelname).1s [%(asctime)s] %(message)s', "%m-%d %H:%M:%S")
    # handler = logging.StreamHandler()
    # handler.setFormatter(fmt)
    # logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    level_styles = {'critical': {'color': 'red', 'bold': True},
                    'error': {'color': 'red'},
                    'warning': {'color': 'yellow'},
                    'success': {'color': 'green', 'bold': True},
                    'notice': {'color': 'magenta'},
                    'info': {},
                    'verbose': {'color': 'black', 'bright': True},
                    'debug': {'color': 'black'},
                    'spam': {'color': 'black', 'faint': True},
                    }
    coloredlogs.install(level='DEBUG', logger=logger, fmt='%(levelname).1s [%(asctime)s] %(message)s', datefmt="%m-%d %H:%M:%S",
                        field_styles={'asctime': {'color': 'white'}},
                        level_styles=level_styles)

try:
    logger
except NameError:
    init_logger()

if logger is None:
    init_logger()
