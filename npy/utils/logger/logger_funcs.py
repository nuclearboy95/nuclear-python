from .logger_set import logger


__all__ = ['logc', 'loge', 'logw', 'logi', 'logd']


def get_message(*args):
    return ' '.join(args)


def printc(*args):
    message = get_message(*args)
    logger.critical(message)


def printe(*args):
    message = get_message(*args)
    logger.error(message)


def printw(*args):
    message = get_message(*args)
    logger.warning(message)


def printi(*args):
    message = get_message(*args)
    logger.info(message)


def printd(*args):
    message = get_message(*args)
    logger.debug(message)


logc = printc
loge = printe
logw = printw
logi = printi
logd = printd
