from .setters import logger


__all__ = ['saye', 'sayw', 'sayi', 'sayd', 'says']


def get_message(*args):
    return ' '.join([str(v) for v in args])


def printe(*args):
    msg = get_message(*args)
    logger.critical(msg)


def printw(*args):
    msg = get_message(*args)
    logger.error(msg)


def prints(*args):
    msg = get_message(*args)
    logger.warning(msg)


def printi(*args):
    msg = get_message(*args)
    logger.info(msg)


def printd(*args):
    msg = get_message(*args)
    logger.debug(msg)


saye = printe
sayw = printw
sayi = printi
sayd = printd
says = prints
