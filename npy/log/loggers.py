from .setters import _logger


__all__ = ['saye', 'sayw', 'sayi', 'sayd', 'says']


def get_message(*args):
    return ' '.join([str(v) for v in args])


def printe(*args):
    msg = get_message(*args)
    _logger.critical(msg)


def printw(*args):
    msg = get_message(*args)
    _logger.error(msg)


def prints(*args):
    msg = get_message(*args)
    _logger.warning(msg)


def printi(*args):
    msg = get_message(*args)
    _logger.info(msg)


def printd(*args):
    msg = get_message(*args)
    _logger.debug(msg)


saye = printe
sayw = printw
sayi = printi
sayd = printd
says = prints
