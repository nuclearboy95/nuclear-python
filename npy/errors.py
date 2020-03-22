
__all__ = ['UnknownImageShapeError', 'UnknownImageFormatError', 'UnknownImageDtypeError',
           'UnknownImageDtypeRangeError', 'UnknownImageFloatRangeError']


class NPYError(ValueError):
    pass


# Image

class UnknownImageShapeError(NPYError):
    def __init__(self, shape_):
        super().__init__(f'Unknown shape: {shape_}')


class UnknownImageFormatError(NPYError):
    def __init__(self, fmt):
        super().__init__(f'Unknown fmt: {fmt}')


class UnknownImageDtypeError(NPYError):
    def __init__(self, dtype):
        super().__init__(f'Unknown image dtype: {dtype}')


class UnknownImageDtypeRangeError(NPYError):
    def __init__(self, dtype_range):
        super().__init__(f'Unknown image dtype_range: {dtype_range}')


class UnknownImageFloatRangeError(NPYError):
    def __init__(self, min_v, max_v):
        super().__init__(f'Unknown float image range. Min: {min_v}, Max: {max_v}')
