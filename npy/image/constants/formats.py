__all__ = ['NPYImageFormat', 'NHWC', 'NHW', 'HWC', 'HW']


class NPYImageFormat:
    @classmethod
    def get_shape(cls, N, H, W, C):
        return NotImplemented


class NHWC(NPYImageFormat):
    @classmethod
    def get_shape(cls, N, H, W, C):
        return N, H, W, C


class NHW(NPYImageFormat):
    @classmethod
    def get_shape(cls, N, H, W, C):
        return N, H, W


class HWC(NPYImageFormat):
    @classmethod
    def get_shape(cls, N, H, W, C):
        return H, W, C


class HW(NPYImageFormat):
    @classmethod
    def get_shape(cls, N, H, W, C):
        return H, W
