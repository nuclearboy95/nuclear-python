import numpy as np


__all__ = ['NPYDtypeRange',
           'FLOAT32_0_1', 'FLOAT32_0_255', 'FLOAT32_m1_1', 'FLOAT32_m256_255',
           'FLOAT64_0_1', 'FLOAT64_0_255', 'FLOAT64_m1_1', 'FLOAT64_m256_255',
           'UINT8_0_255']


class NPYDtypeRange:
    min_v = None
    max_v = None
    dtype = None

    @classmethod
    def get_minmax(cls):
        return cls.min_v, cls.max_v

    @classmethod
    def get_dtype(cls):
        return cls.dtype


class FLOAT32_0_1(NPYDtypeRange):
    min_v = 0.
    max_v = 1.
    dtype = np.float32


class FLOAT64_0_1(NPYDtypeRange):
    min_v = 0.
    max_v = 1.
    dtype = np.float64


class FLOAT32_m1_1(NPYDtypeRange):
    min_v = -1.
    max_v = 1.
    dtype = np.float32


class FLOAT64_m1_1(NPYDtypeRange):
    min_v = -1.
    max_v = 1.
    dtype = np.float64


class FLOAT32_0_255(NPYDtypeRange):
    min_v = 0.
    max_v = 255.
    dtype = np.float32


class FLOAT64_0_255(NPYDtypeRange):
    min_v = 0.
    max_v = 255.
    dtype = np.float64


class FLOAT32_m256_255(NPYDtypeRange):
    min_v = -256.
    max_v = 255.
    dtype = np.float32


class FLOAT64_m256_255(NPYDtypeRange):
    min_v = -256.
    max_v = 255.
    dtype = np.float64


class UINT8_0_255(NPYDtypeRange):
    min_v = 0
    max_v = 255
    dtype = np.uint8
