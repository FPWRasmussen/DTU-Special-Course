# power_number_cython.pyx
import numpy as np
cimport numpy as np

def cython_test(np.ndarray[double] arr, double power):
    cdef np.ndarray[double] res = arr ** power
    return res