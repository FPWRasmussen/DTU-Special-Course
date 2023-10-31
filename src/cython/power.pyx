import numpy as np
cimport numpy as np

def power(np.ndarray[double] arr, double power):
    cdef np.ndarray[double] res = arr ** power
    return res