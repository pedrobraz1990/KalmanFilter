import numpy as np
cimport numpy as np

cimport cython

from scipy.linalg.cython_blas cimport dgemm

import time




@cython.boundscheck(False)
@cython.wraparound(False)


def t1(x,y):

    # Test #1 - Naive Python

    start_time = time.time()

    res = np.dot(x,y)

    print("--- FINAL: {sec:1.5f} seconds ---".format(sec = time.time() - start_time))

    print(res.sum().sum())

def t2(x,y):

    # Test #2 - Typed numpy

    cdef np.ndarray[double, ndim=2, mode="c"] typed_x = x.copy()
    cdef np.ndarray[double, ndim=2, mode="c"] typed_y = y.copy()
    cdef np.ndarray[double, ndim=2, mode="c"] typed_res


    start_time = time.time()

    typed_res = np.dot(typed_x,typed_y)

    print("--- FINAL: {sec:1.5f} seconds ---".format(sec = time.time() - start_time))

    print(typed_res.sum().sum())