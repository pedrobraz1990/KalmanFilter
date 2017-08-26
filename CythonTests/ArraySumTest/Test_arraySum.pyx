

cimport cython


import numpy as np
cimport numpy as np

import time




cdef extern double c_sum (double* array, int size)


@cython.boundscheck(False)
@cython.wraparound(False)

def main():
    # Test #1

    n = 100000000
    x = np.random.rand(n)

    start_time = time.time()

    s = x.sum()


    print("--- FINAL: {sec:1.5f} seconds ---".format(sec = time.time() - start_time))


    print(s)


    #Test #2

    cdef int n2 = 100000000

    cdef np.ndarray x2 = np.empty(n2, dtype = np.double)

    x2 = np.random.rand(n2)

    #x =

    start_time = time.time()

    s2 = x2.sum()


    print("--- FINAL: {sec:1.5f} seconds ---".format(sec = time.time() - start_time))


    print(s2)



    #Test #3
    cdef int n3 = n2 #*100
    cdef int i
    cdef double s3 = 0

    cdef np.ndarray[double, ndim=1, mode="c"] x3 = np.empty(n3, dtype = np.double)

    x3 = np.random.rand(n3)


    cdef double [:] x3_view = x3




    start_time = time.time()

    for i in range(n3):
        s3 += x3_view[i]


    print("--- FINAL: {sec:1.10f} seconds ---".format(sec = time.time() - start_time))

    print(s3)

    #Test #4
    cdef double s4


    start_time = time.time()

    s4 = c_sum(&x3[0], n3)

    print("--- FINAL: {sec:1.10f} seconds ---".format(sec = time.time() - start_time))

    print(s4)