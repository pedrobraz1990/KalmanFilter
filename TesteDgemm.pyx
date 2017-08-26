import numpy as np
cimport numpy as np
cimport scipy.linalg.cython_blas as blas


def f(np.ndarray[np.double_t,ndim=2, mode='fortran'] a, np.ndarray[np.double_t,ndim=2, mode='fortran'] b):

    a.dot(b)


    cdef double[::1,:] c

    c = np.empty((2, 2), float, order="F")

    cdef double[::1,:] a_view = a
    cdef double[::1,:] b_view = b



    cdef int m, n, k, lda, ldb, ldc
    cdef double alpha, beta

    alpha = 1.0
    beta = 0.0
    lda = 2
    ldb = 2
    ldc = 2
    m = 2
    n = 2
    k = 2

    blas.dgemm('n', #TRANSA
               'n', #TRANSB
               &m, #M
               &n, #N
               &k, #K
               &alpha, #ALPHA
               &a_view[0,0], #MAtrix A
               &lda, #LDA
               &b_view[0,0], #MAtrix B
               &ldb, #LDB
               &beta, #BETA
               &c[0,0], #Matrix C
               &ldc) #LDC

    print(c)

#        blas.dgemm('n', #TRANSA
#               'n', #TRANSB
#               <int *> 2, #M
#               <int *> 2, #N
#               <int *> 2, #K
#               <double *> 1, #ALPHA
#               <double *> a_view, #MAtrix A
#               <int *> 2, #LDA
#               <double *> b_view, #MAtrix B
#               <int *> 2, #LDB
#               <double *> 0, #BETA
#               <double *> c_view, #Matrix C
#               <int *> 0) #LDC