import numpy as np
cimport numpy as np
import pandas as pd
import time
from cpython cimport bool
import line_profiler
cimport cython

from libc.math cimport isnan

ctypedef np.double_t DTYPE_t

##### KF
# Univariate version of durbin and koopman
# Should be the same as CKF14 using nogil


#bla

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int getSum(double [:,:] arr,int t, int p) nogil:

    cdef int i
    cdef int s = 0

    for i in range(0,p):
#        print(arr[i])
        if not isnan(arr[t,i]):
             s += 1
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef double innerProduct(double [:] arr1, double [:] arr2 , int m,) nogil:

    cdef double res = 0
    cdef int i

    for i in range(0,m):
        res += arr1[i] * arr2[i]
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void outerProductP(double [:] arr1, double [:] arr2 , int m, double [:,:] res, double F) nogil:

#    cdef double[:,:] res
    cdef int i
    cdef int j

    for i in range(0,m):
        for j in range(0,m):
            res[i,j] -= arr1[i] * arr1[j] * F
#    return res


#@cython.initializedcheck(False)
#@cython.nonecheck(False)
#@profile
@cython.cdivision(True)
@cython.boundscheck(False)
@cython.wraparound(False)
def KalmanFilter(
        np.ndarray[DTYPE_t, ndim=2] y,
        np.ndarray[DTYPE_t, ndim=2] Z,
        np.ndarray[DTYPE_t, ndim=2] Hsq,
        np.ndarray[DTYPE_t, ndim=2] T,
        np.ndarray[DTYPE_t, ndim=2] Q,
        np.ndarray[DTYPE_t, ndim=1] a1,
        np.ndarray[DTYPE_t, ndim=2] P1,
        np.ndarray[DTYPE_t, ndim=2] R
   ) :

    # p = number of variables in Yt
    # y should be (n x p)

    cdef int n,p,m,t,i,pt

    n = y.shape[0]
    p = y.shape[1]
    m = a1.shape[0] #number of states


    cdef np.ndarray[DTYPE_t,ndim=3] a = np.empty((n+1,p+1,m))
    a[0,0,:] = a1

    cdef np.ndarray[DTYPE_t,ndim=2] P = np.empty((m, m))
    P[:,:] = P1

    cdef np.ndarray[DTYPE_t,ndim=1] K = np.empty((m))

    cdef double v,F


    # RQR = np.linalg.multi_dot([R, Q, R.T])
    cdef np.ndarray[DTYPE_t,ndim=2] RQR = R.dot(Q).dot(R.T)

    cdef np.ndarray[DTYPE_t,ndim=2] TT = T.T

    yhat = np.empty((n,p)) #Later I should use it to export in numpy not pandas

#    cdef np.ndarray[np.uint8_t,ndim=1, cast=True] yind = np.empty(p)
    yind = np.empty(p)

    cdef np.ndarray[DTYPE_t,ndim=1] yt = np.empty(p)
    cdef np.ndarray[DTYPE_t,ndim=2] Zt = np.empty((p, m))
    cdef np.ndarray[DTYPE_t,ndim=1] Ht = np.empty(p)


    cdef np.ndarray[DTYPE_t,ndim=1] H = np.diag(Hsq) #ONLY WORKS FOR DIAGONAL H

    cdef double[:,:] y_mv = y
    cdef double[:,:,:] a_mv = a
    cdef double[:,:] Zt_mv = Zt
    cdef double[:] K_mv = K
    cdef double[:,:] P_mv = P

    yindGlobal = ~np.isnan(y)

    for t in range(0, n):
        # decide pt and yt
        yind = yindGlobal[t,:]

#        pt = yind.sum()

#        print(pt)
#        print(yind)
#        print(y[t,:])
        pt = getSum(y_mv, t, p)

        yt[:pt] = y[t,yind]
        Zt[:pt,:] = Z[yind,:]
        Ht[:pt] = H[yind] #ONLY WORKS FOR DIAGONAL H

        for i in range(0, pt):

#            v = yt[i] - np.dot(Zt[i], a[t,i,:])
            v = yt[i] - innerProduct(Zt_mv[i],a_mv[t,i,:],m)

            # F[t,i] = np.linalg.multi_dot([Z[i], P[t, i,:,:], Z[i]]) + H[i, i]
            F = Zt[i].dot(P[:,:]).dot(Zt[i]) + Ht[i]

            K[:] = P[:,:].dot(Zt[i]) * (1/F)

            a[t,i+1,:] = a[t,i,:] + K[:] * v

#            P[:,:] += - np.outer(K * F, K)
#            P[:,:] += - np.dot(K[:,None] * F, K[None,:])
            outerProductP(K_mv,K_mv,m,P_mv, F)


        a[t+1,0,:] =  T.dot(a[t, pt, :])

        # P[t+1, 0,:,:] = np.linalg.multi_dot([T, P[t, i + 1,:,:], TT]) + RQR
        P[:, :] = T.dot(P[:,:]).dot(TT) + RQR


        # times.append(temp1 == temp2)


    alpha = a[:n, 0,:]
    yhat = pd.DataFrame(np.dot(Z, alpha.T).T)

    # pd.DataFrame(times).to_pickle("Mult")
    # pd.DataFrame(times).to_pickle("Dot")
    return yhat

