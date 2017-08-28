import numpy as np
cimport numpy as np
import pandas as pd
import time
from cpython cimport bool
import line_profiler

ctypedef np.double_t DTYPE_t

##### KF3
# Univariate version of durbin and koopman
# Should be the same as CFK4 but switching outer for dot+reshape


#def KalmanFilter(y, Z, Hsq, T, Q, a1, P1, R):

def KalmanFilter(
        np.ndarray[DTYPE_t, ndim=2] y,
        np.ndarray[DTYPE_t, ndim=2] Z,
        np.ndarray[DTYPE_t, ndim=2] Hsq,
        np.ndarray[DTYPE_t, ndim=2] T,
        np.ndarray[DTYPE_t, ndim=2] Q,
        np.ndarray[DTYPE_t, ndim=1] a1,
        np.ndarray[DTYPE_t, ndim=2] P1,
        np.ndarray[DTYPE_t, ndim=2] R
   ):

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

#    cdef double[:,:] v_mv = v
#    cdef double[:,:] F_mv = F
#    cdef double[:,:,:] K_mv = K
#    cdef double[:,:,:] a_mv = a
#    cdef double[:,:,:,:] P_mv = P

    # times = []
    cdef np.ndarray[DTYPE_t,ndim=1] H = np.diag(Hsq) #ONLY WORKS FOR DIAGONAL H

    for t in range(0, n):
        # decide pt and yt
        yind = ~np.isnan(y[t,:])
        pt = yind.sum()
        yt[:pt] = y[t,yind]
        Zt[:pt,:] = Z[yind,:]
        Ht[:pt] = H[yind] #ONLY WORKS FOR DIAGONAL H

        for i in range(0, pt):

            v = yt[i] - np.inner(Zt[i], a[t,i,:])

            # F[t,i] = np.linalg.multi_dot([Z[i], P[t, i,:,:], Z[i]]) + H[i, i]
            F = Zt[i].dot(P[:,:]).dot(Zt[i]) + Ht[i]

            K[:] = P[:,:].dot(Zt[i]) * (F**(-1))

            a[t,i+1,:] = a[t,i,:] + K[:] * v

#            P[:,:] += - np.outer(K * F, K)
            P[:,:] += - np.dot(K[:,None] * F, K[None,:])


        a[t+1,0,:] =  T.dot(a[t, pt, :])

        # P[t+1, 0,:,:] = np.linalg.multi_dot([T, P[t, i + 1,:,:], TT]) + RQR
        P[:, :] = T.dot(P[:,:]).dot(TT) + RQR


        # times.append(temp1 == temp2)


    alpha = a[:n, 0,:]
    yhat = pd.DataFrame(np.dot(Z, alpha.T).T)

    # pd.DataFrame(times).to_pickle("Mult")
    # pd.DataFrame(times).to_pickle("Dot")
    return yhat






# Before Partial Nulls
#     for t in range(0, n):
#
#         for i in range(0, p):
#
#             v[t,i] = y[t,i] - Z[i].dot(a[t,i,:])
#
#             # F[t,i] = np.linalg.multi_dot([Z[i], P[t, i,:,:], Z[i]]) + H[i, i]
#             F[t, i] = Z[i].dot(P[t,i,:,:]).dot(Z[i]) + H[i, i]
#
#             K[t,i,:] = P[t,i,:,:].dot(Z[i]) * (F[t, i]**(-1))
#             a[t,i+1,:] = a[t,i,:] + K[t,i,:] * v[t,i]
#             P[t, i+1,:,:] = P[t, i,:,:] - np.outer(K[t,i,:] * F[t,i], K[t,i,:])
#
#
#         a[t+1,0,:] =  T.dot(a[t, i+1, :])
#
#         # P[t+1, 0,:,:] = np.linalg.multi_dot([T, P[t, i + 1,:,:], TT]) + RQR
#         P[t + 1, 0, :, :] = T.dot(P[t,i,:,:]).dot(TT) + RQR