import numpy as np
cimport numpy as np
import pandas as pd
import time
from cpython cimport bool
import line_profiler
cimport cython

from libc.math cimport isnan
from libc.math cimport log
from libc.math cimport M_PI

ctypedef np.double_t DTYPE_t

##### KF
# Univariate version of durbin and koopman
# Should be the same as CKF24 with a constant defined



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int getSum(double [:,:] arr,int t, int p) nogil:

    cdef int i
    cdef int s = 0

    for i in range(0,p):
#        print(arr[i])
        if not isnan(arr[t,i]):
             s += 1
    return s


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double yZa(double [:] arr1, double [:] arr2 , int m, double y) nogil:

    cdef double res = 0
    cdef int i

    for i in range(0,m):
        res += arr1[i] * arr2[i]
    return y-res


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void outerProductP(double [:] arr1, double [:] arr2 , int m, double [:,:] res, double F) nogil:

#    cdef double[:,:] res
    cdef int i
    cdef int j

    for i in range(0,m):
        for j in range(0,m):
            res[i,j] -= arr1[i] * arr1[j] * F
#    return res



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void PZF(double[:] K, double F, double[:,:] P, double[:] Zti ,
int n, int p, int m) nogil:
#Note n, m and p are the generical matrix sizes from
#https://en.wikipedia.org/wiki/Matrix_multiplication_algorithm

#K[:] = P[:,:].dot(Zt[i]) * (1/F)
#K(m x 1) = P(m x m) * Zt[i](m x 1) * (1/F) (1 x 1)
#Matrix = problem
# n = m
# p = 1
# m = m

    cdef double invF = 1/F
    for i in range(0,n):
        K[i] = 0
        for k in range(0,m):
            K[i] += P[i,k] * Zti[k]
        K[i] *= invF



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void AKv(double[:] a1, double[:] a0 , double[:] K, double v ,
int m) nogil:

#a[t,i+1,:] = a[t,i,:] + K[:] * v
#AKv(a_mv[t,i+1,:], a_mv[t,i,:], K_mv[:], v, m)

    for i in range(0,m):
        a1[i] = a0[i] + K[i] * v



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline double ZPZH(double[:] Zt, double[:,:] P, double Ht, int m) nogil:

#F = Zt[i].dot(P[:,:]).dot(Zt[i]) + Ht[i]
#F(1 x 1) = Zt[i] (1 x m) * P (m x m) * Zt[i] (m x 1) + Ht[i] (1 x 1)

    cdef double F = 0
    cdef double temp = 0
    for i in range(0,m):
        temp = 0
        for j in range(0,m):
            temp += Zt[j]*P[j,i]

        F += temp * Zt[i]

    return F + Ht


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void TPT(double[:,:] T, double[:,:] P, double[:,:] TT,
double[:,:] RQR, int m, double[:,:] temp) nogil:

# P[:, :] = T.dot(P[:,:]).dot(TT) + RQR
# P(m x m) = T(m x m) P(m x m) TT(m x m) + RQR (m x m)
# Will do 2 matrix multiplications, can be done better, by doing it in a single
#operation


    cdef double tempk2, tempk1

    for i in range(0,m):
        for j in range(0,m):
            tempk1 = 0
            for k1 in range(0,m):
                tempk2 = 0
                for k2 in range(0,m):
                    tempk2 += T[i,k2] * P[k2,k1]
                tempk1 += tempk2 * TT[j,i]
            temp[i,j] = tempk1

    for i in range(0,m):
        for j in range(0,m):
            P[i,j] = temp[i,j] + RQR[i,j]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void Ta(double[:] a1, double[:] a0 , double[:,:] T, int m) nogil:

# a[t+1,0,:] =  T.dot(a[t, pt, :])
#a(m x 1) = T(m x m) * a(m x 1)
    for i in range(0,m):
        a1[i] = 0
        for j in range(0,m):
            a1[i] += T[i,j] * a0[j]



@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int indexNans(double[:] yt, double[:] yt_orig,double[:,:] Zt, double[:,:] Zt_orig, double[:] H, double[:] H_orig, int p) nogil:
#yt[:pt] = y[t,yind]
#Zt[:pt,:] = Z[yind,:]
#Ht[:pt] = H[yind]
#should do all that and return pt
#yt(p x 1)
#Zt(p x p)

    cdef int i,j
    cdef int pt = 0

    for i in range(0,p):
        if not isnan(yt_orig[i]):
            yt[pt] = yt_orig[i]
            H[pt] = H_orig[i]

            for j in range(0,p):
                Zt[pt,j] = Zt_orig[i,j]

            pt += 1

    return pt




#@cython.initializedcheck(False)
#@cython.nonecheck(False)
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
        np.ndarray[DTYPE_t, ndim=2] R,
        likelihood = True,
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

    cdef np.ndarray[DTYPE_t,ndim=2] mmTemp = np.empty((m, m))

    cdef np.ndarray[DTYPE_t,ndim=1] K = np.empty((m))

    cdef double v,F


    # RQR = np.linalg.multi_dot([R, Q, R.T])
    cdef np.ndarray[DTYPE_t,ndim=2] RQR = R.dot(Q).dot(R.T)

    cdef np.ndarray[DTYPE_t,ndim=2] TT = T.T

    yhat = np.empty((n,p)) #Later I should use it to export in numpy not pandas


    cdef np.ndarray[DTYPE_t,ndim=1] yt = np.empty(p)
    cdef np.ndarray[DTYPE_t,ndim=1] yInd = np.empty(p)
    cdef np.ndarray[DTYPE_t,ndim=2] Zt = np.empty((p, m))
    cdef np.ndarray[DTYPE_t,ndim=1] Ht = np.empty(p)


    cdef np.ndarray[DTYPE_t,ndim=1] H = np.diag(Hsq) #ONLY WORKS FOR DIAGONAL H

#    cdef np.ndarray[DTYPE_t,ndim=2] y_orig = y.copy()
    cdef np.ndarray[DTYPE_t,ndim=2] Z_orig = Z.copy()
    cdef np.ndarray[DTYPE_t,ndim=1] H_orig = H.copy()


    cdef double[:,:] y_mv = y
#    cdef double[:,:] y_orig_mv = y_orig
    cdef double[:,:] Z_orig_mv = Z_orig
    cdef double[:] H_orig_mv = H_orig
    cdef double[:] yt_mv = yt
    cdef double[:] yInd_mv = yInd
    cdef double[:,:,:] a_mv = a
    cdef double[:,:] Zt_mv = Zt
    cdef double[:] K_mv = K
    cdef double[:,:] P_mv = P
    cdef double[:] Ht_mv = Ht
    cdef double[:,:] T_mv = T
    cdef double[:,:] TT_mv = TT
    cdef double[:,:] RQR_mv = RQR
    cdef double[:,:] mmTemp_mv = mmTemp

    cdef double pst
    cdef double pstSum
    cdef double ll = 0
    cdef double lpi= log(2*M_PI)

    for t in range(0, n):


#        pt = getSum(y_mv, t, p)
#
#        yt[:pt] = y[t,yind]
#        Zt[:pt,:] = Z[yind,:]
#        Ht[:pt] = H[yind] #ONLY WORKS FOR DIAGONAL H

        pt = indexNans(yt_mv, y_mv[t,:],Zt_mv[:,:], Z_orig_mv[:,:],Ht_mv[:],H_orig_mv[:], p)

        pst = 0


        for i in range(0, pt):

#            v = yt[i] - np.dot(Zt[i], a[t,i,:])
#            v = yt_mv[i] - innerProduct(Zt_mv[i],a_mv[t,i,:],m)
            v =  yZa(Zt_mv[i],a_mv[t,i,:],m, yt_mv[i])

            # F[t,i] = np.linalg.multi_dot([Z[i], P[t, i,:,:], Z[i]]) + H[i, i]
#            F = Zt[i].dot(P[:,:]).dot(Zt[i]) + Ht[i]
            F = ZPZH(Zt_mv[i], P_mv[:,:], Ht_mv[i], m)

            if likelihood:
                if F>0:
                    ll += log(F) + (v ** 2) / F
                    pst += 1

#            K[:] = P[:,:].dot(Zt[i]) * (1/F)
            PZF(K_mv[:], F, P_mv[:,:], Zt_mv[i] , m, 1, m)

#            a[t,i+1,:] = a[t,i,:] + K[:] * v
            AKv(a_mv[t,i+1,:], a_mv[t,i,:], K_mv[:], v, m)

#            P[:,:] += - np.outer(K * F, K)
#            P[:,:] += - np.dot(K[:,None] * F, K[None,:])
            outerProductP(K_mv,K_mv,m,P_mv, F)

        if likelihood:
            ll += pst * lpi

#        a[t+1,0,:] =  T.dot(a[t, pt, :])
        Ta(a_mv[t+1,0,:], a_mv[t, pt, :], T_mv, m)

        # P[t+1, 0,:,:] = np.linalg.multi_dot([T, P[t, i + 1,:,:], TT]) + RQR
#        P[:, :] = T.dot(P[:,:]).dot(TT) + RQR
        TPT(T_mv, P_mv, TT_mv, RQR_mv, m, mmTemp_mv)

        # times.append(temp1 == temp2)


    alpha = a[:n, 0,:]
    yhat = pd.DataFrame(np.dot(Z, alpha.T).T)
    ll *= -0.5
    # pd.DataFrame(times).to_pickle("Mult")
    # pd.DataFrame(times).to_pickle("Dot")

    if likelihood:
        return ll

    return yhat

