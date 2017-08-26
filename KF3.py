import numpy as np
import pandas as pd
import time

##### KF3
# Univariate version of durbin and koopman
# Should be the same as KF2 but without storing useless data

# @profile
def KalmanFilter(y, Z, H, T, Q, a1, P1, R):

    # p = number of variables in Yt
    # y should be (n x p)

    n = y.shape[0]
    p = y.shape[1]
    m = a1.shape[0] #number of states


    a = np.empty((n+1,p+1,m))
    a[0,0] = a1

    P = np.empty((n+1, p+1, m, m))
    P[0, 0] = P1

    K = np.empty((n, p, m))
    v = np.empty((n, p))
    F = np.empty((n, p))

    # RQR = np.linalg.multi_dot([R, Q, R.T])
    RQR = R.dot(Q).dot(R.T)

    TT = T.T

    yhat = np.empty((n,p)) #Later I should use it to export in numpy not pandas

    yind = np.empty(p)
    yt = np.empty(p)
    Zt = np.empty((p, m))
    Ht = np.empty(p)


    # times = []
    H = np.diag(H) #ONLY WORKS FOR DIAGONAL H

    for t in range(0, n):
        # decide pt and yt
        yind = ~np.isnan(y[t,:])
        pt = yind.sum()
        yt[:pt] = y[t,yind]
        Zt[:pt,:] = Z[yind,:]
        Ht[:pt] = H[yind] #ONLY WORKS FOR DIAGONAL H

        for i in range(0, pt):

            v[t,i] = yt[i] - np.inner(Zt[i], a[t,i,:])

            # F[t,i] = np.linalg.multi_dot([Z[i], P[t, i,:,:], Z[i]]) + H[i, i]
            F[t, i] = Zt[i].dot(P[t,i,:,:]).dot(Zt[i]) + Ht[i]

            K[t,i,:] = P[t,i,:,:].dot(Zt[i]) * (F[t, i]**(-1))

            a[t,i+1,:] = a[t,i,:] + K[t,i,:] * v[t,i]

            P[t, i+1,:,:] = P[t, i,:,:] - np.outer(K[t,i,:] * F[t,i], K[t,i,:])
            # TODO try to overwrite Pt everytime since I don't use it

        a[t+1,0,:] =  T.dot(a[t, pt, :])

        # P[t+1, 0,:,:] = np.linalg.multi_dot([T, P[t, i + 1,:,:], TT]) + RQR
        P[t + 1, 0, :, :] = T.dot(P[t,pt,:,:]).dot(TT) + RQR


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