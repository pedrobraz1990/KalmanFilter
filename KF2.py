import numpy as np
import pandas as pd
import time

##### KF2
# Univariate version of durbin and koopman



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

    RQR = np.linalg.multi_dot([R, Q, R.T])

    TT = T.T

    yhat = np.empty((n,p)) #Later I should use it to export in numpy not pandas

    yind = np.empty(p)


    # times = []

    for t in range(0, n):
        # decide pt and yt
        yind = ~np.isnan(y[t,:]) #TODO debug to make sure
        yt = y[t,yind]
        pt = yind.sum()

        for i in range(0, pt):

            v[t,i] = yt[i] - np.inner(Z[i], a[t,i,:])

            # F[t,i] = np.linalg.multi_dot([Z[i], P[t, i,:,:], Z[i]]) + H[i, i]
            F[t, i] = Z[i].dot(P[t,i,:,:]).dot(Z[i]) + H[i, i]

            K[t,i,:] = P[t,i,:,:].dot(Z[i]) * (F[t, i]**(-1))

            a[t,i+1,:] = a[t,i,:] + K[t,i,:] * v[t,i]

            P[t, i+1,:,:] = P[t, i,:,:] - np.outer(K[t,i,:] * F[t,i], K[t,i,:])


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