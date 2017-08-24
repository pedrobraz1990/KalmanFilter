import numpy as np
import pandas as pd

##### KF1
# Matrix version of durbin and koopman

#TODO If all you want is the likelihood there should be a simplified version of the KF
# With an overhead function and a parameter returnLikelihood = True

# TODO Steps:
# Create a KF that works - Check KF1
# Create a KF for nulls - Check Kf1
# Create a univariate KF that works
# Create a univariate KF that works for nulls

# Separate the KF for likelihood and for getting the states (or not ?)

# Cythonize


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

    yhat = np.empty((n,p))

    for t in range(0, n):

        for i in range(0, p):

            v[t,i] = y[t,i] - np.dot(Z[i], a[t, i,:])
            F[t,i] = np.linalg.multi_dot([Z[i], P[t, i,:,:], Z[i]]) + H[i, i]
            K[t,i,:] = np.dot(P[t, i,:,:], Z[i]) * (F[t, i]**(-1))
            a[t,i+1,:] = a[t,i,:] + K[t,i,:] * v[t,i]
            P[t, i+1,:,:] = P[t, i,:,:] - np.outer(K[t,i,:] * F[t,i], K[t,i,:])


        a[t+1,0,:] = np.dot(T,a[t, i + 1,:])
        P[t+1, 0,:,:] = np.linalg.multi_dot([T, P[t, i + 1,:,:], TT]) + RQR



    alpha = a[:n, 0,:]
    yhat = pd.DataFrame(np.dot(Z, alpha.T).T)

    return yhat