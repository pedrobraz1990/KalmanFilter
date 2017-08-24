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
    at = a1
    Pt = P1
    n = y.shape[0]
    p = y.shape[1]
    m = a1.shape[0] #number of states
    ZT = Z.T #Avoid transposing several times
    TT = T.T
    RQR = np.linalg.multi_dot([R, Q, R.T])
    a = np.empty((m,n))
    a[:, 0] = a1

    nullIndex = np.zeros(n)
    nullIndex[np.isnan(y).any(axis=1)] = 1
    nullIndex[np.isnan(y).all(axis=1)] = 2


    for t in range(0, n - 1):

        ind = nullIndex[t]


        if ind != 1:
            # Eq #1 vt = yt - Zt at
            vt = y[t, :] - np.dot(Z, at) #TODO check if doing it in two steps is faster

            # Eq #2 Ft = Zt Pt Z't + Ht

            Ft = np.linalg.multi_dot([Z, Pt, ZT]) + H #TODO Check function to do several multiplications in a row

            Ft_inv = np.linalg.inv(Ft) #avoid transposing it several times

            if ind == 0:
                # Eq #3 att = at + Pt Z't F-1t vt
                att = at + np.linalg.multi_dot([Pt, ZT, Ft_inv, vt])

                # Eq #4 Ptt = Pt - Pt Z't F-1t Zt Pt
                Ptt = Pt - np.linalg.multi_dot([Pt, ZT, Ft_inv, Z, Pt]) #TODO Maybe store Z't F-1t

            else:
                att = at
                Ptt = Pt


            # Eq #5 at+1 = T att

            at = np.dot(T, att) # Actually at+1
            a[:, t + 1] = at

            # Eq #6 Pt = Tt Ptt T' + R Q R

            Pt = np.linalg.multi_dot([T, Ptt, TT]) + RQR # Actually Pt+1

        else:
            nanIndex = np.isnan(y[t, :])
            Wt = np.identity(p)[~nanIndex]

            # Yst = np.dot(Wt, y[t, :])
            Yst = y[t, :][~nanIndex]


            Zst = np.dot(Wt, Z)
            Hst = np.linalg.multi_dot([Wt,H,Wt.T])
            ZTst = Zst.T

            # Eq #1 vt = yt - Zt at
            vt = Yst - np.dot(Zst, at)  # TODO check if doing it in two steps is faster

            # Eq #2 Ft = Zt Pt Z't + Ht

            Ft = np.linalg.multi_dot([Zst, Pt, ZTst]) + Hst  # TODO Check function to do several multiplications in a row

            Ft_inv = np.linalg.inv(Ft)  # avoid transposing it several times

            # Eq #3 att = at + Pt Z't F-1t vt
            att = at + np.linalg.multi_dot([Pt, ZTst, Ft_inv, vt])

            # Eq #4 Ptt = Pt - Pt Z't F-1t Zt Pt
            Ptt = Pt - np.linalg.multi_dot([Pt, ZTst, Ft_inv, Zst, Pt])  # TODO Maybe store Z't F-1t

            # Eq #5 at+1 = T att

            at = np.dot(T, att)  # Actually at+1
            a[:, t + 1] = at

            # Eq #6 Pt = Tt Ptt T' + R Q R

            Pt = np.linalg.multi_dot([T, Ptt, TT]) + RQR  # Actually Pt+1




    yhat = pd.DataFrame(np.dot(Z, a).T)
    return yhat
