# RAW COPY PASTE FROM THE "KALMAN FILTER PERFORMANCE COMPARISON" NOTEBOOK
import numpy as np
import pandas as pd
import KF1
import KF2
import KF_PaperUni
import KF3
import CKF3
import CKF4
import CKF5
import CKF7
import CKF8
import CKF10
import CKF13
import CKF14
import CKF18
import CKF20
import CKF21
import CKF23
import CKF24

# PARAMETERS
m = 2
p = 4

Z = [[0.3, 0.7], [0.1, 0], [0.5, 0.5], [0, 0.3]]

Z = pd.DataFrame(Z)

H = pd.DataFrame(np.diag([1.0, 2.0, 3.0, 4.0]))

T = pd.DataFrame(np.identity(2))
R = pd.DataFrame(np.identity(2))

Q = pd.DataFrame(np.diag([0.2, 0.4]))

# GENERATE DATA

n = 10000  # sample size
mut = [np.array([1, 10]).reshape(m, 1)]
yt = [np.array([0, 0, 0, 0]).reshape(p, 1)]

for i in range(0, 1000):
    temp = np.multiply(np.random.randn(m, 1), np.diag(Q).reshape((m, 1)))
    temp = R.dot(temp)
    temp = temp + mut[i]
    mut.append(temp)

    temp = np.multiply(np.random.randn(p, 1), np.diag(H).reshape((p, 1)))
    yt.append(temp + Z.dot(mut[i + 1]))

yt[0] = pd.DataFrame(yt[0])
y = pd.concat(yt, axis=1).T.reset_index(drop=True)
mut[0] = pd.DataFrame(mut[0])
mut = pd.concat(mut, axis=1).T.reset_index(drop=True)

# GENERATE NULL

nny = y.copy()
probNan = 0.20
for i in nny.index:
    ran = np.random.uniform(size=nny.iloc[i].shape)
    nny.iloc[i][ran < probNan] = np.nan



y = np.array(y)
nny = np.array(nny)
Z = np.array(Z)
H = np.array(H)
T = np.array(T)
R = np.array(R)
Q = np.array(Q)

#a1 = (m x 1)
a1 = np.zeros(m)
#P1 = (m x m)
P1 = np.diag(np.ones(m) * 1.0)



# ret = CKF3.KalmanFilter(y = y, Z = Z,H = H,T = T,R = R,Q = Q,a1 = a1,P1 = P1)
# ret = CKF3.KalmanFilter(y,Z,H,T,Q,a1,P1, R)
ret = CKF24.KalmanFilter(nny,Z,H,T,Q,a1,P1, R)

# ret = KF3.KalmanFilter(
#     # y = y,
#     y = nny,
#     Z = Z,
#     H = H,
#     T = T,
#     R = R,
#     Q = Q,
#     a1 = a1,
#     P1 = P1,
# )


# ret = KF_PaperUni.KalmanFilter(
#     y = y,
# #     y = wny,
# #     y = nny,
#     Z = Z,
#     H = H,
#     T = T,
#     R = R,
#     Q = Q,
#     a1 = a1,
#     P1 = P1,
#     nStates = a1.shape[0],
#     export=True
# )

# print(ret)


# kernprof -l -v DummyData.py
# python -m line_profiler DummyData.py.lprof
