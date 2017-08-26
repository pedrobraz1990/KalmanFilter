import numpy as np
import TesteDgemm as dgemm

a = np.array(np.random.rand(3,3),float,order='F')
b = np.array(np.random.rand(3,3),float, order='F')
c = np.empty((3,3),float,order='F')

dgemm.f(a,b)