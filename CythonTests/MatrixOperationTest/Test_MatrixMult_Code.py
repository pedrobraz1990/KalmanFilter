import numpy as np

import Test_MatrixMult

dim = 50
x = np.random.rand(dim,dim)
y = np.random.rand(dim,dim)

Test_MatrixMult.t1(x,y)
Test_MatrixMult.t2(x,y)