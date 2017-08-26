
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [

        Extension("Test_arraySum",
                  sources=["Test_arraySum.pyx", "c_sum.c"],
                  include_dirs=[np.get_include()],
                  # language="c++"
                  ),

    ],

)


# python setup.py build_ext --inplace