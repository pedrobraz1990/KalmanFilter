
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [

        # Extension("CKF3",
        #           sources=["CKF3.pyx"],
        #           include_dirs=[np.get_include()],
        #           # language="c++"
        #           ),

        Extension("TesteDgemm",
                  sources=["TesteDgemm.pyx"],
                  include_dirs=[np.get_include()],
                  # language="c++"
                  ),

    ],

)


# python setup.py build_ext --inplace