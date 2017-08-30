
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

# for line profiler
from Cython.Compiler.Options import directive_defaults
directive_defaults['linetrace'] = True
directive_defaults['binding'] = True
# for line profiler


import numpy as np

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [

        Extension("CKF3",
                  sources=["CKF3.pyx"],
                  include_dirs=[np.get_include()],
                  # language="c++"
                  ),

        Extension("CKF4",
                  sources=["CKF4.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF5",
                  sources=["CKF5.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF6",
                  sources=["CKF6.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF7",
                  sources=["CKF7.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF8",
                  sources=["CKF8.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF9",
                  sources=["CKF9.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF10",
                  sources=["CKF10.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF11",
                  sources=["CKF11.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF12",
                  sources=["CKF12.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF13",
                  sources=["CKF13.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF14",
                  sources=["CKF14.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF15",
                  sources=["CKF15.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF16",
                  sources=["CKF16.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF17",
                  sources=["CKF17.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF18",
                  sources=["CKF18.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF19",
                  sources=["CKF19.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF20",
                  sources=["CKF20.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF21",
                  sources=["CKF21.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF22",
                  sources=["CKF22.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        Extension("CKF23",
                  sources=["CKF23.pyx"],
                  include_dirs=[np.get_include()],
                  define_macros=[('CYTHON_TRACE', '1')],
                  # language="c++"
                  ),

        # Extension("CKF5",
        #           sources=["CKF5.pyx"],
        #           include_dirs=[np.get_include()],
        #           # language="c++"
        #           ),

        # Extension("TesteDgemm",
        #           sources=["TesteDgemm.pyx"],
        #           include_dirs=[np.get_include()],
        #           # language="c++"
        #           ),

    ],

)


# python setup.py build_ext --inplace
# kernprof -l -v DummyData.py
# python -m cProfile -o DummyData.prof DummyData.py
# snakeviz DummyData.prof
# python -m pstats DummyData.prof