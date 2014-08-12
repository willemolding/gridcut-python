from distutils.core import setup, Extension
from os.path import expanduser
setup(name='gridcut', version='1.0',  \
      ext_modules=[Extension('gridcut', ['gridcut_wrapper.cpp'],
      				include_dirs=[expanduser("~/anaconda/lib/python2.7/site-packages/numpy/core/include"),
      								"code/GridCut-1.1/include/GridCut"])])