"""
Building the GridCut wrapper

>>> python setup.py build_ext --inplace
>>> python setup.py install
"""

import os
from distutils.core import setup
# from Cython.Distutils import build_ext
from distutils.extension import Extension
import urllib
import zipfile

import numpy as np

PACKAGE = 'GridCut-1.3.zip'

# download code
package = urllib.URLopener()
package.retrieve('http://gridcut.com/dl/%s' % PACKAGE, PACKAGE)
# unzip the package
with zipfile.ZipFile(PACKAGE, 'r') as zip_ref:
    zip_ref.extractall('code')

PATH_GRIDCUT = os.path.join('code', 'include', 'GridCut')
assert os.path.exists(PATH_GRIDCUT), 'missing GridCut source code'
PATH_ALPHAEXP = os.path.join('code', 'examples', 'include', 'AlphaExpansion')
assert os.path.exists(PATH_GRIDCUT), 'missing AplhaExpansion source code'
# parse version
VERSION = os.path.splitext(PACKAGE)[0].split('-')[-1]


setup(
    name='gridcut',
    version=VERSION,
    # cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension(
            'gridcut',
            ['wrapper_gridCut.cpp'],
            include_dirs=[PATH_GRIDCUT, PATH_ALPHAEXP],
            )
    ],
    include_dirs=[np.get_include()],
)
