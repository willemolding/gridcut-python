"""
Building the GridCut wrapper

>>> python setup.py build_ext --inplace
>>> python setup.py install
"""

import os
import requests
import zipfile

try:
    from setuptools import setup, Extension # , Command, find_packages
    from setuptools.command.build_ext import build_ext
except ImportError:
    from distutils.core import setup, Extension # , Command, find_packages
    from distutils.command.build_ext import build_ext

PACKAGE = 'GridCut-1.3.zip'
zip_file_url = 'http://gridcut.com/dl/%s' % PACKAGE

# DOWNLOAD code
try:  # Python2
    import StringIO
    r = requests.get(zip_file_url, stream=True)
    with zipfile.ZipFile(StringIO.StringIO(r.content)) as zip_ref:
        zip_ref.extractall('code')
except Exception:  # Python3
    import io
    r = requests.get(zip_file_url)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zip_ref:
        zip_ref.extractall('code')

PATH_GRIDCUT = os.path.join('code', 'include', 'GridCut')
assert os.path.exists(PATH_GRIDCUT), 'missing GridCut source code'
PATH_ALPHAEXP = os.path.join('code', 'examples', 'include', 'AlphaExpansion')
assert os.path.exists(PATH_GRIDCUT), 'missing AplhaExpansion source code'
# parse version
VERSION = os.path.splitext(PACKAGE)[0].split('-')[-1]


class BuildExt(build_ext):
    """ build_ext command for use when numpy headers are needed.
    SEE: https://stackoverflow.com/questions/2379898
    SEE: https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process:
        # __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())

setup(
    name='gridcut',
    version=VERSION,
    description='pyGridCut: a python wrapper for the grid-cuts package',
    download_url='http://www.gridcut.com/',
    cmdclass={'build_ext': BuildExt},
    ext_modules=[Extension(
        'gridcut',
        ['gridcut.pyx', 'wrapper_gridCut.cpp'],
        language='c++',
        include_dirs=[PATH_GRIDCUT, PATH_ALPHAEXP],
        extra_compile_args=["-fpermissive"]
        )
    ],
    setup_requires=['requests'],
    install_requires=['numpy'],
    classifiers=[
        'Development Status :: 4 - Beta',
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)
