import os
from glob import glob
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'Stable neural network-based traveltime tomography using hard-constrained measurements.'

setup(
    name="hcpinnseikonal",
    description=descr,
    long_description=open(src('README.md')).read(),
    keywords=['inverse problems',
              'generative models',
              'deep learning',
              'tomography',
              'fwi',
              'seismic'],
    author='Mohammad Hasyim Taufik',
    author_email='mohammad.taufik@kaust.edu.sa',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'setuptools_scm',
    ],
)