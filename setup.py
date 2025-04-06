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
              'deep learning',
              'tomography',
              'pinns',
              'pde',
              'seismic'],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
    author='Mohammad Taufik',
    author_email='mohammad.taufik@kaust.edu.sa',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('src/hcpinnseikonal/version.py')),
    setup_requires=['setuptools_scm'],

)
