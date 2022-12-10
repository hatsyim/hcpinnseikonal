import os
from setuptools import setup, find_packages

def src(pth):
    return os.path.join(os.path.dirname(__file__), pth)

# Project description
descr = 'Repository for DWXXX: A robust seismic tomography framework via physics-informed machine learning with hard constrained data.'

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
    author='Mohammad Taufik, Tariq Alkhalifah, Umair bin Waheed',
    author_email='mohammad.taufik@kaust.edu.sa, tariq.alkhalifah@kaust.edu.sa, umair.waheed@kfupm.edu.sa',
    install_requires=['numpy >= 1.15.0',
                      'torch >= 1.2.0'],
    packages=find_packages(),
    use_scm_version=dict(root='.',
                         relative_to=__file__,
                         write_to=src('hcpinnseikonal/version.py')),
    setup_requires=['setuptools_scm'],

)
