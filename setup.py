from setuptools import setup, find_packages

PACKAGENAME = "sstcam_simulation"
DESCRIPTION = "SST camera low-level simulation package"
AUTHOR = "Jason J Watson"
AUTHOR_EMAIL = "jason.watson@physics.ox.ac.uk"
VERSION = "0.1.0"

setup(
    name=PACKAGENAME,
    packages=find_packages(),
    version=VERSION,
    description=DESCRIPTION,
    license='BSD3',
    install_requires=[
        'astropy',
        'scipy',
        'numpy',
        'matplotlib',
        'tqdm',
        'numba',
        'h5py',
        'PyYAML',
        'ctapipe>=0.7',
        'CHECLabPy @ git+https://github.com/sstcam/CHECLabPy@master',
    ],
    setup_requires=['pytest-runner', ],
    tests_require=['pytest', ],
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    package_data={
        '': ['data/*'],
    },
)
