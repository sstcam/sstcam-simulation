from setuptools import setup, find_packages

PACKAGENAME = "sstcam_simulation"
DESCRIPTION = "SST camera low-level simulation package"
AUTHOR = "Jason J Watson"
AUTHOR_EMAIL = "jason.watson@desy.de"
VERSION = "2.2.0"

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
        'pandas',
        'tables',
        'matplotlib',
        'tqdm',
        'numba',
        'h5py',
        'PyYAML',
        'eventio>=1.4.1',
        'ctapipe~=0.9.0',
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
