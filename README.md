# sstcam-simulation ![tests](https://github.com/sstcam/sstcam-simulation/workflows/tests/badge.svg)
Low-level and simple simulation package for the SST Camera. Intended to inform on decisions for the final camera design.

### Why not simtelarray?
1. We want something simple - This project is intended to be a simple Python package, avoiding the need to dig into simtelarray code, or the need to account for all the inputs required for a simtelarray production.
2. We only care about the performance of our camera -  We don't need to worry about ray tracing, array layouts, or other cameras. 
3. Finer specification of the camera - simtelarray does not allow us to easily investigate some parameters of our camera, such as the real saturation behaviour and ASIC calibration.
4. Trigger performance - investigating all aspects of trigger performance in simtelarray is difficult (due to lack of "noise events"). Writing a simple simulation package ourselves for this should be very easy.

## Setup

### Prerequisites
It is recommended to use a conda environment running Python 3.6 (or above).
Instructions on how to setup such a conda environment can be found in
https://forge.in2p3.fr/projects/gct/wiki/Installing_CHEC_Software. The
required python dependencies (which can be installed using
`conda install ...` or `pip install ...`) are:
* numpy
* scipy
* astropy (for units)
* matplotlib
* tqdm
* numba

### Install
```bash
cd ~/Software
git clone https://github.com/cta-chec/sstCASSIM.git
cd sstCASSIM
conda activate <environment_name>
python setup.py develop
```

### Contributing
Currently pushing directly to this repository is permitted. This will change 
in the future.
```bash
git add ...  # Stage files
git commit -m  # Create commit
git push -u origin master  # First time push
git push  # Future pushes
```

### Updating local copy
```bash
git pull
```
