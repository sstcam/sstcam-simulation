# sstcam-simulation [![tests](https://github.com/sstcam/sstcam-simulation/workflows/tests/badge.svg)](https://github.com/sstcam/sstcam-simulation/actions?query=workflow%3Atests) [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sstcam/sstcam-simulation/master)

Low-level and simple simulation package for the SST camera. 


### Purpose

* Provide a simulation framework of the SST camera where we have full control over the camera description, readout chain, and trigger logic. 
* Inform on decisions for the final camera design.
* Demonstrate the capability of the camera design to meet the level-C CTA requirements.


### Why not simtelarray?

1. We want something simple - This project is intended to be a simple Python package, avoiding the need to dig into simtelarray code, or the need to account for all the inputs required for a simtelarray production.
2. We only care about the performance of our camera -  We don't need to worry about ray tracing, array layouts, or other cameras. 
3. Finer specification of the camera - Simtelarray does not allow us to easily investigate some parameters of our camera, such as the crosstalk between pixels, and realistic noise spectrum.
4. Direct control of pixel illumination - Many of the important camera specifications can be examined without Corsika Cherenkov shower and ray tracing simulations. Specifying exactly the average illumination you wish to simulate for each pixel allows more statistics to be gathered with less CPU time.
5. Trigger performance - Investigating all aspects of trigger performance in simtelarray is difficult (due to lack of "noise events"). This is a simple operation in this package.

**It is important to note that this package is not intended to replace simtelarray. Instead, this package should complement it, allowing for a better understanding of the SST camera performance and appropriate simulation input parameters, resulting in a more accurate simtelarray description and final performance expectation.**

## Install

An environment.yml is provided to setup a conda environment with all the 
required dependencies.

```bash
git clone https://github.com/sstcam/sstcam-simulation.git
cd sstcam-simulation
conda env create -f environment.yml
conda activate sstcam-simulation
python setup.py develop
```


## Design

This package does not provide a single pipeline, and is not configured through 
input files. Users are instead expected to create scripts which piece together the 
parts of this package they require to obtain the output they are interested in. This 
provides the user with full control over the camera description, and can avoid 
performing operations that may be unnecessary for a particular investigation. For 
example, investigating the trigger rate of a single superpixel from NSB only requires 
one superpixel to be simulated, and does not require the Cherenkov shower, 
the waveform sampling, or the backplane trigger to be simulated. As a result, 
users gain learn how the SST camera operates, instead of working 
with a black-box simulation. This design also allows the package to be kept 
extremely simple, requiring no complex configuration or factory classes in 
order to flexibly define the camera.

Typical scripts that utilise this package are summarised in four steps:
1. Define the camera (Pulse shape, SPE spectrum, noise spectrum, number of pixels...).
2. Simulate the photoelectrons (NSB, uniform light, Cherenkov shower ellipse...).
3. Acquire the event (readout/trigger) by processing the input through the camera electronics.
4. Perform the analysis you require on the camera outputs to investigate the camera performance.


## Tutorials

Tutorial notebooks are provided in the tutorials directory, detailing the 
possible operations this package provides, and also some demonstrations on 
obtaining camera performance results.

These notebooks can also be ran without installing the package locally, through 
clicking the Binder badge above.


## Common components

* `n_photoelectrons` : Integer number of photoelectrons
* `photoelectron_charge` : Floating point charge that is reported by the photosensor when a photoelectron is generated. PDF of the charge measured is defined by the photosensor's photoelectron spectrum. Units: photoelectrons. The average result when repeatedly summing the charge of N photoelectrons is N p.e..
* `continuous_readout` : Finely sampled array emulating continuous readout from the photosensor. Photoelectrons which arrive during the readout are convolved with the reference pulse shape of the camera. Integral of the readout equals the total charge in p.e..
* `digital_trigger_line` : Boolean output from each superpixel, indicating if the line is "high" (above threshold)
* `waveform` : Sampled waveform, resulting from integrating the continuous readout across bins of typically 1 ns in width. Sum of waveform samples equals the total charge in p.e..


## Schematic

A schematic of the package, the simulation chain, and how its classes map to the components of the SST camera is shown below:

![image](https://user-images.githubusercontent.com/17825673/79463069-a7ad7080-7ff8-11ea-8772-2496a42d2259.png)


