The nominal NSB rate is 0.23 photons / (sr cm^2 ns). This corresponds to 96.6 MHz photon rate before the PDE is accounted for. So this would be true for any SiPM.

However, illuminating the SiPM with an LED with 96.6 MHz photons @ 405nm is not going to have the same effect on the SiPMs as the same illumination with the NSB spectrum, because of the wavelength dependance of the PDE. In fact this illumination @ 405nm will have a greater effect on the SiPMs than the nominal NSB would, as normally the 96.6 MHz photons would be spread over other wavelengths that the PDE is less sensitive to, rather than at 405nm where it is quite sensitive.

So we need to ask ourselves what photon illumination at 405nm is representative (in the effect it has on the SiPM) as the nominal NSB spectrum. To do this we must calculate the nominal NSB photoelectron rate, and then divide it by the PDE at 405nm:

​			$R^{ph}_{405nm} = \frac{R^{pe}_{nom}}{{PDE}_{405nm}}$

This equals the photon rate needed at 405nm so that we have the same "detectable" rate of photons as we would have from the NSB spectrum.

What is the ratio between this rate and the nominal NSB photon rate?

​		$$\frac{R^{ph}_{nom}}{R^{ph}_{405nm}} = \frac{R^{ph}_{nom}}{R^{pe}_{nom}} \times {PDE}_{405nm} = \frac{\int_{\lambda_1}^{\lambda_2} {F(\lambda)}_{@ pix} \,d\lambda}{\int_{\lambda_1}^{\lambda_2} {F(\lambda)}_{@ pix} \times PDE(\lambda) \,d\lambda} \times {PDE}_{405nm}$$

Where $\lambda_1 = 200 nm$ and $\lambda_2 = 999 nm$. This ratio is around 1.375 for the current default sstcam configuration in the `CameraEfficiency` class. The required LED photon rate is ~70.2 MHz for this configuration.

As you can see from the equations, if the entire PDE curve was changed by a scale factor (e.g. corresponding to a drop in overvoltage), the resulting ratio and required LED photon rate would remain unchanged. This makes calibrating the LED into units of photons better than photoelectrons, as the axis then becomes independant of overvoltage drop.

However, if the PDE shape changes at all, then the ratio will have a different value. Consequently, different SiPM tiles will have a different required LED photon rate (@ 405 nm) to accurately replicate the conditions under nominal NSB. I can provide these photon rates for the different tiles if I am given the PDE curves vs wavelength for each of the SiPM candidates.

