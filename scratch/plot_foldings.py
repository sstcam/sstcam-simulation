from CHECLabPy.plotting.setup import Plotter
import numpy as np
import pandas as pd
from sstcam_simulation.utils.efficiency import CameraEfficiency
from sstcam_simulation.utils.sipm import SiPMOvervoltage
from sstcam_simulation.utils.window_durham_needle import SSTWindowRun3, Window

MINIMGAMP_GAMMA = 250
MINIMGAMP_PROTON = 480
BTEL1170_PDE = 0.2
BTEL0090_SNR = 0.35
PROD4_OPCT = 0.08
PROD4_MINIMGAMP_GAMMA = 153
PROD4_MINIMGAMP_PROTON = 208


def main():
    sipm_tool = SiPMOvervoltage.lvr3_6mm_50um_uncoated()
    sipm_tool.overvoltage = 6
    pde_at_450nm = sipm_tool.pde
    window_tool = SSTWindowRun3()

    eff = CameraEfficiency.from_sstcam(
        fov_angle=0,
        pde_at_450nm=pde_at_450nm,
        window=window_tool,
    )

    p = Plotter()
    p.ax.plot(eff.wavelength, eff._cherenkov_diff_flux_on_ground, "-", color='blue', alpha=0.3, label="On-ground")
    p.ax.plot(eff.wavelength, eff._cherenkov_diff_flux_at_camera, "--", color='blue', alpha=0.6, label="At-camera")
    p.ax.plot(eff.wavelength, eff._cherenkov_diff_flux_at_pixel, "-.", color='blue', alpha=0.9, label="At-pixel")
    p.ax.plot(eff.wavelength, eff._cherenkov_diff_flux_inside_pixel, ":", color='blue', alpha=1, label="Inside-pixel")
    p.ax.set_ylim(0, 0.45)
    p.ax.set_xlabel("Wavelength [nm]")
    p.ax.set_ylabel("Cherenkov photons [100 * 1 / nm]")
    p.save("foldings/spectra_cherenkov.pdf")

    p = Plotter()
    p.ax.plot(eff.wavelength, eff._nsb_diff_flux_on_ground, "-", color='red', alpha=0.3, label="On-ground")
    p.ax.plot(eff.wavelength, eff._nsb_diff_flux_at_camera, "--", color='red', alpha=0.6, label="At-camera")
    p.ax.plot(eff.wavelength, eff._nsb_diff_flux_at_pixel, "-.", color='red', alpha=0.9, label="At-pixel")
    p.ax.plot(eff.wavelength, eff._nsb_diff_flux_inside_pixel, ":", color='red', alpha=1, label="Inside-pixel")
    p.ax.set_ylim(0, 25)
    p.ax.set_xlabel("Wavelength [nm]")
    p.ax.set_ylabel("NSB photons [ 1 / (nm m2 ns sr) ]")
    p.save("foldings/spectra_nsb.pdf")

    p = Plotter()
    camera_sensitivity = eff.pde * eff.window_transmissivity
    p.ax.plot(eff.wavelength, eff.pde, label="PDE")
    p.ax.plot(eff.wavelength, eff.window_transmissivity, label="Window Transmissivity")
    p.ax.plot(eff.wavelength, camera_sensitivity, label="PDE * Window Transmissivity")
    # p.ax.plot(eff.wavelength, eff.telescope_transmissivity, label="Telescope Transmissivity")
    p.ax.plot(eff.wavelength, eff.mirror_reflectivity, label="Mirror Reflectivity")
    p.ax.set_xlabel("Wavelength [nm]")
    p.ax.set_ylabel("Sensitivity")
    p.add_legend()
    p.save("foldings/sensitivity.pdf")

    p = Plotter()
    def add_window(window_tool):
        angles = np.array([0, 20, 45, 50, 60])
        arrays = np.vstack([
            window_tool.df['M0'],
            window_tool.df['M20'],
            window_tool.df['M45'],
            window_tool.df['M50'],
            window_tool.df['M60'],
        ])
        window_tool = Window(incidence_angles=angles, transmission=arrays)
        
        eff = CameraEfficiency.from_sstcam(
            fov_angle=0,
            pde_at_450nm=pde_at_450nm,
            window=window_tool,
        )
        label = f"NSB={eff.nominal_nsb_rate.to_value('MHz'):.2f}, PDE={eff.camera_cherenkov_pde:.2f}, S/N={eff.camera_signal_to_noise:.2f}"
        x = eff.wavelength.value
        y = eff.window_transmissivity
        p.ax.plot(x, y, alpha=0.7, label=label)

    add_window(window_tool)
    # from IPython import embed
    # embed()
    # window_tool.df.loc[window_tool.df.index < 300] = 0
    # add_window(window_tool)
    # window_tool.df.loc[window_tool.df.index > 750] = 0.2
    # add_window(window_tool)
    p.ax.set_xlabel("Wavelength [nm]")
    p.ax.set_ylabel("Sensitivity")
    p.add_legend()
    p.save("foldings/window.pdf")




if __name__ == '__main__':
    main()
