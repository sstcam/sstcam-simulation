from CHECLabPy.plotting.setup import Plotter
import numpy as np
import pandas as pd
from sstcam_simulation.utils.efficiency import CameraEfficiency

MINIMGAMP_GAMMA = 250
MINIMGAMP_PROTON = 480
BTEL1170_PDE = 0.2
BTEL0090_SNR = 0.35
PROD4_OPCT = 0.08
PROD4_MINIMGAMP_GAMMA = 153
PROD4_MINIMGAMP_PROTON = 208


def main():
    path = "candidate/performance.h5"
    with pd.HDFStore(path) as store:
        df = store['data']

    df = df.loc[df['fov_angle'] == 0]

    prod4_eff = CameraEfficiency.from_prod4()

    p_cherenkov_pde = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_cherenkov_pde.ax.plot(group['overvoltage'], group['camera_cherenkov_pde'], 'x-', label=candidate)
    p_cherenkov_pde.ax.axhline(prod4_eff.camera_cherenkov_pde, ls='--', color='blue', label="Prod4")
    p_cherenkov_pde.ax.set_xlabel("Overvoltage (V)")
    p_cherenkov_pde.ax.set_ylabel("Camera Cherenkov PDE")
    p_cherenkov_pde.add_legend(loc="best")
    p_cherenkov_pde.save("candidate/sipm/camera_cherenkov_pde.pdf")

    p_B_TEL_1170_pde = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_B_TEL_1170_pde.ax.plot(group['overvoltage'], group['B_TEL_1170_pde'], 'x-', label=candidate)
    p_B_TEL_1170_pde.ax.axhline(BTEL1170_PDE, ls='--', color='black', label="Requirement")
    p_B_TEL_1170_pde.ax.axhline(prod4_eff.B_TEL_1170_pde, ls='--', color='blue', label="Prod4")
    p_B_TEL_1170_pde.ax.set_xlabel("Overvoltage (V)")
    p_B_TEL_1170_pde.ax.set_ylabel("B-TEL-1170 PDE")
    p_B_TEL_1170_pde.add_legend(loc="best")
    p_B_TEL_1170_pde.save("candidate/sipm/B_TEL_1170_pde.pdf")

    p_sn = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_sn.ax.plot(group['overvoltage'], group['telescope_signal_to_noise'], 'x-', label=candidate)
    p_sn.ax.axhline(BTEL0090_SNR, ls='--', color='black', label="Requirement")
    p_sn.ax.axhline(prod4_eff.telescope_signal_to_noise, ls='--', color='blue', label="Prod4")
    p_sn.ax.set_xlabel("Overvoltage (V)")
    p_sn.ax.set_ylabel("B-TEL-0090 SNR")
    p_sn.add_legend(loc="best")
    p_sn.save("candidate/sipm/B-TEL-0090_snr.pdf")

    p_opct = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_opct.ax.plot(group['overvoltage'], group['opct'], 'x-', label=candidate)
    p_opct.ax.axhline(PROD4_OPCT, ls='--', color='blue', label="Prod4")
    p_opct.ax.set_xlabel("Overvoltage (V)")
    p_opct.ax.set_ylabel("Optical Crosstalk")
    p_opct.add_legend(loc="best")
    p_opct.save("candidate/sipm/opct.pdf")

    p_nominal_nsb = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_nominal_nsb.ax.plot(group['overvoltage'], group['nominal_nsb_rate'], 'x-', label=candidate)
    p_nominal_nsb.ax.axhline(prod4_eff.nominal_nsb_rate.to_value("MHz"), ls='--', color='blue', label="Prod4")
    p_nominal_nsb.ax.set_xlabel("Overvoltage (V)")
    p_nominal_nsb.ax.set_ylabel("Nominal NSB Rate (MHz)")
    p_nominal_nsb.add_legend(loc="best")
    p_nominal_nsb.save("candidate/sipm/nominal_nsb.pdf")

    p_maximum_nsb = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_maximum_nsb.ax.plot(group['overvoltage'], group['maximum_nsb_rate'], 'x-', label=candidate)
    p_maximum_nsb.ax.axhline(prod4_eff.maximum_nsb_rate.to_value("MHz"), ls='--', color='blue', label="Prod4")
    p_maximum_nsb.ax.set_xlabel("Overvoltage (V)")
    p_maximum_nsb.ax.set_ylabel("Maximum NSB Rate (MHz)")
    p_maximum_nsb.add_legend(loc="best")
    p_maximum_nsb.save("candidate/sipm/maximum_nsb.pdf")

    p_mia_g = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_mia_g.ax.plot(group['overvoltage'], group['minimum_image_amplitude'], 'x-', label=candidate)
    p_mia_g.ax.axhline(MINIMGAMP_GAMMA, ls='--', color='black', label="Requirement")
    p_mia_g.ax.axhline(PROD4_MINIMGAMP_GAMMA, ls='--', color='blue', label="Prod4")
    p_mia_g.ax.set_xlabel("Overvoltage (V)")
    p_mia_g.ax.set_ylabel("Minimum Image Amplitude (ph.)")
    p_mia_g.add_legend(loc="best")
    p_mia_g.save("candidate/sipm/mia_gamma.pdf")




if __name__ == '__main__':
    main()
