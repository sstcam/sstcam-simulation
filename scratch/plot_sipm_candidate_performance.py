from CHECLabPy.plotting.setup import Plotter
import numpy as np
import pandas as pd

MINIMGAMP_GAMMA = 250
MINIMGAMP_PROTON = 480
BTEL1170_PDE = 0.2


def main():
    path = "sipm_candidate_performance.h5"
    with pd.HDFStore(path) as store:
        df = store['data']

    df = df.loc[df['fov_angle'] == 0]

    p_cherenkov_pde = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_cherenkov_pde.ax.plot(group['overvoltage'], group['camera_cherenkov_pde'], 'x-', label=candidate)
    p_cherenkov_pde.ax.set_xlabel("Overvoltage (V)")
    p_cherenkov_pde.ax.set_ylabel("Camera Cherenkov PDE")
    p_cherenkov_pde.add_legend(loc="best")
    p_cherenkov_pde.save("camera_cherenkov_pde.pdf")

    p_B_TEL_1170_pde = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_B_TEL_1170_pde.ax.plot(group['overvoltage'], group['B_TEL_1170_pde'], 'x-', label=candidate)
    p_B_TEL_1170_pde.ax.axhline(BTEL1170_PDE, ls='--', color='black', label="Requirement")
    p_B_TEL_1170_pde.ax.set_xlabel("Overvoltage (V)")
    p_B_TEL_1170_pde.ax.set_ylabel("B-TEL-1170 PDE")
    p_B_TEL_1170_pde.add_legend(loc="best")
    p_B_TEL_1170_pde.save("B_TEL_1170_pde.pdf")

    p_opct = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_opct.ax.plot(group['overvoltage'], group['opct'], 'x-', label=candidate)
    p_opct.ax.set_xlabel("Overvoltage (V)")
    p_opct.ax.set_ylabel("Optical Crosstalk")
    p_opct.add_legend(loc="best")
    p_opct.save("opct.pdf")

    p_nominal_nsb = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_nominal_nsb.ax.plot(group['overvoltage'], group['nominal_nsb_rate'], 'x-', label=candidate)
    p_nominal_nsb.ax.set_xlabel("Overvoltage (V)")
    p_nominal_nsb.ax.set_ylabel("Nominal NSB Rate (MHz)")
    p_nominal_nsb.add_legend(loc="best")
    p_nominal_nsb.save("nominal_nsb.pdf")

    p_mia = Plotter()
    for candidate, group in df.groupby("sipm_candidate"):
        p_mia.ax.plot(group['overvoltage'], group['minimum_image_amplitude'], 'x-', label=candidate)
    p_mia.ax.axhline(MINIMGAMP_GAMMA, ls='--', color='black', label="Requirement")
    p_mia.ax.set_xlabel("Overvoltage (V)")
    p_mia.ax.set_ylabel("Minimum Image Amplitude (ph.)")
    p_mia.add_legend(loc="best")
    p_mia.save("mia.pdf")


if __name__ == '__main__':
    main()
