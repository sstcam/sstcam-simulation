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

    df = df.loc[df["sipm_candidate"] == "lvr3_6mm_50um_uncoated"]
    df = df.loc[df["overvoltage"] == 6]

    prod4_eff = CameraEfficiency.from_prod4()

    p_cherenkov_pde = Plotter()
    for candidate, group in df.groupby("window_candidate"):
        p_cherenkov_pde.ax.plot(group['fov_angle'], group['camera_cherenkov_pde'], 'x-', label=candidate)
    p_cherenkov_pde.ax.axhline(prod4_eff.camera_cherenkov_pde, ls='--', color='blue', label="Prod4")
    p_cherenkov_pde.ax.set_xlabel("Off-Axis Angle (deg)")
    p_cherenkov_pde.ax.set_ylabel("Camera Cherenkov PDE")
    p_cherenkov_pde.add_legend(loc="best")
    p_cherenkov_pde.save("candidate/window/camera_cherenkov_pde.pdf")

    p_B_TEL_1170_pde = Plotter()
    for candidate, group in df.groupby("window_candidate"):
        p_B_TEL_1170_pde.ax.plot(group['fov_angle'], group['B_TEL_1170_pde'], 'x-', label=candidate)
    p_B_TEL_1170_pde.ax.axhline(BTEL1170_PDE, ls='--', color='black', label="Requirement")
    p_B_TEL_1170_pde.ax.axhline(prod4_eff.B_TEL_1170_pde, ls='--', color='blue', label="Prod4")
    p_B_TEL_1170_pde.ax.set_xlabel("Off-Axis Angle (deg)")
    p_B_TEL_1170_pde.ax.set_ylabel("B-TEL-1170 PDE")
    p_B_TEL_1170_pde.add_legend(loc="best")
    p_B_TEL_1170_pde.save("candidate/window/B_TEL_1170_pde.pdf")

    p_sn = Plotter()
    for candidate, group in df.groupby("window_candidate"):
        p_sn.ax.plot(group['fov_angle'], group['telescope_signal_to_noise'], 'x-', label=candidate)
    p_sn.ax.axhline(BTEL0090_SNR, ls='--', color='black', label="Requirement")
    p_sn.ax.axhline(prod4_eff.telescope_signal_to_noise, ls='--', color='blue', label="Prod4")
    p_sn.ax.set_xlabel("Off-Axis Angle (deg)")
    p_sn.ax.set_ylabel("B-TEL-0090 SNR")
    p_sn.add_legend(loc="best")
    p_sn.save("candidate/window/B-TEL-0090_snr.pdf")


if __name__ == '__main__':
    main()
