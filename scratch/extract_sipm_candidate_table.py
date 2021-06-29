from sstcam_simulation.utils.efficiency import CameraEfficiency
from sstcam_simulation.utils.sipm import SiPMOvervoltage
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

SIM_PATH = "/Users/Jason/Software/sstcam-simulation/optimisation_studies/overvoltage/performance.h5"


class MinimumImageAmplitudeInterpolator:
    def __init__(self, df):
        xcol, ycol, zcol, vcol = "opct", "nsb_rate", "mv_per_pe", "minimum_image_amplitude"
        df = df.sort_values(by=[xcol, ycol, zcol])
        xvals = df[xcol].unique()
        yvals = df[ycol].unique()
        zvals = df[zcol].unique()
        vvals = df[vcol].values.reshape(len(xvals), len(yvals), len(zvals))

        self.f = RegularGridInterpolator((xvals, yvals, zvals), vvals)

    def __call__(self, opct, nsb, mv_per_pe):
        pts = np.column_stack([opct, nsb, mv_per_pe])
        return self.f(pts)

    @classmethod
    def gamma(cls):
        with pd.HDFStore(SIM_PATH, mode='r') as store:
            df = store['data']
            df = df.loc[df['shower_primary_id'] == 0]
        return cls(df)

    @classmethod
    def proton(cls):
        with pd.HDFStore(SIM_PATH, mode='r') as store:
            df = store['data']
            df = df.loc[df['shower_primary_id'] == 101]
        return cls(df)


def main():
    overvoltages = np.linspace(3, 7.5, 10)
    fov_angles = np.linspace(0, 10, 11)
    d_list = []

    sipm_candidates = dict(
        lct5_6mm_50um_epoxy=SiPMOvervoltage.lct5_6mm_50um_epoxy(),
        lct5_6mm_75um_epoxy=SiPMOvervoltage.lct5_6mm_75um_epoxy(),
        lvr3_6mm_50um_silicon=SiPMOvervoltage.lvr3_6mm_50um_silicon(),
        lvr3_6mm_50um_uncoated=SiPMOvervoltage.lvr3_6mm_50um_uncoated(),
        lvr3_6mm_75um_silicon=SiPMOvervoltage.lvr3_6mm_75um_silicon(),
        lvr3_6mm_75um_uncoated=SiPMOvervoltage.lvr3_6mm_75um_uncoated(),
    )

    mia_gamma_interp = MinimumImageAmplitudeInterpolator.gamma()

    for sipm_candidate, sipm_tool in tqdm(sipm_candidates.items()):
        for overvoltage in overvoltages:
            sipm_tool.overvoltage = overvoltage
            pde_at_450nm = sipm_tool.pde
            for fov_angle in fov_angles:
                eff = CameraEfficiency.from_sstcam(fov_angle=fov_angle, pde_at_450nm=pde_at_450nm)

                opct = sipm_tool.opct
                gain = sipm_tool.gain
                camera_cherenkov_pde = eff.camera_cherenkov_pde
                mv_per_pe = 4
                nominal_nsb_rate = eff.nominal_nsb_rate.to_value("MHz")
                mia_pe = mia_gamma_interp(opct, nominal_nsb_rate, mv_per_pe)[0]
                mia_photons = mia_pe / camera_cherenkov_pde

                d_list.append(dict(
                    sipm_candidate=sipm_candidate,
                    overvoltage=overvoltage,
                    pde_at_450nm=pde_at_450nm,
                    opct=opct,
                    gain=gain,
                    fov_angle=fov_angle,
                    B_TEL_1170_pde=eff.B_TEL_1170_pde,
                    camera_cherenkov_pde=camera_cherenkov_pde,
                    telescope_cherenkov_pde=eff.telescope_cherenkov_pde,
                    camera_nsb_pde=eff.camera_nsb_pde.to_value(),
                    telescope_nsb_pde=eff.telescope_nsb_pde.to_value(),
                    camera_signal_to_noise=eff.camera_signal_to_noise.to_value(),
                    telescope_signal_to_noise=eff.telescope_signal_to_noise.to_value(),
                    nominal_nsb_rate=nominal_nsb_rate,
                    maximum_nsb_rate=eff.maximum_nsb_rate.to_value("MHz"),
                    n_cherenkov_photoelectrons=eff.n_cherenkov_photoelectrons,
                    minimum_image_amplitude=mia_photons,
                ))

    df = pd.DataFrame(d_list)
    with pd.HDFStore("sipm_candidate_performance.h5") as store:
        store['data'] = df


if __name__ == '__main__':
    main()