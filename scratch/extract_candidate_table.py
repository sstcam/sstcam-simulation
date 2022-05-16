from sstcam_simulation.utils.efficiency import CameraEfficiency
from sstcam_simulation.utils.sipm import SiPMOvervoltage
from sstcam_simulation.utils.window_durham_needle import (
    WindowDurhamNeedle,
    SSTWindowRun2,
    SSTWindowRun3,
    SSTWindowRun4,
    Prod4Window,
)
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

SIM_PATH = "/Users/Jason/Software/sstcam-simulation/optimisation_studies/overvoltage/performance.h5"


class PEInterpolator:
    def __init__(self, df, column):
        xcol, ycol, zcol, vcol = "opct", "nsb_rate", "mv_per_pe", column
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
    def gamma(cls, column):
        with pd.HDFStore(SIM_PATH, mode='r') as store:
            df = store['data']
            df = df.loc[df['shower_primary_id'] == 0]
        return cls(df, column)

    @classmethod
    def proton(cls, column):
        with pd.HDFStore(SIM_PATH, mode='r') as store:
            df = store['data']
            df = df.loc[df['shower_primary_id'] == 101]
        return cls(df, column)


def main():
    overvoltages = np.linspace(3, 7.5, 10)
    fov_angles = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.4]
    d_list = []

    sipm_candidates = dict(
        lct5_6mm_50um_epoxy=SiPMOvervoltage.lct5_6mm_50um_epoxy(),
        lct5_6mm_75um_epoxy=SiPMOvervoltage.lct5_6mm_75um_epoxy(),
        lvr3_6mm_50um_silicon=SiPMOvervoltage.lvr3_6mm_50um_silicon(),
        lvr3_6mm_50um_uncoated=SiPMOvervoltage.lvr3_6mm_50um_uncoated(),
        lvr3_6mm_75um_silicon=SiPMOvervoltage.lvr3_6mm_75um_silicon(),
        lvr3_6mm_75um_uncoated=SiPMOvervoltage.lvr3_6mm_75um_uncoated(),
    )

    window_candidates = dict(
        durham_needle=WindowDurhamNeedle(),
        run2=SSTWindowRun2(),
        run3=SSTWindowRun3(),
        run4=SSTWindowRun4(),
        prod4window=Prod4Window(),
    )

    mia_gamma_interp = PEInterpolator.gamma("minimum_image_amplitude")
    trigger_threshold_mean_interp = PEInterpolator.gamma("trigger_threshold_mean")

    for window_candidate, window_tool in tqdm(window_candidates.items()):
        for sipm_candidate, sipm_tool in tqdm(sipm_candidates.items()):
            for overvoltage in overvoltages:
                sipm_tool.overvoltage = overvoltage
                pde_at_450nm = sipm_tool.pde
                for fov_angle in fov_angles:
                    eff = CameraEfficiency.from_sstcam(
                        fov_angle=fov_angle,
                        pde_at_450nm=pde_at_450nm,
                        window=window_tool,
                    )

                    opct = sipm_tool.opct
                    gain = sipm_tool.gain
                    camera_cherenkov_pde = eff.camera_cherenkov_pde
                    mv_per_pe = 4
                    nominal_nsb_rate = eff.nominal_nsb_rate.to_value("MHz")
                    mia_pe = mia_gamma_interp(opct, nominal_nsb_rate, mv_per_pe)[0]
                    mia_photons = mia_pe / camera_cherenkov_pde
                    trig_thresh_pe = trigger_threshold_mean_interp(opct, nominal_nsb_rate, mv_per_pe)[0]
                    trig_thresh_photons = trig_thresh_pe / camera_cherenkov_pde

                    d_list.append(dict(
                        sipm_candidate=sipm_candidate,
                        window_candidate=window_candidate,
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
                        trigger_threshold_pe=trig_thresh_pe,
                        trigger_threshold_photons=trig_thresh_photons,
                    ))

    df = pd.DataFrame(d_list)
    with pd.HDFStore("candidate/performance.h5") as store:
        store['data'] = df


if __name__ == '__main__':
    main()

    # from sstcam_simulation.utils.sipm.pde import PDEvsWavelength
    # from astropy import units as u
    #
    # sipm_tool = SiPMOvervoltage.lvr3_6mm_50um_uncoated()
    # sipm_tool.overvoltage = 5.9
    # pde_at_450nm = sipm_tool.pde
    # pde_vs_wavelength = PDEvsWavelength.LVR3_75um_6mm()
    # pde_vs_wavelength.scale(u.Quantity(450, u.nm), pde_at_450nm)
    # weighted_pde = pde_vs_wavelength.weight_by_incidence_angle(
    #     off_axis_angle=0
    # )
    # np.savetxt("PDE.txt", np.column_stack([pde_vs_wavelength.wavelength, weighted_pde]), fmt=("%.3f", "%.5f"), delimiter="\t")
