from sstcam_simulation.data import get_data
from sstcam_simulation.utils.photon_incidence_angles import PhotonIncidenceAnglesSiPM
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from astropy import units as u
from scipy.interpolate import interp1d
from IPython import embed


def interpolate_at(df, new_idxs):
    new_idxs = pd.Index(new_idxs)
    df = df.drop_duplicates().dropna()
    df = df.reindex(df.index.append(new_idxs).unique())
    df = df.sort_index()
    df = df.interpolate()
    return df.loc[new_idxs]


class PDEvsWavelength:
    def __init__(self, df):
        self.wavelength = np.arange(200, 1000)
        self._df_unscaled = interpolate_at(df, self.wavelength) / 100
        self._pde_scale = 1

    @u.quantity_input
    def scale(self, wavelength: u.nm, pde_at_wavelength):
        wavelength_arr = self._df_unscaled.index << u.nm
        pde_arr = self._df_unscaled["PDE at 0 deg"].values
        self._pde_scale = pde_at_wavelength / np.interp(wavelength, wavelength_arr, pde_arr)

    @property
    def df(self):
        return self._df_unscaled * self._pde_scale

    def plot(self, alpha=1):
        d = {
            "0deg":  'PDE at 0 deg',
            "20deg":  'PDE at 20 deg',
            "40deg":  'PDE at 40 deg',
            "60deg":  'PDE at 60 deg',
            "70deg":  'PDE at 70 deg',
            "80deg":  'PDE at 80 deg',
            "-20deg":  'PDE at -20 deg',
            "-40deg":  'PDE at -40 deg',
            "-60deg":  'PDE at -60 deg',
            "-70deg":  'PDE at -70 deg',
            "-80deg":  'PDE at -80 deg',
        }

        fig, ax = plt.subplots()
        for label, col in d.items():
            color = ax._get_lines.get_next_color()
            ax.plot(self.df.index, self.df[col], "-", alpha=alpha, color=color, label=label)

        from sstcam_simulation.utils.efficiency import CameraEfficiency
        c = CameraEfficiency.from_prod4()
        ax.plot(c.wavelength, c.pde, alpha=alpha, color='black', label="prod4")
        ax.legend()
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("PDE")

    def interpolate(self, angle):
        angles = np.array([0, 20, 40, 60, 70, 80])
        arrays = np.vstack([
            self.df['PDE at 0 deg'],
            self.df['PDE at 20 deg'],
            self.df['PDE at 40 deg'],
            self.df['PDE at 60 deg'],
            self.df['PDE at 70 deg'],
            self.df['PDE at 80 deg'],
        ])
        f = interp1d(angles, arrays, axis=0)
        return f(angle)

    def weight_by_incidence_angle(self, off_axis_angle: float):
        incidence_angles = PhotonIncidenceAnglesSiPM()

        # Interpolate the input item (window/PDE) to matching incidence angles
        incidence_angle = incidence_angles.incidence_angle
        in_range = incidence_angle <= 80
        incidence_angle = incidence_angle[in_range]
        input_2d = self.interpolate(incidence_angle)

        # Extract the incidence array for the specified off-axis angle
        incidence_pdf = incidence_angles.interpolate_at_off_axis_angle(off_axis_angle)
        incidence_pdf = incidence_pdf[in_range]

        return np.trapz(  # TODO: trapz or sum?
            input_2d * incidence_pdf[:, None], incidence_angle, axis=0
        ) / np.trapz(incidence_pdf, incidence_angle)

    @classmethod
    def LVR3_75um_6mm(cls):
        path = get_data("datasheet/efficiency/pde_LVR3_75um_6mm.csv")
        df = pd.read_csv(path, index_col="Wavelength")
        return cls(df=df)
        # embed()


def read_lct5_resin_coated():
    path = "/Users/Jason/Software/sstcam-simulation/sstcam_simulation/data/datasheet/efficiency/pde_LCT5_50um_resin.csv"
    wavelength, pde = np.loadtxt(path, delimiter=', ', unpack=True)
    wavelength_interp = np.arange(200, 1000)
    f = interp1d(wavelength, pde, fill_value="extrapolate")
    pde_interp = f(wavelength_interp)
    pde_interp[pde_interp < 0] = 0
    return pde_interp * 0.01


def read_prototype():
    path = "/Users/Jason/Software/sstcam-simulation/sstcam_simulation/data/datasheet/efficiency/pde_prototype.csv"
    wavelength, pde = np.loadtxt(path, delimiter=', ', unpack=True)
    wavelength_interp = np.arange(200, 1000)
    f = interp1d(wavelength, pde, fill_value="extrapolate")
    pde_interp = f(wavelength_interp)
    pde_interp[pde_interp < 0] = 0
    return pde_interp


if __name__ == '__main__':
    off_axis_angles = [0, 2, 4]
    pdes = [
        PDEvsWavelength.LVR3_75um_6mm(),
    ]

    for pde in pdes:
        fig, ax = plt.subplots()

        x = pde.wavelength
        y = pde.interpolate(0)
        ax.plot(x, y, '-', alpha=0.1)

        for angle in off_axis_angles:
            x = pde.wavelength
            y = pde.weight_by_incidence_angle(angle)
            ax.plot(x, y, label=f"{angle}deg")

        ax.legend(title="Off-Axis Angle")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmission")
        ax.set_title(pde.__class__.__name__)
        fig.savefig(f"{pde.__class__.__name__}.pdf")

    # w = WindowDurhamNeedle()
    # w.plot_measured()
    # plt.legend()
    # plt.show()
