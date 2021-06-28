from sstcam_simulation.data import get_data
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
        wavelength = np.arange(200, 1000)
        self._df_unscaled = interpolate_at(df, wavelength) / 100
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

    @classmethod
    def LVR3_75um_6mm(cls):
        path = get_data("datasheet/efficiency/pde_LVR3_75um_6mm.csv")
        df = pd.read_csv(path, index_col="Wavelength")
        return cls(df=df)
        # embed()


if __name__ == '__main__':
    pde = PDEvsWavelength.LVR3_75um_6mm()
    pde.scale(u.Quantity(400, u.nm), 0.7)
    pde.plot()
    plt.plot(pde.df.index, pde.interpolate(75), label="interp")
    plt.show()
