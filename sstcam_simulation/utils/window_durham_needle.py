from sstcam_simulation.data import get_data
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from IPython import embed


class WindowDurhamNeedle:
    def __init__(self, path=get_data("datasheet/efficiency/window_durham_needle.csv")):
        self.df = pd.read_csv(path)
        self.df = self.df.set_index("wavelength") / 100

        # embed()
        self._df_no_interp = self.df.copy()

        self.df = self.df.interpolate(method='cubic', limit_area="inside")
        # self.df.iloc[:20, :] = self.df.iloc[:20, :].fillna(0)
        # self.df.iloc[-100:, :] = self.df.iloc[-100:, :].fillna(0.35)
        # self.df = self.df.interpolate(method="cubic")
        self.df = self.df.interpolate(limit_area="outside", limit_direction="both")

    def plot_measured(self, alpha=1):
        df = self.df
        df_no_interp = self._df_no_interp

        d = {
            "0deg":  '0_measured',
            "20deg": '20_measured',
            "45deg": '45_measured',
            "50deg": '50_measured',
            "60deg": '60_measured',
        }

        fig, ax = plt.subplots()
        for label, col in d.items():
            color = ax._get_lines.get_next_color()
            ax.plot(df.index, df[col], ":", alpha=alpha, color=color)
            ax.plot(df_no_interp.index, df_no_interp[col], "-", alpha=alpha, color=color, label=label)

        from sstcam_simulation.utils.efficiency import CameraEfficiency
        c = CameraEfficiency.from_prod4()
        ax.plot(c.wavelength, c.window_transmissivity, alpha=alpha, color='black', label="prod4")
        ax.legend()
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Transmissivity")

    def interpolate(self, angle):
        angles = np.array([0, 20, 45, 50, 60])
        arrays = np.vstack([
            self.df['0_measured'],
            self.df['20_measured'],
            self.df['45_measured'],
            self.df['50_measured'],
            self.df['60_measured'],
        ])
        f = interp1d(angles, arrays, axis=0)
        return f(angle)


if __name__ == '__main__':
    w = WindowDurhamNeedle()
    w.plot_measured()
    plt.legend()
    plt.show()
