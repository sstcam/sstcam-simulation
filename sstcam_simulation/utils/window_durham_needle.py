from sstcam_simulation.data import get_data
from sstcam_simulation.utils.photon_incidence_angles import PhotonIncidenceAnglesWindow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class Window:
    def __init__(self, incidence_angles, transmission):
        self.incidence_angles = incidence_angles
        self.transmission = transmission
        assert self.incidence_angles.shape[0] == self.transmission.shape[0]

    def interpolate_at_incidence_angle(self, angle):
        f = interp1d(self.incidence_angles, self.transmission, axis=0)
        return f(angle)

    def weight_by_incidence_angle(self, off_axis_angle: float):
        incidence_angles = PhotonIncidenceAnglesWindow()

        # Interpolate the input item (window/PDE) to matching incidence angles
        incidence_angle = incidence_angles.incidence_angle
        in_range = incidence_angle <= 60
        incidence_angle = incidence_angle[in_range]
        input_2d = self.interpolate_at_incidence_angle(incidence_angle)

        # Extract the incidence array for the specified off-axis angle
        incidence_pdf = incidence_angles.interpolate_at_off_axis_angle(off_axis_angle)
        incidence_pdf = incidence_pdf[in_range]

        return np.trapz(  # TODO: trapz or sum?
            input_2d * incidence_pdf[:, None], incidence_angle, axis=0
        ) / np.trapz(incidence_pdf, incidence_angle)


class WindowDurhamNeedle(Window):
    def __init__(self):
        path = get_data("datasheet/efficiency/window_durham_needle.csv")
        self.df = pd.read_csv(path)
        self.df = self.df.set_index("wavelength") / 100

        self._df_no_interp = self.df.copy()

        self.df = self.df.interpolate(method='cubic', limit_area="inside")
        # self.df.iloc[:20, :] = self.df.iloc[:20, :].fillna(0)
        # self.df.iloc[-100:, :] = self.df.iloc[-100:, :].fillna(0.35)
        # self.df = self.df.interpolate(method="cubic")
        self.df = self.df.interpolate(limit_area="outside", limit_direction="both")

        angles = np.array([0, 20, 45, 50, 60])
        arrays = np.vstack([
            self.df['0_measured'],
            self.df['20_measured'],
            self.df['45_measured'],
            self.df['50_measured'],
            self.df['60_measured'],
        ])
        super().__init__(incidence_angles=angles, transmission=arrays)

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


class _SSTWindow(Window):
    def __init__(self, path):
        df = pd.read_csv(path)
        self.df = df.set_index("Wavelength (nm)") / 100
        self.df = self.df.reindex(np.arange(200, 1000))

        def interpolate(x, y):
            mask = ~np.isnan(y)
            x = x[mask]
            y = y[mask]
            from scipy.interpolate import interp1d
            f = interp1d(x, y, fill_value="extrapolate")
            x_new = np.arange(200, 1000)
            y_new = f(x_new)
            y_new[x_new < 300] = 0
            return y_new

        self.df['M0'] = interpolate(self.df.index, self.df["M0"])
        self.df['M20'] = interpolate(self.df.index, self.df["M20"])
        self.df['M45'] = interpolate(self.df.index, self.df["M45"])
        self.df['M50'] = interpolate(self.df.index, self.df["M50"])
        self.df['M60'] = interpolate(self.df.index, self.df["M60"])

        angles = np.array([0, 20, 45, 50, 60])
        arrays = np.vstack([
            self.df['M0'],
            self.df['M20'],
            self.df['M45'],
            self.df['M50'],
            self.df['M60'],
        ])
        super().__init__(incidence_angles=angles, transmission=arrays)

class SSTWindowRun2(_SSTWindow):
    def __init__(self):
        path = get_data("datasheet/efficiency/SST-Window-Run-2-RW.csv")
        super().__init__(path=path)


class SSTWindowRun3(_SSTWindow):
    def __init__(self):
        path = get_data("datasheet/efficiency/SST-Window-Run-3-RW.csv")
        super().__init__(path=path)


class SSTWindowRun4(_SSTWindow):
    def __init__(self):
        path = get_data("datasheet/efficiency/SST-Window-Run-4-RW.csv")
        super().__init__(path=path)


class DurhamNeedleWindowD2208Prod1FilterAR(Window):
    def __init__(self):
        path = get_data("datasheet/efficiency/Durham_Needle_Window_d2208_Prod1_FilterAR.csv")
        df = pd.read_csv(path)
        self.df = df.set_index("Wavelength (nm)") / 100
        self.df = self.df.reindex(np.arange(200, 1000))
        self.df = self.df.fillna(0)

        angles = np.array([0, 20, 45, 50, 60])
        arrays = np.vstack([
            self.df['M0'],
            self.df['M20'],
            self.df['M45'],
            self.df['M50'],
            self.df['M60'],
        ])
        super().__init__(incidence_angles=angles, transmission=arrays)


class Prod4Window(Window):
    def __init__(self):
        PROD4_PATH_WINDOW = get_data("datasheet/efficiency/window_prod4.csv")
        df_window = pd.read_csv(PROD4_PATH_WINDOW)
        window_transmissivity = df_window['window_transmissivity'].values
        mask = df_window["wavelength"] < 280
        window_transmissivity[mask] = 0
        super().__init__(np.array([0]), window_transmissivity[None, :])

    def weight_by_incidence_angle(self, off_axis_angle: float):
        return self.transmission[0]


class AkiraWindow(Window):
    def __init__(self):
        path = get_data("datasheet/efficiency/akira_window.csv")
        df = pd.read_csv(path, names=["wavelength", "transmission"])
        xs = df["wavelength"].values
        ys = df["transmission"].values / 100

        f = interp1d(xs, ys, fill_value='extrapolate')
        xnew = np.arange(200, 1000)
        ynew = f(xnew)
        super().__init__(np.array([0]), ynew[None, :])

    def weight_by_incidence_angle(self, off_axis_angle: float):
        return self.transmission[0]


if __name__ == '__main__':
    fig, ax = plt.subplots()

    window = AkiraWindow()
    x = np.arange(200, 1000)
    y = window.weight_by_incidence_angle(0)
    ax.plot(x, y)

    # window = DurhamNeedleWindowD2208Prod1FilterAR()
    # x = window.df.index.values
    # y_at_45 = window.interpolate_at_incidence_angle(45)
    # ax.plot(x, y_at_45, label=f"@45deg incidence")
    # for offaxis in range(6):
    #     y = window.weight_by_incidence_angle(offaxis)
    #     ax.plot(x, y, label=f"offaxis={offaxis}")

    # y0 = window.weight_by_incidence_angle(0)
    # y1 =
    # ax.plot(x, y, label=f"sstcam-weighted")

    # mask = ~np.isnan(y)
    # x = x[mask]
    # y = y[mask]
    # np.savetxt("transmission.txt", np.column_stack([x, y]), fmt=("%.0f", "%.5f"), delimiter="\t")


    # y = window_sstcam.interpolate_at_incidence_angle(45)
    # ax.plot(x, y, label=f"sstcam-45deg")
    # window_sstcam = Prod4Window()
    # y = window_sstcam.weight_by_incidence_angle(0)
    # ax.plot(x, y, label=f"prod4-weighted")

    ax.legend()
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission")
    ax.set_ylim([0, 1])
    ax.set_title(window.__class__.__name__)
    fig.savefig(f"window_comparison.pdf")
