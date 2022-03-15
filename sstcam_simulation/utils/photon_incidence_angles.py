from sstcam_simulation.data import get_data
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d


class _PhotonIncidenceAngles:
    def __init__(self, array_key: str):
        path = get_data("datasheet/efficiency/angular_dist_1M_v2.npz")
        npz = np.load(path)

        self.incidence_pdf = npz[array_key]
        self.incidence_angle = npz["bins"]
        self.off_axis_angle = np.linspace(0, 5.4, 55)

        # Normalise PDF
        integral_bins = np.trapz(self.incidence_pdf, self.incidence_angle, axis=1)
        integral = np.trapz(integral_bins, self.off_axis_angle)
        # self.incidence_pdf /= integral

    def interpolate_at_off_axis_angle(self, off_axis_angle):
        f = interp1d(self.off_axis_angle, self.incidence_pdf, axis=0)
        return f(off_axis_angle)

    def plot(self, off_axis_angles=None, ax=None):
        if off_axis_angles is None:
            off_axis_angles = [0, 1, 2, 3, 4, 5]

        if ax is None:
            fig, ax = plt.subplots()

        for angle in off_axis_angles:
            array = self.interpolate_at_off_axis_angle(angle)
            ax.plot(self.incidence_angle, array, label=f"{angle}deg")

        ax.legend(title="Off-Axis Angle")
        ax.set_xlabel("Incidence Angle (deg)")
        ax.set_ylabel("PDF")


class PhotonIncidenceAnglesWindow(_PhotonIncidenceAngles):
    def __init__(self):
        super().__init__(array_key="arrays_window")


class PhotonIncidenceAnglesSiPM(_PhotonIncidenceAngles):
    def __init__(self):
        super().__init__(array_key="arrays_sipm")


if __name__ == '__main__':
    fig: plt.Figure = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    incidence_angles_sipm = PhotonIncidenceAnglesSiPM()
    incidence_angles_window = PhotonIncidenceAnglesWindow()

    incidence_angles_sipm.plot(ax=ax1)
    incidence_angles_window.plot(ax=ax2)

    ax1.set_title("SiPM Pixels")
    ax2.set_title("Flat Window")
    fig.savefig("photon_incidence_angles.pdf")
    # plt.show()
