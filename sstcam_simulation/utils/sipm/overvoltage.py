import numpy as np
from numpy.polynomial.polynomial import polyfit, polyval
from scipy.interpolate import interp1d
from sstcam_simulation.data import get_data
from matplotlib import pyplot as plt


class SiPMOvervoltage:
    def __init__(self, overvoltage, gain, opct, pde):
        """
        SiPM parameters and their relation to overvoltage
        Input arrays must all have the same size

        SiPM datasheets can be read using the `from_csv` classmethod.
        The CSVs can be downloaded from the Nextcloud at:
        https://pcloud.mpi-hd.mpg.de/index.php/f/259852

        Parameters
        ----------
        overvoltage : ndarray
        gain : ndarray
        opct : ndarray
        pde : ndarray
        """
        self._overvoltage_array = overvoltage
        self._gain_array = gain
        self._opct_array = opct
        self._pde_array = pde

        self.overvoltage = (overvoltage.max() + overvoltage.min()) / 2

    @classmethod
    def from_csv(cls, path):
        return cls(*np.loadtxt(path, unpack=True))

    @classmethod
    def lct5_6mm_50um_epoxy(cls):
        path = get_data("datasheet/ov_lct5_6mm_50um_epoxy.txt")
        return SiPMOvervoltage.from_csv(path)

    @classmethod
    def lct5_6mm_75um_epoxy(cls):
        path = get_data("datasheet/ov_lct5_6mm_75um_epoxy.txt")
        return SiPMOvervoltage.from_csv(path)

    @classmethod
    def lvr3_6mm_50um_silicon(cls):
        path = get_data("datasheet/ov_lvr3_6mm_50um_silicon.txt")
        return SiPMOvervoltage.from_csv(path)

    @classmethod
    def lvr3_6mm_50um_uncoated(cls):
        path = get_data("datasheet/ov_lvr3_6mm_50um_uncoated.txt")
        return SiPMOvervoltage.from_csv(path)

    @classmethod
    def lvr3_6mm_75um_silicon(cls):
        path = get_data("datasheet/ov_lvr3_6mm_75um_silicon.txt")
        return SiPMOvervoltage.from_csv(path)

    @classmethod
    def lvr3_6mm_75um_uncoated(cls):
        path = get_data("datasheet/ov_lvr3_6mm_75um_uncoated.txt")
        return SiPMOvervoltage.from_csv(path)

    @property
    def overvoltage(self):
        return self._overvoltage

    @property
    def gain(self):
        return self._gain

    @property
    def opct(self):
        return self._opct

    @property
    def pde(self):
        return self._pde

    @overvoltage.setter
    def overvoltage(self, overvoltage):
        self._overvoltage = overvoltage
        self._gain = np.interp(overvoltage, self._overvoltage_array, self._gain_array)
        self._opct = np.interp(overvoltage, self._overvoltage_array, self._opct_array)
        self._pde = np.interp(overvoltage, self._overvoltage_array, self._pde_array)

    @gain.setter
    def gain(self, gain):
        self.overvoltage = np.interp(gain, self._gain_array, self._overvoltage_array)

    @opct.setter
    def opct(self, opct):
        self.overvoltage = np.interp(opct, self._opct_array, self._overvoltage_array)

    @pde.setter
    def pde(self, pde):
        self.overvoltage = np.interp(pde, self._pde_array, self._overvoltage_array)

    def scale_gain(self, overvoltage, gain_at_overvoltage):
        current = np.interp(overvoltage, self._overvoltage_array, self._gain_array)
        scale = gain_at_overvoltage / current
        self._gain_array = self._gain_array * scale
        self._gain = self._gain * scale

    def scale_opct(self, overvoltage, opct_at_voltage):
        current = np.interp(overvoltage, self._overvoltage_array, self._opct_array)
        scale = opct_at_voltage / current
        self._opct_array = self._opct_array * scale
        self._opct = self._opct * scale


def _create_from_extracted_data(path_gain, path_opct, path_pde):
    x_gain, y_gain = np.loadtxt(path_gain, unpack=True, delimiter=',')
    x_opct, y_opct = np.loadtxt(path_opct, unpack=True, delimiter=',')
    x_pde, y_pde = np.loadtxt(path_pde, unpack=True, delimiter=',')

    # overvoltage_min = min(x_gain[0], x_opct[0], x_pde[0])
    # overvoltage_max = max(x_gain[-1], x_opct[-1], x_pde[-1])
    overvoltage_min = x_pde.min()
    overvoltage_max = x_pde.max()
    overvoltage = np.linspace(overvoltage_min, overvoltage_max, 100)

    c_gain = polyfit(x_gain, y_gain, [1])
    gain = polyval(overvoltage, c_gain)

    c_opct = polyfit(x_opct, y_opct, 1)
    opct = polyval(overvoltage, c_opct)

    f = interp1d(x_pde, y_pde, bounds_error=False, fill_value='extrapolate')
    pde = f(overvoltage)

    data = np.column_stack([overvoltage, gain, opct, pde])
    np.savetxt("sipm.txt", data, fmt='%4f')

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(x_gain, y_gain, '.', color='red', alpha=0.2)
    ax2.plot(x_opct, y_opct, '.', color='blue', alpha=0.2)
    ax2.plot(x_pde, y_pde, '.', color='green', alpha=0.2)
    ln1 = ax1.plot(overvoltage, gain, color='red', label="Gain")
    ln2 = ax2.plot(overvoltage, opct, color='blue', label="OPCT")
    ln3 = ax2.plot(overvoltage, pde, color='green', label="PDE")
    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='best')
    ax1.set_xlabel("Overvoltage (V)")
    ax1.set_ylabel("Gain")
    ax2.set_ylabel("PDE & OPCT")

    fig.savefig("sipm.pdf")


if __name__ == '__main__':
    path_gain = "/Users/Jason/Downloads/Gain.csv"
    path_opct = "/Users/Jason/Downloads/OPCT.csv"
    path_pde = "/Users/Jason/Downloads/PDE.csv"
    _create_from_extracted_data(path_gain, path_opct, path_pde)
