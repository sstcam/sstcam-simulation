from sstcam_simulation.utils.sipm import SiPMOvervoltage
import numpy as np
import pytest


@pytest.fixture(scope='module')
def sipm_data():
    overvoltage = np.linspace(1, 5, 100)
    gain = overvoltage * 200
    opct = overvoltage * 0.1 - 0.05
    pde = overvoltage * 0.12 + 0.3
    return overvoltage, gain, opct, pde


def test_from_csv(sipm_data, tmp_path):
    path = str(tmp_path / "test.txt")
    np.savetxt(path, np.column_stack([*sipm_data]))
    sipm = SiPMOvervoltage.from_csv(path)
    np.testing.assert_allclose(sipm._overvoltage_array, sipm_data[0])
    np.testing.assert_allclose(sipm._gain_array, sipm_data[1])
    np.testing.assert_allclose(sipm._opct_array, sipm_data[2])
    np.testing.assert_allclose(sipm._pde_array, sipm_data[3])


def test_setters(sipm_data):
    sipm = SiPMOvervoltage(*sipm_data)
    sipm.overvoltage = 3.123
    overvoltage = sipm.overvoltage
    gain = sipm.gain
    opct = sipm.opct
    pde = sipm.pde
    np.testing.assert_allclose(sipm.overvoltage, 3.123)

    sipm.overvoltage = 4.1
    np.testing.assert_allclose(sipm.overvoltage, 4.1)
    assert sipm.gain > gain
    assert sipm.opct > opct
    assert sipm.pde > pde

    sipm.overvoltage = 4.1
    np.testing.assert_allclose(sipm.overvoltage, 4.1)
    sipm.gain = gain
    np.testing.assert_allclose(sipm.overvoltage, overvoltage)

    sipm.overvoltage = 4.1
    np.testing.assert_allclose(sipm.overvoltage, 4.1)
    sipm.opct = opct
    np.testing.assert_allclose(sipm.overvoltage, overvoltage)

    sipm.overvoltage = 4.1
    np.testing.assert_allclose(sipm.overvoltage, 4.1)
    sipm.pde = pde
    np.testing.assert_allclose(sipm.overvoltage, overvoltage)


def test_scale_gain(sipm_data):
    sipm = SiPMOvervoltage(*sipm_data)
    sipm.scale_gain(3, 123)
    sipm.overvoltage = 3
    np.testing.assert_allclose(sipm.gain, 123)
    sipm.scale_gain(3, 2)
    np.testing.assert_allclose(sipm.gain, 2)


def test_scale_opct(sipm_data):
    sipm = SiPMOvervoltage(*sipm_data)
    sipm.scale_opct(3, 123)
    sipm.overvoltage = 3
    np.testing.assert_allclose(sipm.opct, 123)
    sipm.scale_opct(3, 2)
    np.testing.assert_allclose(sipm.opct, 2)
