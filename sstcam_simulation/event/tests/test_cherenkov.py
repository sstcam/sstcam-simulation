from sstcam_simulation.camera import Camera
from sstcam_simulation.event.cherenkov import get_cherenkov_shower_image
import numpy as np


def test_cherenkov():
    camera = Camera()
    pdf, time = get_cherenkov_shower_image(
        xpix=camera.pixel.x,
        ypix=camera.pixel.y,
        centroid_x=0,
        centroid_y=0,
        length=0.01,
        width=0.01,
        psi=0,
        time_gradient=1,
        time_intercept=20,
    )
    np.testing.assert_allclose(pdf.sum(), 1)
    np.testing.assert_allclose(np.polyfit(camera.pixel.x, time, 1), [1.0, 20.0])

    pdf, time = get_cherenkov_shower_image(
        xpix=camera.pixel.x,
        ypix=camera.pixel.y,
        centroid_x=0.1,
        centroid_y=0,
        length=0.1,
        width=0.03,
        psi=360,
        time_gradient=1,
        time_intercept=20,
    )
    np.testing.assert_allclose(pdf.sum(), 1)
    np.testing.assert_allclose(np.polyfit(camera.pixel.x, time, 1), [1.0, 19.9])

    pdf, time = get_cherenkov_shower_image(
        xpix=camera.pixel.x,
        ypix=camera.pixel.y,
        centroid_x=0,
        centroid_y=0,
        length=0.01,
        width=0.01,
        psi=90,
        time_gradient=1,
        time_intercept=20,
    )
    np.testing.assert_allclose(pdf.sum(), 1)
    np.testing.assert_allclose(np.polyfit(camera.pixel.y, time, 1), [1.0, 20.0])
