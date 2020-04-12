from astropy import units as u
from astropy.coordinates import Angle
import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from ctapipe.image.toymodel import Gaussian
    from ctapipe.image.hillas import camera_to_shower_coordinates

__all__ = [
    "get_cherenkov_shower_image",
]


def get_cherenkov_shower_image(
    xpix, ypix, centroid_x, centroid_y, length, width, psi, time_gradient, time_intercept
):
    """
    Obtain the PDF and time images for a Cherenkov shower ellipse

    Uses the toymodel methods defined in ctapipe.

    Parameters
    ----------
    xpix : ndarray
        Pixel X coordinates. Unit: m
    ypix : ndarray
        Pixel Y coordinates. Unit: m
    centroid_x : float
        X coordinate for the center of the ellipse. Unit: m
    centroid_y : float
        Y coordinate for the center of the ellipse. Unit: m
    length : float
        Length of the ellipse. Unit: m
    width : float
        Width of the ellipse. Unit: m
    psi : float
        Rotation of the ellipse major axis from the X axis. Unit: degrees
    time_gradient : float
        Rate at which the time changes with distance along the shower axis
        Unit: ns / m
    time_intercept : float
        Pulse time at the shower centroid. Unit: ns

    Returns
    -------
    pdf : ndarray
        Probability density function of the Cherenkov shower ellipse amplitude
    time : ndarray
        Pulse time per pixel. Unit: ns
    """
    xpix = u.Quantity(xpix, u.m)
    ypix = u.Quantity(ypix, u.m)
    centroid_x = u.Quantity(centroid_x, u.m)
    centroid_y = u.Quantity(centroid_y, u.m)
    psi = Angle(psi, unit='deg')

    shower_image_pdf = Gaussian(
        x=centroid_x,
        y=centroid_y,
        length=u.Quantity(length, u.m),
        width=u.Quantity(width, u.m),
        psi=psi,
    ).pdf(xpix, ypix)

    # Normalise
    shower_image_pdf /= shower_image_pdf.sum()

    # TODO: replace when ctapipe 0.8 is released
    longitudinal = camera_to_shower_coordinates(
        xpix, ypix, centroid_x, centroid_y, psi
    )[0].to_value(u.m)
    time = longitudinal * time_gradient + time_intercept

    return shower_image_pdf, time
