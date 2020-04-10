from CHECLabPy.plotting.camera import CameraImage as CLPCameraImage
from ..camera import PixelMapping, SuperpixelMapping


class CameraImage(CLPCameraImage):
    @classmethod
    def from_mapping(cls, mapping, **kwargs):
        """
        Generate the class using a sstcam-simulation mapping object

        Parameters
        ----------
        mapping : PixelMapping or SuperpixelMapping
        kwargs
            Arguments passed to `CHECLabPy.plotting.setup.Plotter`

        Returns
        -------
        CameraImage
        """
        return cls(mapping.x, mapping.y, mapping.size, **kwargs)
