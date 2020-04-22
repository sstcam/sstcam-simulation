from CHECLabPy.plotting.camera import CameraImage as CLPCameraImage
from ..camera.mapping import CameraCoordinates


class CameraImage(CLPCameraImage):
    @classmethod
    def from_coordinates(cls, coordinates, **kwargs):
        """
        Generate the class using a sstcam-simulation CameraCoordinates object

        Parameters
        ----------
        coordinates : CameraCoordinates
        kwargs
            Arguments passed to `CHECLabPy.plotting.setup.Plotter`

        Returns
        -------
        CameraImage
        """
        return cls(coordinates.x, coordinates.y, coordinates.size, **kwargs)
