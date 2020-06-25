from ..data import get_data
from dataclasses import dataclass
import numpy as np

__all__ = [
    "get_square_grid_neighbours",
    "CameraCoordinates",
    "SSTCameraMapping",
]


def get_square_grid_neighbours(row, column, diagonal):
    """
    Obtain the neighbours for a position on a square grid

    Parameters
    ----------
    row : ndarray
        Integer row of all squares on the grid
    column : ndarray
        Integer column of all squares on the grid
    diagonal : bool
        Include diagonals as a neighbour?

    Returns
    -------
    neighbours : ndarray
        2D array listing all neighbour combinations (no duplicates)
        Shape: (n_combinations, 2)
    """
    row_sep = np.abs(row - row[:, None])
    col_sep = np.abs(column - column[:, None])
    neighbours = np.array(np.where(
        (row_sep <= 1) & (col_sep <= 1) &  # Include all neighbouring pixels
        (~((row_sep == 0) & (col_sep == 0))) &  # Remove self
        (~((row_sep == 1) & (col_sep == 1)) | diagonal)  # Remove diagonals
    )).T

    # Remove duplicate pairs
    if neighbours.size > 0:
        neighbours = np.unique(np.sort(neighbours, axis=1), axis=0)

    return neighbours


@dataclass(frozen=True)
class CameraCoordinates:
    """
    Container for coordinates relating to any aspect of the camera focal plane
    (pixels, superpixels, ...)
    """
    i: np.ndarray  # Pixel index
    x: np.ndarray  # X coordinate. Unit: meters
    y: np.ndarray  # Y coordinate. Unit: meters
    row: np.ndarray  # Square grid row
    column: np.ndarray  # Square grid column
    neighbours: np.ndarray  # Unique neighbour pairs. Shape: (n_neighbours, 2)
    size: float  # Square vertex size. Unit: meters


class SSTCameraMapping:
    def __init__(self, n_pixels=None):
        """
        Container for the camera coordinates of the SST Camera, and the mapping
        between different coordinate sets (e.g. pixel to superpixel)

        Parameters
        ----------
        n_pixels : int
            Number of pixels in the simulated camera

        Attributes
        ----------
        self.n_pixels : int
            Number of pixels in the simulated camera
        self.pixel : CameraCoordinate
            Container for the pixel coordinates
        self.pixel_to_superpixel : ndarray
            Superpixel index for each pixel. Shape: (n_pixels)
        self.n_superpixels : int
            Number of superpixels in the simulated camera
        self.superpixel : CameraCoordinate
            Container for the superpixel coordinates
        """
        mapping_path = get_data("mapping/checs.txt")
        # noinspection PyTypeChecker
        table = np.genfromtxt(mapping_path, delimiter='\t', names=True, dtype=None)

        # Pixel Coordinates
        self.n_pixels = n_pixels if n_pixels else table.size
        pix_i = table['pixel'][:n_pixels]
        pix_x = table['xpix'][:n_pixels]  # Units: m
        pix_y = table['ypix'][:n_pixels]  # Units: m
        pix_row = table['row'][:n_pixels]
        pix_col = table['col'][:n_pixels]
        pix_2_sp = table['superpixel'][:n_pixels]
        pix_neighbours = get_square_grid_neighbours(
            row=pix_row, column=pix_col, diagonal=True
        )
        pix_separations = np.diff(np.sort(pix_x[pix_row == pix_row.max() // 2]))
        pix_size = 1 if not pix_separations.size else np.min(pix_separations)
        self.pixel = CameraCoordinates(
            i=pix_i,
            x=pix_x,
            y=pix_y,
            row=pix_row,
            column=pix_col,
            neighbours=pix_neighbours,
            size=pix_size
        )
        self.pixel_to_superpixel = pix_2_sp

        # Superpixel Coordinates
        self.n_superpixels = (self.n_pixels - 1) // 4 + 1
        sp_i = np.arange(self.n_superpixels)
        sp_x = self._get_superpixel_coordinate(self.n_superpixels, pix_2_sp, pix_x)
        sp_y = self._get_superpixel_coordinate(self.n_superpixels, pix_2_sp, pix_y)
        sp_row = self._get_superpixel_rowcol(self.n_superpixels, pix_2_sp, pix_row)
        sp_col = self._get_superpixel_rowcol(self.n_superpixels, pix_2_sp, pix_col)
        sp_neighbours = get_square_grid_neighbours(
            row=sp_row, column=sp_col, diagonal=True
        )
        sp_separations = np.diff(np.sort(sp_x[sp_row == sp_row.max() // 2]))
        sp_size = 1 if not sp_separations.size else np.min(sp_separations)
        self.superpixel = CameraCoordinates(
            i=sp_i,
            x=sp_x,
            y=sp_y,
            row=sp_row,
            column=sp_col,
            neighbours=sp_neighbours,
            size=sp_size
        )

    @staticmethod
    def _get_superpixel_coordinate(n_superpixels, superpixel, coordinate):
        superpixel_coord = np.zeros((n_superpixels, 4))
        superpixel_coord[superpixel, np.argsort(superpixel) % 4] = coordinate
        return superpixel_coord.mean(axis=1)

    @staticmethod
    def _get_superpixel_rowcol(n_superpixels, superpixel, rowcol):
        superpixel_rowcol = np.zeros((n_superpixels, 4))
        superpixel_rowcol[superpixel, np.argsort(superpixel) % 4] = rowcol
        return superpixel_rowcol[:, 0] // 2

    def reinitialise(self, n_pixels):
        """
        Initialise the mapping from scratch, providing the opportunity to
        change the number of pixels

        WARNING: can have odd consequences. Avoid using this if possible.

        Parameters
        ----------
        n_pixels : int
            Number of pixels in the simulated camera
        """
        self.__init__(n_pixels)
