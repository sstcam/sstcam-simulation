from ..data import get_data
import numpy as np

__all__ = [
    "get_neighbours",
    "PixelMapping",
    "SuperpixelMapping",
]


DEFAULT_MAPPING_PATH = get_data("mapping/checs.txt")


def get_neighbours(row, column, diagonal):
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


class PixelMapping:
    def __init__(self, mapping_path=DEFAULT_MAPPING_PATH, n_pixels=None):
        """
        Container for the pixel mapping

        Parameters
        ----------
        mapping_path : str
            Path to a txt file containing the pixel mapping definition
        n_pixels : int
            Number of pixels to simulate
            If None, then all pixels in file are simulated
        """
        # noinspection PyTypeChecker
        table = np.genfromtxt(mapping_path, delimiter='\t', names=True, dtype=None)
        self.n_pixels = n_pixels if n_pixels else table.size
        self.i = table['pixel'][:n_pixels]
        self.x = table['xpix'][:n_pixels]
        self.y = table['ypix'][:n_pixels]
        self.row = table['row'][:n_pixels]
        self.column = table['col'][:n_pixels]
        self.superpixel = table['superpixel'][:n_pixels]
        self.neighbours = get_neighbours(row=self.row, column=self.column, diagonal=True)


class SuperpixelMapping:
    def __init__(self, pixel):
        """
        Container for the superpixel mapping

        Parameters
        ----------
        pixel : PixelMapping
            Mapping of the pixels, used to generate the superpixel mapping
        """
        self.n_superpixels = (pixel.n_pixels - 1) // 4 + 1

        self.i = np.arange(self.n_superpixels)
        self.x = self._get_coordinate(self.n_superpixels, pixel.superpixel, pixel.x)
        self.y = self._get_coordinate(self.n_superpixels, pixel.superpixel, pixel.y)
        self.row = self._get_rowcol(self.n_superpixels, pixel.superpixel, pixel.row)
        self.column = self._get_rowcol(self.n_superpixels, pixel.superpixel, pixel.column)
        self.neighbours = get_neighbours(row=self.row, column=self.column, diagonal=True)

    @staticmethod
    def _get_coordinate(n_superpixels, superpixel, coordinate):
        superpixel_coord = np.zeros((n_superpixels, 4))
        superpixel_coord[superpixel, np.argsort(superpixel) % 4] = coordinate
        return superpixel_coord.mean(axis=1)

    @staticmethod
    def _get_rowcol(n_superpixels, superpixel, rowcol):
        superpixel_rowcol = np.zeros((n_superpixels, 4))
        superpixel_rowcol[superpixel, np.argsort(superpixel) % 4] = rowcol
        return superpixel_rowcol[:, 0] // 2
