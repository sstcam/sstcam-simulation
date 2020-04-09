from sstcam_simulation.camera.mapping import get_neighbours, PixelMapping, \
    SuperpixelMapping
import numpy as np


def test_get_neighbours():
    row = np.array([0, 0, 1, 1, 2])
    col = np.array([0, 1, 0, 1, 1])

    neighbours = get_neighbours(row, col, False)
    neighbours_expected = np.array([
        [0, 1], [0, 2], [1, 3], [2, 3], [3, 4]
    ])
    assert np.array_equal(neighbours, neighbours_expected)

    neighbours = get_neighbours(row, col, True)
    neighbours_expected = np.array([
        [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4]
    ])
    assert np.array_equal(neighbours, neighbours_expected)

    # Single Pixel
    neighbours = get_neighbours(np.array([0]), np.array([0]), False)
    neighbours_expected = np.array([[], []]).T
    assert np.array_equal(neighbours, neighbours_expected)


def test_pixel_mapping():
    pixel_mapping = PixelMapping()
    assert pixel_mapping.n_pixels == 2048
    assert pixel_mapping.i.size == pixel_mapping.n_pixels
    assert np.unique(pixel_mapping.row).size == 48
    assert np.unique(pixel_mapping.column).size == 48
    assert pixel_mapping.neighbours.shape == (7910, 2)

    pixel_mapping = PixelMapping(n_pixels=64)
    assert pixel_mapping.n_pixels == 64
    assert pixel_mapping.i.size == pixel_mapping.n_pixels
    assert np.unique(pixel_mapping.row).size == 8
    assert np.unique(pixel_mapping.column).size == 8
    assert pixel_mapping.neighbours.shape == (210, 2)

    pixel_mapping = PixelMapping(n_pixels=4)
    assert pixel_mapping.n_pixels == 4
    assert pixel_mapping.i.size == pixel_mapping.n_pixels
    assert np.unique(pixel_mapping.row).size == 2
    assert np.unique(pixel_mapping.column).size == 2
    assert pixel_mapping.neighbours.shape == (6, 2)

    pixel_mapping = PixelMapping(n_pixels=1)
    assert pixel_mapping.n_pixels == 1
    assert pixel_mapping.i.size == pixel_mapping.n_pixels
    assert np.unique(pixel_mapping.row).size == 1
    assert np.unique(pixel_mapping.column).size == 1
    assert pixel_mapping.neighbours.shape == (0, 2)


def test_superpixel_mapping():
    pixel_mapping = PixelMapping()
    superpixel_mapping = SuperpixelMapping(pixel_mapping)
    assert superpixel_mapping.n_superpixels == 512
    assert superpixel_mapping.i.size == superpixel_mapping.n_superpixels
    assert np.unique(superpixel_mapping.row).size == 24
    assert np.unique(superpixel_mapping.column).size == 24
    assert superpixel_mapping.neighbours.shape == (1910, 2)

    pixel_mapping = PixelMapping(n_pixels=64)
    superpixel_mapping = SuperpixelMapping(pixel_mapping)
    assert superpixel_mapping.n_superpixels == 16
    assert superpixel_mapping.i.size == superpixel_mapping.n_superpixels
    assert np.unique(superpixel_mapping.row).size == 4
    assert np.unique(superpixel_mapping.column).size == 4
    assert superpixel_mapping.neighbours.shape == (42, 2)

    pixel_mapping = PixelMapping(n_pixels=4)
    superpixel_mapping = SuperpixelMapping(pixel_mapping)
    assert superpixel_mapping.n_superpixels == 1
    assert superpixel_mapping.i.size == superpixel_mapping.n_superpixels
    assert np.unique(superpixel_mapping.row).size == 1
    assert np.unique(superpixel_mapping.column).size == 1
    assert superpixel_mapping.neighbours.shape == (0, 2)

    pixel_mapping = PixelMapping(n_pixels=1)
    superpixel_mapping = SuperpixelMapping(pixel_mapping)
    assert superpixel_mapping.n_superpixels == 1
    assert superpixel_mapping.i.size == superpixel_mapping.n_superpixels
    assert np.unique(superpixel_mapping.row).size == 1
    assert np.unique(superpixel_mapping.column).size == 1
    assert superpixel_mapping.neighbours.shape == (0, 2)
