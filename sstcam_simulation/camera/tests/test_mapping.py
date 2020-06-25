from sstcam_simulation.camera.mapping import (
    get_square_grid_neighbours,
    SSTCameraMapping,
)
import numpy as np


def test_get_neighbours():
    row = np.array([0, 0, 1, 1, 2])
    col = np.array([0, 1, 0, 1, 1])

    neighbours = get_square_grid_neighbours(row, col, False)
    neighbours_expected = np.array([[0, 1], [0, 2], [1, 3], [2, 3], [3, 4]])
    assert np.array_equal(neighbours, neighbours_expected)

    neighbours = get_square_grid_neighbours(row, col, True)
    neighbours_expected = np.array(
        [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3], [2, 4], [3, 4]]
    )
    assert np.array_equal(neighbours, neighbours_expected)

    # Single Pixel
    neighbours = get_square_grid_neighbours(np.array([0]), np.array([0]), False)
    neighbours_expected = np.array([[], []]).T
    assert np.array_equal(neighbours, neighbours_expected)


def test_pixel_mapping():
    mapping = SSTCameraMapping()
    assert mapping.n_pixels == 2048
    assert mapping.pixel.i.size == mapping.n_pixels
    assert np.unique(mapping.pixel.row).size == 48
    assert np.unique(mapping.pixel.column).size == 48
    assert mapping.pixel.neighbours.shape == (7910, 2)

    mapping = SSTCameraMapping(n_pixels=64)
    assert mapping.n_pixels == 64
    assert mapping.pixel.i.size == mapping.n_pixels
    assert np.unique(mapping.pixel.row).size == 8
    assert np.unique(mapping.pixel.column).size == 8
    assert mapping.pixel.neighbours.shape == (210, 2)

    mapping = SSTCameraMapping(n_pixels=4)
    assert mapping.n_pixels == 4
    assert mapping.pixel.i.size == mapping.n_pixels
    assert np.unique(mapping.pixel.row).size == 2
    assert np.unique(mapping.pixel.column).size == 2
    assert mapping.pixel.neighbours.shape == (6, 2)

    mapping = SSTCameraMapping(n_pixels=1)
    assert mapping.n_pixels == 1
    assert mapping.pixel.i.size == mapping.n_pixels
    assert np.unique(mapping.pixel.row).size == 1
    assert np.unique(mapping.pixel.column).size == 1
    assert mapping.pixel.neighbours.shape == (0, 2)


def test_superpixel_mapping():
    mapping = SSTCameraMapping()
    assert mapping.n_superpixels == 512
    assert mapping.superpixel.i.size == mapping.n_superpixels
    assert np.unique(mapping.superpixel.row).size == 24
    assert np.unique(mapping.superpixel.column).size == 24
    assert mapping.superpixel.neighbours.shape == (1910, 2)

    mapping = SSTCameraMapping(n_pixels=64)
    assert mapping.n_superpixels == 16
    assert mapping.superpixel.i.size == mapping.n_superpixels
    assert np.unique(mapping.superpixel.row).size == 4
    assert np.unique(mapping.superpixel.column).size == 4
    assert mapping.superpixel.neighbours.shape == (42, 2)

    mapping = SSTCameraMapping(n_pixels=4)
    assert mapping.n_superpixels == 1
    assert mapping.superpixel.i.size == mapping.n_superpixels
    assert np.unique(mapping.superpixel.row).size == 1
    assert np.unique(mapping.superpixel.column).size == 1
    assert mapping.superpixel.neighbours.shape == (0, 2)

    mapping = SSTCameraMapping(n_pixels=1)
    assert mapping.n_superpixels == 1
    assert mapping.superpixel.i.size == mapping.n_superpixels
    assert np.unique(mapping.superpixel.row).size == 1
    assert np.unique(mapping.superpixel.column).size == 1
    assert mapping.superpixel.neighbours.shape == (0, 2)


def test_reinitialise():
    mapping = SSTCameraMapping(n_pixels=20)
    assert mapping.n_pixels == 20
    assert mapping.pixel.i.size == 20
    mapping.reinitialise(10)
    assert mapping.n_pixels == 10
    assert mapping.pixel.i.size == 10
