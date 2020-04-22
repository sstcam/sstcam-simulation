from sstcam_simulation.camera import Camera
from sstcam_simulation.plotting.image import CameraImage


def test_camera_image():
    camera = Camera()
    image = CameraImage.from_coordinates(camera.mapping.pixel)
    assert image.n_pixels == camera.mapping.n_pixels


def test_camera_image_superpixel():
    camera = Camera()
    image = CameraImage.from_coordinates(camera.mapping.superpixel)
    assert image.n_pixels == camera.mapping.n_superpixels
