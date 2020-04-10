from sstcam_simulation.camera import Camera
from sstcam_simulation.plotting.image import CameraImage


def test_camera_image():
    camera = Camera()
    image = CameraImage.from_mapping(camera.pixel)
    assert image.n_pixels == camera.pixel.n_pixels


def test_camera_image_superpixel():
    camera = Camera()
    image = CameraImage.from_mapping(camera.superpixel)
    assert image.n_pixels == camera.superpixel.n_superpixels
