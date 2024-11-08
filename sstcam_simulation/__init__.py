import importlib.metadata
from .photoelectrons import Photoelectrons
from .camera import Camera, SSTCameraMapping
from .event import PhotoelectronSource, EventAcquisition
from .io import SimtelReader, PhotoelectronWriter, PhotoelectronReader

__version__ = importlib.metadata.version(__package__ or __name__)
