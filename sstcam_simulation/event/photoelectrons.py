from sstcam_simulation.photoelectrons import Photoelectrons
import warnings
warnings.filterwarnings('default', module='sstcam_simulation')

msg = "Photoelectrons class has been moved to sstcam_simulation.photoelectrons"
warnings.warn(msg, DeprecationWarning)
