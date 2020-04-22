# from abc import ABCMeta, abstractmethod
#
# class Trigger(metaclass=ABCMeta):
#     @abstractmethod
#     def __call__(self, continuous_readout):
#         """
#
#         Parameters
#         ----------
#         continuous_readout
#
#         Returns
#         -------
#
#         """
#
#
# class NNSuperpixelAboveThreshold(Trigger):
#     def __init__(self, trigger_threshold, coincidence_window, mapping):
#         self.trigger_threshold = trigger_threshold
#         self.coincidence_window = coincidence_window
#
#     def __call__(self, continuous_readout):
#         line = self.get_superpixel_digital_trigger_line(continuous_readout)
#         extended = self.extend_by_coincidence_window(line)
#         return self.get_backplane_triggers(extended)
#
#     def get_superpixel_digital_trigger_line(self, continuous_readout):
#         # Sum superpixel readouts
#         superpixel = self.camera.pixel.superpixel
#         n_superpixels = self.camera.superpixel.n_superpixels
#         superpixel_sum = sum_superpixels(continuous_readout, superpixel, n_superpixels)
#
#         # Discriminate superpixel readout with threshold
#         # (First convert threshold to sample units, i.e. p.e./ns)
#         sample_unit_per_photoelectron = self.camera.reference_pulse.peak_height
#         threshold = self.camera.trigger_threshold * sample_unit_per_photoelectron
#         above_threshold = superpixel_sum >= threshold
#
#     def extend_by_coincidence_window(self, digital_trigger_line):
#         # Extend by coincidence window length
#         division = self.camera.continuous_readout_sample_division
#         coincidence_samples = self.camera.coincidence_window * division
#         digital_trigger = add_coincidence_window(above_threshold, coincidence_samples)
#
#     def get_backplane_triggers(self, digital_trigger_line):
#         pass