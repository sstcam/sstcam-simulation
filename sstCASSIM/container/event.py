"""
Container class for the events

TODO:
    * Simple numpy array?
    * Numpy array subclass (event attributes)
        * sample_rate
        * duration
        * has_elecnoise
        * has_signal
        * has_nsb
        * photon_times (per pixel)
        * total_signal (per pixel)
        * n_l2_triggers
    * Class containing the different stages?
        * List possible traces...
    * Seperate trigger container?
"""
# import numpy as np
# from dataclasses import dataclass
# from enum import Enum
#
#
# class PhotonType(Enum):
#     SIGNAL = 1
#     NSB = 2
#     BOTH = 3
#
# 
# @dataclass
# class Event:
#     iev: int
#     n_pixels: int
#     n_photons_signal: int
#     photons_signal_: list
#     photons_nsb: list
#
#
#
#     waveform: Waveform
#     #
#     #
#     # n_samples: int
#     # sample_frequency: int
#     # has_electronic_noise: bool = False
#     # _waveform: np.ndarray = None
#     #
#     # @property
#     # def waveform(self):
#     #     if self._waveform is None:
#     #         self._waveform = np.zeros(self.n_samples, dtype=np.float32)
#     #     return self._waveform
#
# class Waveform(np.ndarray):
#     def __new__(cls, n_samples, iev, is_r1=False,
#                 first_cell_id=0, stale=False, missing_packets=False, t_tack=0,
#                 t_cpu_container=0, mc_true=None):
#         obj = np.asarray(input_array).view(cls)
#         obj.iev = iev
#         obj.is_r1 = is_r1
#         obj.first_cell_id = first_cell_id
#         obj.stale = stale
#         obj.missing_packets = missing_packets
#         obj.t_tack = t_tack
#         obj._t_cpu_container = t_cpu_container
#         obj.mc_true = mc_true
#         return obj
#
#     def __array_finalize__(self, obj):
#         if obj is None:
#             return
#         self.iev = getattr(obj, 'iev', None)
#         self.is_r1 = getattr(obj, 'is_r1', None)
#         self.first_cell_id = getattr(obj, 'first_cell_id', None)
#         self.stale = getattr(obj, 'stale', None)
#         self.missing_packets = getattr(obj, 'missing_packets', None)
#         self.t_tack = getattr(obj, 't_tack', None)
#         self._t_cpu_container = getattr(obj, '_t_cpu_container', None)
#         self.mc_true = getattr(obj, 'mc_true', None)
#
#     @property
#     def t_cpu(self):
#         t_cpu_s, t_cpu_ns = self._t_cpu_container
#         return pd.to_datetime(
#             np.int64(t_cpu_s * 1E9) + np.int64(t_cpu_ns),
#             unit='ns'
#         )