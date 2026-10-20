"""Explicit hardware boundaries. No real driver is shipped in Stage A."""
from abc import ABC, abstractmethod
class CameraAdapter(ABC):
    @abstractmethod
    def read_rgb(self): raise NotImplementedError
class GimbalAdapter(ABC):
    @abstractmethod
    def read_state(self): raise NotImplementedError
    @abstractmethod
    def command(self, action): raise NotImplementedError
class NotImplementedHardwareAdapter(CameraAdapter, GimbalAdapter):
    def read_rgb(self): raise RuntimeError('real camera adapter is not implemented in Stage A')
    def read_state(self): raise RuntimeError('real gimbal adapter is not implemented in Stage A')
    def command(self, action): raise RuntimeError('real actuator commands are disabled in Stage A')
