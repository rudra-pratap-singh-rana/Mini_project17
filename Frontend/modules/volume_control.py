"""
volume_control.py
-----------------
Controls system volume using the distance between thumb and index finger.
"""

import platform
import numpy as np


def get_volume_controller():
    """
    Factory function — returns the right VolumeController for the OS.

    Returns:
        VolumeController instance (Windows/Linux/Mac).
    """
    os_name = platform.system()
    if os_name == "Windows":
        return WindowsVolumeController()
    elif os_name == "Darwin":
        return MacVolumeController()
    else:
        return LinuxVolumeController()


class VolumeController:
    """Base class for platform-specific volume control."""

    MIN_DIST = 20   # pixels — minimum hand distance (mute)
    MAX_DIST = 200  # pixels — maximum hand distance (full volume)

    def distance_to_volume(self, distance: float) -> int:
        """
        Convert pixel distance to a volume percentage (0–100).

        Args:
            distance: Pixel distance between thumb tip and index tip.

        Returns:
            Integer volume level 0–100.
        """
        vol = np.interp(distance, [self.MIN_DIST, self.MAX_DIST], [0, 100])
        return int(np.clip(vol, 0, 100))

    def set_volume(self, level: int) -> None:
        """
        Set system volume.

        Args:
            level: Integer 0–100.
        """
        raise NotImplementedError


class WindowsVolumeController(VolumeController):
    """Windows volume control via pycaw."""

    def __init__(self):
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._volume = cast(interface, POINTER(IAudioEndpointVolume))
            self._available = True
        except Exception:
            self._available = False

    def set_volume(self, level: int) -> None:
        if not self._available:
            return
        vol_scalar = level / 100.0
        self._volume.SetMasterVolumeLevelScalar(vol_scalar, None)


class MacVolumeController(VolumeController):
    """macOS volume control via osascript."""

    def set_volume(self, level: int) -> None:
        import subprocess
        subprocess.run(["osascript", "-e", f"set volume output volume {level}"], check=False)


class LinuxVolumeController(VolumeController):
    """Linux volume control via amixer."""

    def set_volume(self, level: int) -> None:
        import subprocess
        subprocess.run(
            ["amixer", "-D", "pulse", "sset", "Master", f"{level}%"],
            check=False,
            capture_output=True,
        )
