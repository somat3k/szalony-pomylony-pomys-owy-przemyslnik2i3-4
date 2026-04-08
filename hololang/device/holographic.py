"""Holographic device abstractions.

Provides a base :class:`HolographicDevice` class and concrete subclasses
for laser projectors, galvanised mirror arrays, and sensors.
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Device status
# ---------------------------------------------------------------------------

class DeviceStatus(Enum):
    OFFLINE     = "offline"
    INITIALISING = "initialising"
    READY       = "ready"
    ACTIVE      = "active"
    ERROR       = "error"
    CALIBRATING = "calibrating"


# ---------------------------------------------------------------------------
# Base device
# ---------------------------------------------------------------------------

class HolographicDevice:
    """Base class for all holographic peripheral devices.

    Parameters
    ----------
    device_id:
        Unique hardware identifier.
    name:
        Human-readable label.
    """

    def __init__(self, device_id: str, name: str) -> None:
        self.device_id: str = device_id
        self.name:      str = name
        self.status:    DeviceStatus = DeviceStatus.OFFLINE
        self.config:    dict[str, Any] = {}
        self.params:    dict[str, Any] = {}
        self._callbacks: dict[str, list[Callable]] = {}
        self._log:      list[str] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialise(self) -> None:
        self.status = DeviceStatus.INITIALISING
        self._log_event("initialise")
        self._on_initialise()
        self.status = DeviceStatus.READY

    def activate(self) -> None:
        if self.status not in (DeviceStatus.READY, DeviceStatus.CALIBRATING):
            raise RuntimeError(
                f"Cannot activate device in state {self.status.value}"
            )
        self.status = DeviceStatus.ACTIVE
        self._log_event("activate")
        self._on_activate()

    def deactivate(self) -> None:
        self.status = DeviceStatus.READY
        self._log_event("deactivate")
        self._on_deactivate()

    def calibrate(self) -> None:
        self.status = DeviceStatus.CALIBRATING
        self._log_event("calibrate")
        self._on_calibrate()
        self.status = DeviceStatus.READY

    def shutdown(self) -> None:
        self.status = DeviceStatus.OFFLINE
        self._log_event("shutdown")

    # ------------------------------------------------------------------
    # Override hooks
    # ------------------------------------------------------------------

    def _on_initialise(self) -> None:
        pass

    def _on_activate(self) -> None:
        pass

    def _on_deactivate(self) -> None:
        pass

    def _on_calibrate(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def configure(self, **kwargs: Any) -> None:
        self.config.update(kwargs)
        self._log_event(f"configure {kwargs}")

    def set_param(self, name: str, value: Any) -> None:
        self.params[name] = value
        self._log_event(f"param {name}={value!r}")

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def on(self, event: str, callback: Callable) -> None:
        self._callbacks.setdefault(event, []).append(callback)

    def _emit(self, event: str, *args, **kwargs) -> None:
        for cb in self._callbacks.get(event, []):
            cb(*args, **kwargs)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log_event(self, msg: str) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] [{self.name}] {msg}"
        self._log.append(entry)

    def get_log(self) -> list[str]:
        return list(self._log)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(id={self.device_id!r}, name={self.name!r}, status={self.status.value})"


# ---------------------------------------------------------------------------
# Laser device
# ---------------------------------------------------------------------------

@dataclass
class BeamParameters:
    """Parameters controlling a single laser beam."""
    wavelength_nm: float = 532.0     # green laser default
    power_mw:      float = 10.0      # milliwatts
    divergence_mrad: float = 1.0     # milliradians
    polarisation:  str = "linear"    # linear | circular | elliptical
    modulation:    str = "cw"        # cw (continuous) | pulsed | sinusoidal


class LaserDevice(HolographicDevice):
    """Simulated laser beam source.

    Supports RGB channel control (650 nm, 532 nm, 445 nm) plus arbitrary
    wavelength configurations.
    """

    WAVELENGTH_RED   = 650.0
    WAVELENGTH_GREEN = 532.0
    WAVELENGTH_BLUE  = 445.0

    def __init__(self, device_id: str, name: str = "Laser") -> None:
        super().__init__(device_id, name)
        self.beams: dict[str, BeamParameters] = {
            "R": BeamParameters(self.WAVELENGTH_RED),
            "G": BeamParameters(self.WAVELENGTH_GREEN),
            "B": BeamParameters(self.WAVELENGTH_BLUE),
        }
        self._shutter_open = False

    def open_shutter(self) -> None:
        self._shutter_open = True
        self._log_event("shutter open")
        self._emit("shutter_open")

    def close_shutter(self) -> None:
        self._shutter_open = False
        self._log_event("shutter close")
        self._emit("shutter_close")

    def set_power(self, channel: str, power_mw: float) -> None:
        if channel not in self.beams:
            raise ValueError(f"Unknown channel {channel!r}")
        self.beams[channel].power_mw = power_mw
        self._log_event(f"power {channel}={power_mw} mW")
        self._emit("power_change", channel=channel, power=power_mw)

    def set_wavelength(self, channel: str, wavelength_nm: float) -> None:
        if channel not in self.beams:
            raise ValueError(f"Unknown channel {channel!r}")
        if not (380 <= wavelength_nm <= 780):
            raise ValueError(
                f"Wavelength {wavelength_nm} nm is outside visible range (380–780 nm)"
            )
        self.beams[channel].wavelength_nm = wavelength_nm
        self._log_event(f"wavelength {channel}={wavelength_nm} nm")

    def beam_at(self, r: float, g: float, b: float) -> dict[str, float]:
        """Set normalised RGB intensities [0–1] and return beam state."""
        self.beams["R"].power_mw = r * 100
        self.beams["G"].power_mw = g * 100
        self.beams["B"].power_mw = b * 100
        state = {ch: p.power_mw for ch, p in self.beams.items()}
        self._log_event(f"beam_at R={r:.2f} G={g:.2f} B={b:.2f}")
        self._emit("beam_change", state=state)
        return state

    def _on_calibrate(self) -> None:
        # Reset all beams to safe defaults
        for bp in self.beams.values():
            bp.power_mw = 0.0
        self._log_event("all channels zeroed during calibration")


# ---------------------------------------------------------------------------
# Galvanised mirror
# ---------------------------------------------------------------------------

@dataclass
class MirrorState:
    """Angular state of a single galvanised mirror axis."""
    angle_deg: float = 0.0      # current angle
    min_deg:   float = -30.0
    max_deg:   float = 30.0
    speed_deg_s: float = 100.0  # max angular velocity


class GalvanizedMirror(HolographicDevice):
    """Two-axis galvanised (galvo) mirror controller.

    Controls X (horizontal scan) and Y (vertical scan) mirrors separately.
    """

    def __init__(self, device_id: str, name: str = "GalvoMirror") -> None:
        super().__init__(device_id, name)
        self.x_axis = MirrorState()
        self.y_axis = MirrorState()
        self._pattern: list[tuple[float, float]] = []

    def set_angle(self, axis: str, angle: float) -> None:
        st = self.x_axis if axis.lower() == "x" else self.y_axis
        angle = max(st.min_deg, min(st.max_deg, angle))
        st.angle_deg = angle
        self._log_event(f"mirror {axis}={angle:.2f}°")
        self._emit("angle_change", axis=axis, angle=angle)

    def goto_xy(self, x_deg: float, y_deg: float) -> None:
        self.set_angle("x", x_deg)
        self.set_angle("y", y_deg)

    def scan_raster(self, x_range: tuple[float, float],
                    y_range: tuple[float, float],
                    steps_x: int = 100,
                    steps_y: int = 100) -> None:
        """Pre-compute a raster scan pattern."""
        self._pattern = []
        x0, x1 = x_range
        y0, y1 = y_range
        for row in range(steps_y):
            y = y0 + (y1 - y0) * row / max(1, steps_y - 1)
            for col in range(steps_x):
                x = x0 + (x1 - x0) * col / max(1, steps_x - 1)
                self._pattern.append((x, y))
        self._log_event(
            f"raster pattern prepared: {steps_x}×{steps_y} points"
        )

    def execute_pattern(self) -> None:
        for x, y in self._pattern:
            self.goto_xy(x, y)
            self._emit("point", x=x, y=y)

    def _on_calibrate(self) -> None:
        self.goto_xy(0.0, 0.0)


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------

class SensorType(Enum):
    PHOTODIODE   = "photodiode"
    CCD          = "ccd"
    CMOS         = "cmos"
    POSITION     = "position"
    TEMPERATURE  = "temperature"
    BEAM_PROFILE = "beam_profile"


class Sensor(HolographicDevice):
    """Generic sensor abstraction for holographic feedback control.

    Parameters
    ----------
    sensor_type:
        One of the :class:`SensorType` variants.
    sample_rate_hz:
        Nominal sample rate.
    """

    def __init__(
        self,
        device_id: str,
        name: str,
        sensor_type: SensorType = SensorType.PHOTODIODE,
        sample_rate_hz: float = 1000.0,
    ) -> None:
        super().__init__(device_id, name)
        self.sensor_type    = sensor_type
        self.sample_rate_hz = sample_rate_hz
        self._buffer:   list[float] = []
        self._reading:  float = 0.0

    def read(self) -> float:
        """Return the latest sensor reading."""
        return self._reading

    def inject(self, value: float) -> None:
        """Inject a simulated reading (for testing / simulation)."""
        self._reading = value
        self._buffer.append(value)
        self._emit("reading", value=value)

    def buffer_snapshot(self) -> list[float]:
        return list(self._buffer)

    def flush(self) -> None:
        self._buffer.clear()

    def __repr__(self) -> str:
        return (
            f"Sensor(id={self.device_id!r}, type={self.sensor_type.value}, "
            f"rate={self.sample_rate_hz} Hz)"
        )
