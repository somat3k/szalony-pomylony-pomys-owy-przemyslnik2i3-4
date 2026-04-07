"""Tests for the holographic device layer."""

import pytest
from hololang.device.holographic import (
    HolographicDevice, DeviceStatus,
    LaserDevice, BeamParameters,
    GalvanizedMirror, MirrorState,
    Sensor, SensorType,
)


# ---------------------------------------------------------------------------
# Base device
# ---------------------------------------------------------------------------

def _make_device() -> HolographicDevice:
    return HolographicDevice("test-001", "TestDevice")


def test_initial_status():
    d = _make_device()
    assert d.status == DeviceStatus.OFFLINE


def test_initialise():
    d = _make_device()
    d.initialise()
    assert d.status == DeviceStatus.READY


def test_activate():
    d = _make_device()
    d.initialise()
    d.activate()
    assert d.status == DeviceStatus.ACTIVE


def test_activate_without_init_raises():
    d = _make_device()
    with pytest.raises(RuntimeError):
        d.activate()


def test_deactivate():
    d = _make_device()
    d.initialise()
    d.activate()
    d.deactivate()
    assert d.status == DeviceStatus.READY


def test_shutdown():
    d = _make_device()
    d.initialise()
    d.shutdown()
    assert d.status == DeviceStatus.OFFLINE


def test_configure():
    d = _make_device()
    d.configure(mode="turbo", channel=3)
    assert d.config["mode"] == "turbo"
    assert d.config["channel"] == 3


def test_set_param():
    d = _make_device()
    d.set_param("power", 50.0)
    assert d.params["power"] == 50.0


def test_event_callback():
    d = _make_device()
    events = []
    d.on("activate", lambda: events.append("activated"))
    d.initialise()
    d._emit("activate")
    assert "activated" in events


def test_log():
    d = _make_device()
    d.initialise()
    log = d.get_log()
    assert any("initialise" in e for e in log)


# ---------------------------------------------------------------------------
# Laser device
# ---------------------------------------------------------------------------

def test_laser_beam_at():
    laser = LaserDevice("laser-001")
    laser.initialise()
    state = laser.beam_at(0.5, 1.0, 0.0)
    assert state["R"] == 50.0
    assert state["G"] == 100.0
    assert state["B"] == 0.0


def test_laser_set_power():
    laser = LaserDevice("laser-002")
    laser.initialise()
    laser.set_power("G", 75.0)
    assert laser.beams["G"].power_mw == 75.0


def test_laser_invalid_channel():
    laser = LaserDevice("laser-003")
    laser.initialise()
    with pytest.raises(ValueError):
        laser.set_power("X", 10.0)


def test_laser_wavelength():
    laser = LaserDevice("laser-004")
    laser.initialise()
    laser.set_wavelength("R", 650.0)
    assert laser.beams["R"].wavelength_nm == 650.0


def test_laser_wavelength_out_of_range():
    laser = LaserDevice("laser-005")
    laser.initialise()
    with pytest.raises(ValueError):
        laser.set_wavelength("R", 200.0)  # UV – not visible


def test_laser_shutter():
    laser = LaserDevice("laser-006")
    laser.initialise()
    assert laser._shutter_open is False
    laser.open_shutter()
    assert laser._shutter_open is True
    laser.close_shutter()
    assert laser._shutter_open is False


def test_laser_calibration_zeros_power():
    laser = LaserDevice("laser-007")
    laser.initialise()
    laser.set_power("G", 50.0)
    laser.calibrate()
    assert laser.beams["G"].power_mw == 0.0


# ---------------------------------------------------------------------------
# Galvanized mirror
# ---------------------------------------------------------------------------

def test_mirror_set_angle():
    m = GalvanizedMirror("mirror-001")
    m.initialise()
    m.set_angle("x", 15.0)
    assert m.x_axis.angle_deg == 15.0


def test_mirror_clamps_angle():
    m = GalvanizedMirror("mirror-002")
    m.initialise()
    m.set_angle("x", 999.0)
    assert m.x_axis.angle_deg == m.x_axis.max_deg


def test_mirror_goto_xy():
    m = GalvanizedMirror("mirror-003")
    m.initialise()
    m.goto_xy(-5.0, 10.0)
    assert m.x_axis.angle_deg == -5.0
    assert m.y_axis.angle_deg == 10.0


def test_mirror_raster_pattern():
    m = GalvanizedMirror("mirror-004")
    m.initialise()
    m.scan_raster((-10, 10), (-5, 5), steps_x=5, steps_y=3)
    assert len(m._pattern) == 15  # 5 × 3


def test_mirror_calibrate_zeros():
    m = GalvanizedMirror("mirror-005")
    m.initialise()
    m.set_angle("x", 20.0)
    m.calibrate()
    assert m.x_axis.angle_deg == 0.0


# ---------------------------------------------------------------------------
# Sensor
# ---------------------------------------------------------------------------

def test_sensor_inject_read():
    s = Sensor("sensor-001", "PhotoSensor")
    s.initialise()
    s.inject(3.14)
    assert s.read() == 3.14


def test_sensor_buffer():
    s = Sensor("sensor-002", "TestSensor")
    s.initialise()
    s.inject(1.0)
    s.inject(2.0)
    s.inject(3.0)
    buf = s.buffer_snapshot()
    assert buf == [1.0, 2.0, 3.0]


def test_sensor_flush():
    s = Sensor("sensor-003", "Flush")
    s.inject(5.0)
    s.flush()
    assert s.buffer_snapshot() == []


def test_sensor_event():
    s = Sensor("sensor-004", "Event")
    readings = []
    s.on("reading", lambda value: readings.append(value))
    s.inject(7.7)
    assert 7.7 in readings


def test_sensor_type():
    s = Sensor("sensor-005", "CCD", SensorType.CCD)
    assert s.sensor_type == SensorType.CCD
