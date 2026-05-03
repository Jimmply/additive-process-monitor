"""Unit tests for 3D printer data generator."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from data_generator import (
    FAILURE_MODES,
    SENSOR_COLS,
    FailureEvent,
    PrintConfig,
    PrintFleetGenerator,
    PrintJobGenerator,
)


def test_normal_job_output():
    gen = PrintJobGenerator(random_seed=0)
    df = gen.generate()
    assert len(df) == 400  # default n_layers
    for col in SENSOR_COLS:
        assert col in df.columns
    assert "failure_label" in df.columns
    assert "is_anomaly" in df.columns


def test_normal_job_no_anomalies():
    gen = PrintJobGenerator(random_seed=1)
    df = gen.generate()
    assert (df["failure_label"] == "Normal").all()
    assert df["is_anomaly"].sum() == 0


def test_failure_injection():
    fe = FailureEvent(mode="Clog", start_layer=100, duration_layers=80)
    gen = PrintJobGenerator(failure_event=fe, random_seed=2)
    df = gen.generate()
    assert (df[df["layer"].between(100, 179)]["failure_label"] == "Clog").all()
    assert df["is_anomaly"].sum() == 80


def test_all_failure_modes():
    for mode in [m for m in FAILURE_MODES if m != "Normal"]:
        fe = FailureEvent(mode=mode, start_layer=50, duration_layers=60)
        gen = PrintJobGenerator(failure_event=fe, random_seed=3)
        df = gen.generate()
        assert mode in df["failure_label"].values


def test_fleet_generation():
    gen = PrintFleetGenerator(n_jobs=5, random_seed=4)
    df = gen.generate()
    assert df["job_id"].nunique() == 5
    assert len(df) > 0


def test_sensor_values_realistic():
    gen = PrintJobGenerator(random_seed=5)
    df = gen.generate()
    assert df["nozzle_temp_c"].between(180, 260).all()
    assert df["bed_temp_c"].between(40, 90).all()
    assert (df["extruder_current_a"] > 0).all()


def test_reproducibility():
    df1 = PrintJobGenerator(random_seed=99).generate()
    df2 = PrintJobGenerator(random_seed=99).generate()
    assert df1["nozzle_temp_c"].equals(df2["nozzle_temp_c"])
