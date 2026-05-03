"""
FDM 3D Printer In-Process Synthetic Data Generator
====================================================
Simulates layer-by-layer sensor telemetry for a fused-deposition modelling
printer fleet, including injected failure events.

Sensor channels (per layer):
  nozzle_temp_c       - Extruder nozzle temperature (°C)
  bed_temp_c          - Build plate temperature (°C)
  extruder_current_a  - Extruder stepper motor current (A)
  layer_height_dev_mm - Deviation of measured layer height from target (mm)
  print_speed_pct     - Feed-rate as % of nominal setting
  ambient_temp_c      - Enclosure ambient temperature (°C)

Failure modes injected:
  Clog         - Partial nozzle blockage: current spikes, height deviation rises
  Warping      - Bed adhesion loss: bed temp drops, height deviation increases
  Stringing    - Over-temperature ooze: nozzle temp drifts high
  Delamination - Thermal boundary: nozzle temp drops → poor layer bonding
  Normal       - Nominal operation throughout
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FAILURE_MODES = ["Normal", "Clog", "Warping", "Stringing", "Delamination"]
FAILURE_COLORS = {
    "Normal": "#2ecc71",
    "Clog": "#e74c3c",
    "Warping": "#e67e22",
    "Stringing": "#9b59b6",
    "Delamination": "#3498db",
}
SENSOR_COLS = [
    "nozzle_temp_c",
    "bed_temp_c",
    "extruder_current_a",
    "layer_height_dev_mm",
    "print_speed_pct",
    "ambient_temp_c",
]


@dataclass
class PrintConfig:
    """Nominal operating parameters for an FDM printer."""
    n_layers: int = 400
    nozzle_temp_nominal: float = 215.0   # °C
    bed_temp_nominal: float = 60.0       # °C
    extruder_current_nominal: float = 1.2  # A
    target_layer_height: float = 0.2     # mm
    print_speed_nominal: float = 100.0   # % of max
    ambient_temp_nominal: float = 26.0   # °C


@dataclass
class FailureEvent:
    """A failure event injected into a print job."""
    mode: str
    start_layer: int
    duration_layers: int


class PrintJobGenerator:
    """
    Generates layer-by-layer sensor data for a single 3D print job.

    Parameters
    ----------
    config : PrintConfig, optional
    failure_event : FailureEvent, optional
        If None, generates a normal (no-failure) print.
    random_seed : int
    """

    def __init__(
        self,
        config: Optional[PrintConfig] = None,
        failure_event: Optional[FailureEvent] = None,
        random_seed: int = 42,
    ) -> None:
        self.config = config or PrintConfig()
        self.failure_event = failure_event
        self._rng = np.random.default_rng(random_seed)

    def generate(self) -> pd.DataFrame:
        """Return layer-indexed DataFrame with sensor readings and labels."""
        cfg = self.config
        n = cfg.n_layers
        layers = np.arange(1, n + 1)

        nozzle = self._base_signal(cfg.nozzle_temp_nominal, sigma=0.8, n=n)
        bed    = self._base_signal(cfg.bed_temp_nominal,    sigma=0.4, n=n)
        cur    = self._base_signal(cfg.extruder_current_nominal, sigma=0.03, n=n)
        height = self._rng.normal(0.0, 0.004, n)  # deviation from target
        speed  = self._base_signal(cfg.print_speed_nominal, sigma=1.0, n=n)
        ambient = self._base_signal(cfg.ambient_temp_nominal, sigma=0.3, n=n)

        failure_label = np.full(n, "Normal", dtype=object)

        if self.failure_event is not None:
            fe = self.failure_event
            s = fe.start_layer - 1
            e = min(s + fe.duration_layers, n)
            failure_label[s:e] = fe.mode
            nozzle, bed, cur, height, speed = self._inject_failure(
                fe, s, e, nozzle, bed, cur, height, speed, n
            )

        df = pd.DataFrame({
            "layer": layers,
            "nozzle_temp_c": nozzle.round(2),
            "bed_temp_c": bed.round(2),
            "extruder_current_a": cur.round(4),
            "layer_height_dev_mm": height.round(4),
            "print_speed_pct": speed.round(1),
            "ambient_temp_c": ambient.round(2),
            "failure_label": failure_label,
            "is_anomaly": failure_label != "Normal",
        })
        logger.info(
            "Print job: %d layers, failure=%s",
            n,
            self.failure_event.mode if self.failure_event else "None",
        )
        return df

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _base_signal(self, mean: float, sigma: float, n: int) -> np.ndarray:
        noise = self._rng.normal(0, sigma, n)
        drift = self._mean_reverting_walk(n, sigma=sigma * 0.15, theta=0.06)
        return mean + noise + drift

    def _mean_reverting_walk(self, n: int, sigma: float, theta: float) -> np.ndarray:
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = x[i - 1] * (1 - theta) + self._rng.normal(0, sigma)
        return x

    def _inject_failure(
        self, fe: FailureEvent, s: int, e: int,
        nozzle: np.ndarray, bed: np.ndarray, cur: np.ndarray,
        height: np.ndarray, speed: np.ndarray, n: int,
    ) -> Tuple[np.ndarray, ...]:
        dur = e - s
        ramp = np.linspace(0, 1, dur)

        if fe.mode == "Clog":
            # Current spikes, layer height deviation increases (underextrusion)
            cur[s:e] += 0.6 * ramp + self._rng.normal(0, 0.05, dur)
            height[s:e] += 0.12 * ramp + self._rng.normal(0, 0.008, dur)
            speed[s:e] -= 15 * ramp  # printer slows trying to push through

        elif fe.mode == "Warping":
            # Bed temp drops (adhesion lost), height deviation grows at edges
            bed[s:e] -= 12 * ramp + self._rng.normal(0, 0.5, dur)
            height[s:e] += 0.08 * ramp ** 1.5 + self._rng.normal(0, 0.006, dur)

        elif fe.mode == "Stringing":
            # Nozzle too hot → ooze between travels → erratic height readings
            nozzle[s:e] += 14 * ramp + self._rng.normal(0, 0.6, dur)
            height[s:e] += self._rng.normal(0, 0.025, dur) * ramp

        elif fe.mode == "Delamination":
            # Nozzle temp drops → cold extrusion → poor layer adhesion
            nozzle[s:e] -= 20 * ramp + self._rng.normal(0, 0.5, dur)
            cur[s:e] += 0.3 * ramp  # motor works harder pushing cold filament
            height[s:e] -= 0.03 * ramp  # slightly under-extruded layers

        return nozzle, bed, cur, height, speed


class PrintFleetGenerator:
    """
    Generates a fleet of print jobs with mixed failure modes.

    Parameters
    ----------
    n_jobs : int
        Total number of print jobs to simulate.
    config : PrintConfig, optional
    random_seed : int
    """

    def __init__(
        self,
        n_jobs: int = 30,
        config: Optional[PrintConfig] = None,
        random_seed: int = 42,
    ) -> None:
        self.n_jobs = n_jobs
        self.config = config or PrintConfig()
        self._rng = np.random.default_rng(random_seed)

    def generate(self) -> pd.DataFrame:
        """Return concatenated DataFrame for all print jobs."""
        mode_weights = [0.40, 0.18, 0.18, 0.12, 0.12]  # Normal + 4 failures
        modes_pool = ["Normal"] + [m for m in FAILURE_MODES if m != "Normal"]
        frames: List[pd.DataFrame] = []

        for job_id in range(self.n_jobs):
            mode = self._rng.choice(modes_pool, p=mode_weights)
            n_layers = int(self.config.n_layers * self._rng.uniform(0.6, 1.4))
            cfg = PrintConfig(n_layers=n_layers, **{
                k: v for k, v in self.config.__dict__.items() if k != "n_layers"
            })

            fe = None
            if mode != "Normal":
                start = self._rng.integers(
                    int(n_layers * 0.1), int(n_layers * 0.7)
                )
                duration = self._rng.integers(
                    int(n_layers * 0.15), int(n_layers * 0.50)
                )
                fe = FailureEvent(mode, int(start), int(duration))

            gen = PrintJobGenerator(
                config=cfg, failure_event=fe,
                random_seed=int(self._rng.integers(0, 99999)),
            )
            df = gen.generate()
            df.insert(0, "job_id", f"JOB-{job_id:03d}")
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Fleet: %d jobs, %d total layers.", self.n_jobs, len(combined)
        )
        return combined
