# Additive Process Monitor

Layer-by-layer anomaly detection and failure mode classification for FDM 3D printer fleets. Combines rolling Z-score detection with an XGBoost classifier to identify failure modes (clogs, warping, stringing, delamination) before a print is lost — saving material, machine time, and operator intervention.

---

## What it does

| Capability | Detail |
|---|---|
| **Real-time layer health score** | 0–100 health score updated every layer, with configurable alert thresholds |
| **Failure mode classification** | XGBoost identifies Normal / Clog / Warping / Stringing / Delamination |
| **Z-score anomaly flagging** | Rolling 30-layer statistical baseline catches sensor excursions instantly |
| **Fleet failure distribution** | Pie chart of dominant failure mode across all active print jobs |
| **Alert log** | Layer-indexed log of WARNING and CRITICAL events for operator review |

---

## Failure modes and sensor signatures

| Failure | Key signals |
|---|---|
| **Clog** | Extruder current ↑, layer height deviation ↑, print speed ↓ |
| **Warping** | Bed temperature ↓, layer height deviation ↑ at print edges |
| **Stringing** | Nozzle temperature ↑ (ooze between travel moves) |
| **Delamination** | Nozzle temperature ↓ → cold extrusion → poor inter-layer bonding |

---

## Sensors monitored

| Sensor | Unit |
|---|---|
| Nozzle Temperature | °C |
| Bed Temperature | °C |
| Extruder Motor Current | A |
| Layer Height Deviation | mm |
| Print Speed | % of nominal |
| Ambient Temperature | °C |

---

## Quickstart

```bash
git clone https://github.com/Jimmply/additive-process-monitor
cd additive-process-monitor
pip install -r requirements.txt
streamlit run src/app.py
```

---

## Project structure

```
additive-process-monitor/
├── src/
│   ├── data_generator.py   # FDM physics simulation + failure event injection
│   ├── monitor.py          # Rolling Z-score + XGBoost failure classifier
│   └── app.py              # Streamlit dashboard with layer health timeline
├── tests/
│   └── test_generator.py
└── requirements.txt
```

---

## Methodology

**Data generation** — `PrintJobGenerator` simulates nominal FDM operation (Ornstein-Uhlenbeck noise on all channels for realistic drift) then injects failure events at random layers using physics-based disturbance models. `PrintFleetGenerator` assembles a fleet of 30 jobs with mixed failure modes.

**Detection pipeline**
1. **Rolling Z-score** (30-layer window): flags any sensor reading beyond 3σ from local baseline — catches sudden events.
2. **XGBoost classifier** (10-layer rolling mean + std features): identifies the failure *mode* and provides a calibrated probability that becomes the health score.

**Health score** = P(Normal) × 100 — gives a continuous 0–100 metric that alert thresholds can be applied against without retraining.

---

## Business value

A 400-layer PLA print on a Prusa MK4 costs ~3–4 hours and $2–5 of material. Detecting a clog at layer 80 (20% complete) rather than layer 380 (95% complete) saves:
- **~3 hours** of unattended machine time
- The material consumed in layers 80–380
- Downstream rework if the failed part reached assembly

At fleet scale (20+ printers), automated early-stop logic driven by this monitor can recover hundreds of machine-hours per month.

---

## Tech stack

Python · XGBoost · Scikit-learn · Pandas · NumPy · Streamlit · Plotly
