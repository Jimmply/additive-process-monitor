# Additive Process Monitor

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/Jimmply/additive-process-monitor/workflows/CI/badge.svg)
![XGBoost](https://img.shields.io/badge/model-XGBoost+Z--score-orange)
![Accuracy](https://img.shields.io/badge/accuracy-90%25-brightgreen)

Layer-by-layer anomaly detection and failure mode classification for FDM 3D printer fleets. Combines rolling Z-score detection with an XGBoost classifier to identify failure modes (clogs, warping, stringing, delamination) before a print is lost — saving material, machine time, and operator intervention.

---

## Results

| Metric | Value |
|---|---|
| **Test Accuracy** | ~90% |
| **Failure modes** | 5 (Normal / Clog / Warping / Stringing / Delamination) |
| **Detection latency** | 1–3 layers (< 30 seconds at standard print speed) |
| **Fleet size** | 30 concurrent print jobs |
| **Health score** | P(Normal) × 100 — continuous 0–100, threshold-configurable |

---

## What it does

| Capability | Detail |
|---|---|
| **Real-time layer health score** | 0–100 health score updated every layer with configurable alert thresholds |
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

# Generate fleet simulation data
python scripts/generate_data.py

# Train classifier
python scripts/train.py

# Launch fleet dashboard
streamlit run src/app.py
```

---

## Project structure

```
additive-process-monitor/
├── .github/workflows/ci.yml   # pytest + smoke test
├── config/settings.yaml       # thresholds, alert levels, model params
├── scripts/
│   ├── generate_data.py       # generate and save fleet simulation dataset
│   └── train.py               # train, evaluate, save classifier
├── src/
│   ├── data_generator.py      # FDM physics simulation + failure event injection
│   ├── monitor.py             # rolling Z-score + XGBoost failure classifier
│   └── app.py                 # Streamlit fleet dashboard with layer health timeline
├── tests/
│   └── test_generator.py
└── requirements.txt
```

---

## Methodology

**Data generation** — `PrintJobGenerator` simulates nominal FDM operation with Ornstein-Uhlenbeck noise on all channels for realistic sensor drift, then injects failure events at random layers using physics-based disturbance models. `PrintFleetGenerator` assembles a fleet of 30 jobs with mixed failure modes and staggered start layers.

**Detection pipeline**
1. **Rolling Z-score** (30-layer window): flags any sensor reading beyond 3σ from local baseline — catches sudden step-change events.
2. **XGBoost classifier** (10-layer rolling mean + std features): identifies the failure *mode* and provides a calibrated probability that becomes the health score.

**Health score** = P(Normal) × 100 — gives a continuous 0–100 metric that alert thresholds can be applied against without model retraining.

---

## Business value

A 400-layer PLA print on a standard FDM printer costs ~3–4 hours and $2–5 of material. Detecting a clog at layer 80 (20% complete) rather than layer 380 (95% complete) saves:
- **~3 hours** of unattended machine time
- The material consumed in layers 80–380
- Downstream rework if the failed part reached assembly

At fleet scale (20+ printers), automated early-stop logic driven by this monitor can recover hundreds of machine-hours per month.

---

## Tech stack

Python · XGBoost · Scikit-learn · Pandas · NumPy · Streamlit · Plotly

---

## License

MIT © Dmitri Shurkhai
