"""
3D Printer In-Process Anomaly Monitor
======================================
Applies rolling-window statistical detection and an XGBoost failure
classifier to layer-by-layer FDM printer telemetry.

Detection pipeline:
  1. Rolling Z-score flags statistically unusual sensor readings per layer.
  2. XGBClassifier trained on multi-sensor windows identifies failure mode.
  3. Health score (0–100) aggregates per-layer anomaly signals.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from data_generator import FAILURE_COLORS, FAILURE_MODES, SENSOR_COLS

logger = logging.getLogger(__name__)

WINDOW = 10  # Layers in rolling statistics window


@dataclass
class MonitorResults:
    classification_report: str
    feature_importances: pd.Series
    label_encoder: LabelEncoder
    test_accuracy: float


class PrintMonitor:
    """
    Anomaly detection and failure classification for FDM print jobs.

    Usage::

        monitor = PrintMonitor()
        results = monitor.fit(fleet_df)
        scored = monitor.score(job_df)
    """

    def __init__(self, window: int = WINDOW) -> None:
        self.window = window
        self._clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            verbosity=0,
        )
        self._le = LabelEncoder()
        self._trained = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> MonitorResults:
        """Train the failure classifier on fleet-level data."""
        featured = self._build_features(df)
        X = featured[self._feature_names(df)].values
        y = self._le.fit_transform(df.loc[featured.index, "failure_label"])

        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )
        self._clf.fit(X_tr, y_tr)
        self._trained = True

        y_pred = self._clf.predict(X_te)
        accuracy = (y_pred == y_te).mean()
        report = classification_report(
            y_te, y_pred, target_names=self._le.classes_
        )
        importances = (
            pd.Series(
                self._clf.feature_importances_,
                index=self._feature_names(df),
            ).sort_values(ascending=True)
        )
        logger.info("Trained. Test accuracy=%.3f", accuracy)
        return MonitorResults(report, importances, self._le, accuracy)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score a single job's layer data.

        Returns the input DataFrame with added columns:
          - zscore_anomaly : bool   (any sensor exceeded 3-sigma rolling Z)
          - predicted_failure : str
          - health_score : float  (0=critical, 100=healthy)
          - alert_level : str     (OK / WARNING / CRITICAL)
        """
        if not self._trained:
            raise RuntimeError("Call fit() before score().")

        featured = self._build_features(df)
        X = featured[self._feature_names(df)].values

        enc = self._clf.predict(X)
        proba = self._clf.predict_proba(X)

        out = df.copy()
        out.loc[featured.index, "predicted_failure"] = self._le.inverse_transform(enc)
        good_idx = list(self._le.classes_).index("Normal")
        normal_prob = proba[:, good_idx]
        out.loc[featured.index, "health_score"] = (normal_prob * 100).round(1)

        # Fill first (window-1) rows that have no rolling stats
        out["predicted_failure"] = out["predicted_failure"].fillna("Normal")
        out["health_score"] = out["health_score"].fillna(100.0)

        # Z-score anomaly flag
        zscore_flag = self._zscore_anomaly(df)
        out["zscore_anomaly"] = zscore_flag

        out["alert_level"] = pd.cut(
            out["health_score"],
            bins=[-1, 40, 70, 101],
            labels=["CRITICAL", "WARNING", "OK"],
        )
        return out

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute rolling mean + std features over the sensor window."""
        feats = {}
        for col in SENSOR_COLS:
            if col not in df.columns:
                continue
            rolling = df[col].rolling(self.window, min_periods=self.window)
            feats[f"{col}_mean"] = rolling.mean()
            feats[f"{col}_std"]  = rolling.std().fillna(0)
        feat_df = pd.DataFrame(feats, index=df.index).dropna()
        return feat_df

    def _feature_names(self, df: pd.DataFrame) -> list[str]:
        names = []
        for col in SENSOR_COLS:
            if col in df.columns:
                names += [f"{col}_mean", f"{col}_std"]
        return names

    def save(self, path: str | Path) -> None:
        """Persist the trained monitor to disk."""
        if not self._trained:
            raise RuntimeError("Nothing to save — call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"clf": self._clf, "le": self._le, "window": self.window}, path)
        logger.info("Monitor saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "PrintMonitor":
        """Load a previously saved monitor from disk."""
        data = joblib.load(path)
        obj = cls.__new__(cls)
        obj._clf = data["clf"]
        obj._le = data["le"]
        obj.window = data["window"]
        obj._trained = True
        return obj

    def _zscore_anomaly(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.Series:
        """Flag layers where any sensor deviates > threshold sigma (rolling)."""
        flag = pd.Series(False, index=df.index)
        for col in SENSOR_COLS:
            if col not in df.columns:
                continue
            roll = df[col].rolling(30, min_periods=5)
            z = (df[col] - roll.mean()) / (roll.std().replace(0, np.nan))
            flag |= z.abs() > threshold
        return flag.fillna(False)
