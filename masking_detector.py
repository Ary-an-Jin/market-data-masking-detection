"""
nse_audit/masking_detector.py

Detects which institutional data masking protocols have been applied to a
client-provided dataset before the main audit runs.

Detectable protocols (from standard institutional sanitisation practice):
  1. Affine price scaling      P' = alpha * P + beta
  2. Log-scaled volume         V' = ln(1 + V) * gamma
  3. Information bars          Volume/dollar bars instead of time bars
  4. Cross-sectional z-score   Zi,t = (Xi,t - mu_t) / sigma_t  (multi-asset only)
  5. Epoch shift               T' = T + delta_global
  6. Timestamp jitter          T' = T + epsilon_noise (sub-millisecond)
  7. Column hashing            Feature_001 style column names
  8. Synthetic GAN data        Statistically near-perfect distributions

NOT detectable without a reference dataset:
  - Directional inversion (multiplying by -1) -- impossible to distinguish
    from genuine price decline without knowing the original series.

Entry point:
  profile = detect(df, time_col, price_col, volume_col,
                   raw_columns=None, session_hours=(9, 16))
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# Affine scaling
_KNOWN_TICK_SIZES  = {0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0, 10.0}
_AFFINE_TICK_TOL   = 0.01     # within 1% of a known tick size = clean
_AFFINE_MIN_DELTAS = 20       # minimum non-zero price deltas required

# Log-scaled volume
_LOG_VOL_SKEW_MAX  = 1.5      # genuine financial volume skewness is normally > 2
_LOG_VOL_MIN_ROWS  = 50

# Information bars
_INFO_TIME_CV_MIN  = 0.40     # CV of time deltas above this = suspiciously irregular
_INFO_VOL_CV_MAX   = 0.15     # CV of volume below this = suspiciously regular
_INFO_MIN_ROWS     = 30

# Cross-sectional z-score (multi-asset)
_ZSCORE_MEAN_TOL   = 0.05     # cross-sectional mean within +/- 0.05 of 0
_ZSCORE_STD_TOL    = 0.15     # cross-sectional std within +/- 0.15 of 1
_ZSCORE_MIN_ASSETS = 3

# Epoch shift
_EPOCH_IN_SESSION_MIN = 0.40  # at least 40% of bars must fall in session hours

# Timestamp jitter
_JITTER_US_STD_MIN = 200      # microsecond std above this = jitter present

# Column hashing
_HASHED_PATTERN    = re.compile(
    r'^(feature|col|field|x|f|dim|var|attr)_?\d+$', re.IGNORECASE
)
_HASHED_MIN_COLS   = 2

# Synthetic data
_SYNTH_MIN_ROWS         = 100
_SYNTH_KURTOSIS_MAX     = 0.8   # excess kurtosis below this is suspiciously normal


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ProtocolDetection:
    detected   : bool
    confidence : float   # 0.0 = no evidence, 1.0 = certain
    evidence   : str     # one-line human-readable explanation

    def __bool__(self) -> bool:
        return self.detected


@dataclass
class MaskingProfile:
    affine_scaled      : ProtocolDetection
    log_scaled_volume  : ProtocolDetection
    information_bars   : ProtocolDetection
    z_score_normalized : ProtocolDetection
    epoch_shifted      : ProtocolDetection
    timestamp_jitter   : ProtocolDetection
    column_hashed      : ProtocolDetection
    synthetic_data     : ProtocolDetection
    row_count          : int
    notes              : list[str] = field(default_factory=list)

    @property
    def any_detected(self) -> bool:
        return any(p.detected for _, p in self._named())

    def detected_names(self) -> list[str]:
        return [name for name, p in self._named() if p.detected]

    def summary_lines(self) -> list[str]:
        lines = []
        for name, p in self._named():
            tag  = "DETECTED" if p.detected else "not detected"
            conf = f" [{p.confidence:.0%} confidence]" if p.detected else ""
            lines.append(f"  {name:<26}: {tag}{conf}")
            lines.append(f"  {'':26}  {p.evidence}")
        return lines

    def _named(self) -> list[tuple[str, ProtocolDetection]]:
        return [
            ("affine_scaling",     self.affine_scaled),
            ("log_scaled_volume",  self.log_scaled_volume),
            ("information_bars",   self.information_bars),
            ("z_score_norm",       self.z_score_normalized),
            ("epoch_shift",        self.epoch_shifted),
            ("timestamp_jitter",   self.timestamp_jitter),
            ("column_hashing",     self.column_hashed),
            ("synthetic_data",     self.synthetic_data),
        ]


# ---------------------------------------------------------------------------
# Individual detection functions
# ---------------------------------------------------------------------------

def _detect_affine_scaling(df: pd.DataFrame, price_col: str) -> ProtocolDetection:
    """
    Affine-scaled prices have a minimum tick size that is not a clean fraction.
    Real NSE equity prices move in multiples of 0.05 or 0.01.
    """
    if price_col not in df.columns:
        return ProtocolDetection(False, 0.0, f"Column '{price_col}' not found")

    deltas = df[price_col].diff().abs().dropna()
    deltas = deltas[deltas > 0]

    if len(deltas) < _AFFINE_MIN_DELTAS:
        return ProtocolDetection(False, 0.0,
            f"Insufficient price movements to analyse ({len(deltas)} non-zero deltas)")

    min_tick = float(deltas.min())

    # Check if min_tick is within tolerance of any known tick size
    for known in _KNOWN_TICK_SIZES:
        if abs(min_tick - known) / known < _AFFINE_TICK_TOL:
            return ProtocolDetection(False, 0.0,
                f"Min tick {min_tick:.6f} matches known tick size {known}")

    # Check if min_tick is a valid sub-division of a known tick size.
    # Logic: known / min_tick must be a small clean integer from the valid set.
    # e.g. min_tick=0.025, known=0.05 -> ratio=2.0 (valid) -> clean tick.
    # Guard: only check when min_tick < known (can't be a sub-multiple of smaller value).
    _VALID_MULTIPLIERS = {2, 4, 5, 10, 20, 25, 50, 100}
    for known in _KNOWN_TICK_SIZES:
        if min_tick >= known:
            continue
        ratio   = known / min_tick
        rounded = round(ratio)
        if (rounded in _VALID_MULTIPLIERS
                and abs(ratio - rounded) / rounded < _AFFINE_TICK_TOL):
            return ProtocolDetection(False, 0.0,
                f"Min tick {min_tick:.6f} is a valid sub-division of {known} "
                f"(1/{rounded})")

    # Confidence: how far is min_tick from the nearest known tick (as a fraction)?
    # Distance > 10% from the nearest known tick = high confidence it's been scaled.
    min_rel_dist = min(abs(min_tick - k) / k for k in _KNOWN_TICK_SIZES)
    confidence   = min(0.95, 0.5 + 0.5 * min(1.0, min_rel_dist / 0.10))
    return ProtocolDetection(True, confidence,
        f"Min price movement is {min_tick:.8f} -- not a recognised tick size; "
        f"affine scaling (P' = alpha*P + beta) likely applied")


def _detect_log_scaled_volume(df: pd.DataFrame, volume_col: str) -> ProtocolDetection:
    """
    Genuine financial volume follows a power-law with high right skewness (> 2).
    Log-transformation compresses the tail, reducing skewness to near 0.
    """
    if volume_col not in df.columns:
        return ProtocolDetection(False, 0.0, f"Column '{volume_col}' not found")

    vol = df[volume_col].dropna()
    vol = vol[vol > 0]

    if len(vol) < _LOG_VOL_MIN_ROWS:
        return ProtocolDetection(False, 0.0,
            f"Insufficient volume rows ({len(vol)}) for skewness analysis")

    skewness = float(vol.skew())

    if skewness < _LOG_VOL_SKEW_MAX:
        confidence = min(0.95, max(0.5, 1.0 - (skewness / _LOG_VOL_SKEW_MAX)))
        return ProtocolDetection(True, confidence,
            f"Volume skewness is {skewness:.2f} (expected > 2 for raw financial volume); "
            f"logarithmic transformation likely applied")

    return ProtocolDetection(False, 0.0,
        f"Volume skewness is {skewness:.2f} -- consistent with natural power-law distribution")


def _detect_information_bars(
    df: pd.DataFrame, time_col: str, volume_col: str
) -> ProtocolDetection:
    """
    Time bars have irregular volume but regular time gaps.
    Volume/dollar bars have regular volume but irregular time gaps.
    High CV(time_deltas) + low CV(volume) = information bars.
    """
    if time_col not in df.columns or volume_col not in df.columns:
        return ProtocolDetection(False, 0.0, "Required columns not found")

    if len(df) < _INFO_MIN_ROWS:
        return ProtocolDetection(False, 0.0,
            f"Insufficient rows ({len(df)}) for information bar detection")

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        return ProtocolDetection(False, 0.0, "Timestamp column is not datetime type")

    time_deltas = df[time_col].diff().dropna().dt.total_seconds()
    time_deltas  = time_deltas[time_deltas > 0]
    vol          = df[volume_col].dropna()
    vol          = vol[vol > 0]

    if time_deltas.mean() == 0 or vol.mean() == 0:
        return ProtocolDetection(False, 0.0, "Zero mean in time deltas or volume")

    time_cv = float(time_deltas.std() / time_deltas.mean())
    vol_cv  = float(vol.std() / vol.mean())

    if time_cv > _INFO_TIME_CV_MIN and vol_cv < _INFO_VOL_CV_MAX:
        confidence = min(0.95,
            0.5 * min(1.0, time_cv / (_INFO_TIME_CV_MIN * 2)) +
            0.5 * min(1.0, _INFO_VOL_CV_MAX / max(vol_cv, 1e-6)))
        return ProtocolDetection(True, confidence,
            f"Time delta CV={time_cv:.2f} (high) and volume CV={vol_cv:.3f} (low); "
            f"dataset appears to use volume/dollar bars rather than time bars")

    return ProtocolDetection(False, 0.0,
        f"Time delta CV={time_cv:.2f}, volume CV={vol_cv:.2f} -- consistent with time bars")


def _detect_z_score_normalization(
    df: pd.DataFrame, time_col: str, price_col: str, asset_col: str = "asset_id"
) -> ProtocolDetection:
    """
    Cross-sectional z-scoring: at each timestamp, the mean across assets is 0
    and the standard deviation is 1.  Only applicable to multi-asset panels.
    """
    if asset_col not in df.columns:
        return ProtocolDetection(False, 0.0,
            "No asset_id column -- single-asset data, z-score norm not applicable")

    n_assets = df[asset_col].nunique()
    if n_assets < _ZSCORE_MIN_ASSETS:
        return ProtocolDetection(False, 0.0,
            f"Only {n_assets} asset(s) -- need >= {_ZSCORE_MIN_ASSETS} for cross-sectional check")

    if time_col not in df.columns or price_col not in df.columns:
        return ProtocolDetection(False, 0.0, "Required columns not found")

    grouped      = df.groupby(time_col)[price_col]
    cross_mean   = float(grouped.mean().mean())
    cross_std    = float(grouped.std().mean())

    mean_ok = abs(cross_mean) < _ZSCORE_MEAN_TOL
    std_ok  = abs(cross_std - 1.0) < _ZSCORE_STD_TOL

    if mean_ok and std_ok:
        conf = 1.0 - (abs(cross_mean) / _ZSCORE_MEAN_TOL + abs(cross_std - 1.0) / _ZSCORE_STD_TOL) / 2
        confidence = min(0.95, max(0.6, conf))
        return ProtocolDetection(True, confidence,
            f"Cross-sectional mean={cross_mean:.4f} (near 0), std={cross_std:.4f} (near 1) "
            f"across {n_assets} assets -- z-score normalization detected")

    return ProtocolDetection(False, 0.0,
        f"Cross-sectional mean={cross_mean:.4f}, std={cross_std:.4f} -- not z-score normalized")


def _detect_epoch_shift(
    df: pd.DataFrame, time_col: str,
    session_start_hour: int = 9, session_end_hour: int = 16
) -> ProtocolDetection:
    """
    If timestamps are globally shifted, bars that should be in session hours
    (09:15--15:30 for NSE) will appear at wrong times of day.
    """
    if time_col not in df.columns:
        return ProtocolDetection(False, 0.0, f"Column '{time_col}' not found")

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        return ProtocolDetection(False, 0.0, "Timestamp column is not datetime type")

    hours       = df[time_col].dt.hour
    in_session  = float(((hours >= session_start_hour) & (hours < session_end_hour)).mean())

    if in_session < _EPOCH_IN_SESSION_MIN:
        confidence = min(0.95, 1.0 - in_session)
        return ProtocolDetection(True, confidence,
            f"Only {in_session:.0%} of bars fall within expected session hours "
            f"({session_start_hour}:00-{session_end_hour}:00) -- global epoch shift likely applied")

    return ProtocolDetection(False, 0.0,
        f"{in_session:.0%} of bars within session hours -- timestamps appear unshifted")


def _detect_timestamp_jitter(df: pd.DataFrame, time_col: str) -> ProtocolDetection:
    """
    Jitter adds sub-millisecond noise to timestamps.
    Real market data typically has clean millisecond or second boundaries.
    """
    if time_col not in df.columns:
        return ProtocolDetection(False, 0.0, f"Column '{time_col}' not found")

    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        return ProtocolDetection(False, 0.0, "Timestamp column is not datetime type")

    microseconds = df[time_col].dt.microsecond
    us_std = float(microseconds.std())

    # Check nanosecond component too (pandas datetime64[ns])
    try:
        nanoseconds = df[time_col].astype("int64") % 1_000
        ns_std = float(nanoseconds.std())
    except Exception:
        ns_std = 0.0

    if us_std > _JITTER_US_STD_MIN:
        confidence = min(0.90, us_std / 500)
        return ProtocolDetection(True, confidence,
            f"Sub-second timestamp variation: microsecond std={us_std:.0f} -- jitter detected")

    if ns_std > 50:
        return ProtocolDetection(True, 0.60,
            f"Sub-microsecond timestamp noise: nanosecond std={ns_std:.1f} -- possible jitter")

    return ProtocolDetection(False, 0.0,
        f"Timestamps have clean boundaries (microsecond std={us_std:.0f})")


def _detect_column_hashing(raw_columns: list[str]) -> ProtocolDetection:
    """
    Hashed columns look like Feature_001, col_02, dim_12, etc.
    Two or more such columns suggest deliberate renaming.
    """
    hashed = [c for c in raw_columns if _HASHED_PATTERN.match(str(c))]
    n      = len(hashed)
    total  = len(raw_columns)

    if n >= _HASHED_MIN_COLS:
        confidence = min(0.95, n / total)
        return ProtocolDetection(True, confidence,
            f"{n}/{total} columns have hashed names: {hashed[:5]}")

    return ProtocolDetection(False, 0.0,
        "Column names appear standard (no Feature_NNN pattern detected)")


def _detect_synthetic_data(df: pd.DataFrame, price_col: str) -> ProtocolDetection:
    """
    GAN-generated data often has near-normal return distributions.
    Real financial returns have fat tails (excess kurtosis > 1, typically > 3).
    Low excess kurtosis is a weak signal; flagged with low confidence.

    Note: this test has meaningful false-positive risk on small or trending datasets.
    Treat as a flag to investigate, not a conclusion.
    """
    if price_col not in df.columns or len(df) < _SYNTH_MIN_ROWS:
        return ProtocolDetection(False, 0.0,
            f"Insufficient rows ({len(df)}) for synthetic data detection")

    returns        = df[price_col].pct_change().dropna()
    excess_kurt    = float(returns.kurt())   # 0 = normal, >0 = fat tails

    if excess_kurt < _SYNTH_KURTOSIS_MAX:
        return ProtocolDetection(True, 0.55,
            f"Return distribution excess kurtosis={excess_kurt:.2f} (near-normal); "
            f"real market returns are typically fat-tailed (>1). "
            f"Possible GAN/synthetic data -- treat as a weak signal only.")

    return ProtocolDetection(False, 0.0,
        f"Return excess kurtosis={excess_kurt:.2f} -- fat tails present, consistent with real data")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def detect(
    df            : pd.DataFrame,
    time_col      : str,
    price_col     : str,
    volume_col    : str,
    raw_columns   : list[str] | None = None,
    asset_col     : str = "asset_id",
    session_hours : tuple[int, int] = (9, 16),
) -> MaskingProfile:
    """
    Run all masking protocol detectors on a client-provided DataFrame.

    Parameters
    ----------
    df            : The loaded client dataset.
    time_col      : Name of the timestamp column (must be datetime64).
    price_col     : Name of the price column to analyse (e.g. 'close').
    volume_col    : Name of the volume column.
    raw_columns   : Original column names before any resolver mapping.
                    Defaults to df.columns if None.
    asset_col     : Column identifying individual assets (multi-asset panels only).
    session_hours : (start_hour, end_hour) for epoch-shift detection.
                    Default (9, 16) covers NSE 09:15--15:30 IST.

    Returns
    -------
    MaskingProfile with detection result for each protocol.
    """
    if raw_columns is None:
        raw_columns = list(df.columns)

    session_start, session_end = session_hours

    notes: list[str] = [
        "Directional inversion (sign flipping) is not detectable without a "
        "reference dataset and is therefore not assessed.",
    ]

    profile = MaskingProfile(
        affine_scaled      = _detect_affine_scaling(df, price_col),
        log_scaled_volume  = _detect_log_scaled_volume(df, volume_col),
        information_bars   = _detect_information_bars(df, time_col, volume_col),
        z_score_normalized = _detect_z_score_normalization(df, time_col, price_col, asset_col),
        epoch_shifted      = _detect_epoch_shift(df, time_col, session_start, session_end),
        timestamp_jitter   = _detect_timestamp_jitter(df, time_col),
        column_hashed      = _detect_column_hashing(raw_columns),
        synthetic_data     = _detect_synthetic_data(df, price_col),
        row_count          = len(df),
        notes              = notes,
    )

    return profile
