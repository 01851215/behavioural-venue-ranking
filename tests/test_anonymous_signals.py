import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from compute_anonymous_venue_signals import (
    compute_checkin_burstiness,
    compute_peak_hour_entropy,
    compute_weekday_ratio,
    compute_temporal_stability_cv,
    compute_visit_velocity_recent,
    compute_growth_trend,
    compute_lunch_dinner_ratio,
    compute_late_night_ratio,
    compute_peak_hour_mode,
    extract_venue_features,
)


def make_timestamps(hours, days_offset=None):
    base = pd.Timestamp("2020-01-06")  # Monday
    if days_offset is None:
        days_offset = list(range(len(hours)))
    return [base + pd.Timedelta(days=d, hours=h) for d, h in zip(days_offset, hours)]


def test_burstiness_uniform_is_low():
    ts = make_timestamps([9] * 10, list(range(10)))
    b = compute_checkin_burstiness(ts)
    assert b < 0.5, f"Expected low burstiness for uniform, got {b}"

def test_burstiness_spiky_is_high():
    ts = make_timestamps([9] * 10, [0, 0, 0, 0, 0, 10, 20, 30, 40, 50])
    b = compute_checkin_burstiness(ts)
    assert b > 0.5, f"Expected high burstiness for spiky, got {b}"

def test_burstiness_single_day_is_nan():
    ts = make_timestamps([9], [0])
    b = compute_checkin_burstiness(ts)
    assert np.isnan(b)

def test_burstiness_single_day_multi_visit_is_nan():
    # Multiple visits all on the same day — still NaN (CV undefined for 1 active day)
    ts = make_timestamps([9, 10, 11], [0, 0, 0])
    b = compute_checkin_burstiness(ts)
    assert np.isnan(b)

def test_entropy_uniform_is_high():
    ts = make_timestamps(list(range(24)), list(range(24)))
    e = compute_peak_hour_entropy(ts)
    assert e > 4.0, f"Expected high entropy for uniform hours, got {e}"

def test_entropy_single_hour_is_zero():
    ts = make_timestamps([9] * 20, list(range(20)))
    e = compute_peak_hour_entropy(ts)
    assert e == pytest.approx(0.0, abs=1e-6)

def test_weekday_ratio_all_weekdays():
    ts = make_timestamps([9] * 5, [0, 1, 2, 3, 4])
    assert compute_weekday_ratio(ts) == pytest.approx(1.0)

def test_weekday_ratio_all_weekend():
    ts = make_timestamps([9] * 2, [5, 6])
    assert compute_weekday_ratio(ts) == pytest.approx(0.0)

def test_weekday_ratio_mixed():
    ts = make_timestamps([9] * 5, [0, 1, 2, 5, 6])
    assert compute_weekday_ratio(ts) == pytest.approx(0.6)

def test_lunch_dinner_ratio_all_lunch():
    ts = make_timestamps([12] * 10, list(range(10)))
    assert compute_lunch_dinner_ratio(ts) == pytest.approx(1.0)

def test_lunch_dinner_ratio_none():
    ts = make_timestamps([3] * 10, list(range(10)))
    assert compute_lunch_dinner_ratio(ts) == pytest.approx(0.0)

def test_late_night_ratio_all_late():
    ts = make_timestamps([23] * 10, list(range(10)))
    assert compute_late_night_ratio(ts) == pytest.approx(1.0)

def test_late_night_ratio_none():
    ts = make_timestamps([12] * 10, list(range(10)))
    assert compute_late_night_ratio(ts) == pytest.approx(0.0)

def test_peak_hour_mode():
    ts = make_timestamps([9, 9, 9, 12, 15], list(range(5)))
    assert compute_peak_hour_mode(ts) == 9

def test_velocity_all_recent():
    ref = pd.Timestamp("2020-12-31")
    ts = [ref - pd.Timedelta(days=d) for d in range(30)]
    assert compute_visit_velocity_recent(ts, ref) == pytest.approx(1.0)

def test_velocity_none_recent():
    ref = pd.Timestamp("2020-12-31")
    ts = [ref - pd.Timedelta(days=400 + d) for d in range(30)]
    assert compute_visit_velocity_recent(ts, ref) == pytest.approx(0.0)

def test_growth_trend_increasing():
    ref = pd.Timestamp("2021-01-01")
    ts = []
    for month_offset in range(6):
        date = ref - pd.DateOffset(months=5 - month_offset)
        n = (month_offset + 1) * 5
        ts.extend([date + pd.Timedelta(days=i) for i in range(n)])
    slope = compute_growth_trend(ts)
    assert slope > 0, f"Expected positive slope for increasing trend, got {slope}"

def test_growth_trend_single_month_is_nan():
    ts = [pd.Timestamp("2020-06-15")] * 5
    assert np.isnan(compute_growth_trend(ts))

def test_extract_venue_features_returns_all_keys():
    ts = make_timestamps(list(range(10)) * 3, list(range(30)))
    ref = pd.Timestamp("2021-01-01")
    result = extract_venue_features(ts, reference_date=ref)
    expected_keys = {
        "total_checkins", "checkin_burstiness", "peak_hour_entropy",
        "weekday_ratio", "temporal_stability_cv", "visit_velocity_recent",
        "growth_trend", "lunch_dinner_ratio", "late_night_ratio", "peak_hour_mode",
    }
    assert set(result.keys()) == expected_keys

def test_extract_venue_features_total_checkins():
    ts = make_timestamps([9] * 7, list(range(7)))
    result = extract_venue_features(ts, reference_date=pd.Timestamp("2021-01-01"))
    assert result["total_checkins"] == 7
