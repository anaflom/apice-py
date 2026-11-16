"""Regression tests for crashes and data corruption in artifact detection.

Bug-review findings #4, #15, #16, #17 and #22 (2026-11).
"""

import warnings

import mne
import numpy as np
import pytest

from apice import EpochsAPICE, RawAPICE
from apice.artifacts_detection import (Amplitude, ChannelCorr, CrossElectrodesOutlier,
                                       FlatChannel, Power, reject_data)
from conftest import quiet
from synthetic import SyntheticEEG


def make_raw(n_channels=32, duration=20.0, flat=None, seed=3):
    s = SyntheticEEG(n_channels, duration=duration, seed=seed)
    if flat is not None:
        s.add_flat(s.ch_names[flat], value=0.0)
    raw, _ = s.build()
    return quiet(RawAPICE, raw)


def make_epochs(n_channels=32, duration=40.0, epoch_len=2.2, seed=3):
    s = SyntheticEEG(n_channels, duration=duration, seed=seed)
    raw, _ = s.build()
    ep = mne.make_fixed_length_epochs(raw, duration=epoch_len, preload=True, verbose=False)
    return quiet(EpochsAPICE, ep)


def run(alg, inst):
    quiet(alg.compute, inst)
    return quiet(alg.reject, inst)


def test_zscore_keeps_flat_channel_finite():
    """#4: z-scoring a flat channel (sd = 0) turned it into NaN permanently."""
    r = make_raw(flat=2)
    before = r.get_data().copy()
    run(CrossElectrodesOutlier(), r)
    after = r.get_data()
    assert np.isfinite(after).all()
    np.testing.assert_allclose(after, before, rtol=1e-10, atol=1e-15)


def test_zscore_does_not_silence_warnings_globally():
    """#4: _compute_z_score called warnings.filterwarnings('ignore')."""
    r = make_raw()
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        n_filters = len(warnings.filters)
        run(CrossElectrodesOutlier(), r)
        assert len(warnings.filters) == n_filters
        assert not any(f[0] == "ignore" and f[2] is Warning for f in warnings.filters)


def test_reference_on_epochs_runs_and_restores_data():
    """#15: do_reference_data=True on epochs raised IndexError."""
    ep = make_epochs()
    before = ep.get_data().copy()
    run(Amplitude(do_reference_data=True), ep)
    np.testing.assert_allclose(ep.get_data(), before, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("thresh", [2.5, np.float64(2.5), np.int64(3)])
def test_cross_electrodes_outlier_accepts_single_threshold(thresh):
    """#16 and #22: the default scalar threshold raised 'two thresholds are required'."""
    r = make_raw()
    _, bct = run(CrossElectrodesOutlier(thresh=thresh), r)
    assert np.asarray(bct).shape[-1] == r.n_times


def test_cross_electrodes_outlier_window_longer_than_data():
    """A window covering the whole recording built a 1-D index and crashed."""
    r = make_raw(duration=2.0)
    run(CrossElectrodesOutlier(time_window=5.0), r)


@pytest.mark.parametrize("alg", [FlatChannel, Power, ChannelCorr])
def test_long_window_detectors_on_short_raw(alg):
    """#17: FlatChannel (10 s window) crashed on 9 s of data."""
    r = make_raw(duration=9.0)
    run(alg(), r)


@pytest.mark.parametrize("alg", [FlatChannel, Power, ChannelCorr])
def test_long_window_detectors_on_short_epochs(alg):
    """#17: the same on 2.2 s epochs."""
    ep = make_epochs()
    run(alg(), ep)


def test_numpy_scalar_threshold_matches_python_scalar():
    """#22: np.float64 / np.int64 thresholds raised TypeError."""
    r = make_raw()
    data = np.abs(r.get_data()) / np.abs(r.get_data()).std()
    ref = quiet(reject_data, data, r, 3.0, rejection="outliers_per_channel", iq_half=True)[0]
    for t in (np.float64(3.0), np.int64(3)):
        got = quiet(reject_data, data, r, t, rejection="outliers_per_channel", iq_half=True)[0]
        np.testing.assert_array_equal(np.asarray(got), np.asarray(ref))
