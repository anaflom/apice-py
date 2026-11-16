"""Regression tests for RawAPICE / EpochsAPICE bookkeeping bugs.

Bug-review findings #6, #7, #18, #19 and #20 (2026-11).
"""

import mne
import numpy as np
import pytest

from apice import EpochsAPICE, RawAPICE
from conftest import quiet
from synthetic import SyntheticEEG

SF = 250.0


def plain_raw(n_channels=8, n_samples=5000, first_samp=0, seed=0):
    rng = np.random.default_rng(seed)
    info = mne.create_info([f"E{i}" for i in range(n_channels)], SF, "eeg")
    return mne.io.RawArray(rng.normal(0, 1e-5, (n_channels, n_samples)), info,
                           first_samp=first_samp, verbose=False)


def synthetic_raw(duration=30.0, first_samp=0):
    raw, _ = SyntheticEEG(32, duration=duration, seed=2, first_samp=first_samp).build()
    return raw


# ---------------------------------------------------------------- #6
@pytest.mark.parametrize("first_samp", [0, 1000])
def test_annotations_without_meas_date_keep_their_samples(first_samp):
    raw = plain_raw(first_samp=first_samp)
    assert raw.info["meas_date"] is None
    raw.set_annotations(mne.Annotations([500 / SF], [100 / SF], ["artifact"], ch_names=[["E3"]]))
    r = quiet(RawAPICE, raw)
    bad = np.flatnonzero(np.asarray(r.artifacts.BCT)[3])
    assert bad.min() == 500 and bad.max() == 599


# ---------------------------------------------------------------- #19
def test_artifact_annotation_without_channels_marks_all_channels():
    raw = plain_raw()
    raw.set_annotations(mne.Annotations([2.0], [0.4], ["artifact"]))
    r = quiet(RawAPICE, raw)
    bct = np.asarray(r.artifacts.BCT)
    assert bct[:, 500:600].all()
    assert not bct[:, :500].any()


def test_artifact_annotation_with_two_channels_marks_both():
    raw = plain_raw()
    raw.set_annotations(mne.Annotations([2.0], [0.4], ["corrected"], ch_names=[["E1", "E5"]]))
    r = quiet(RawAPICE, raw)
    cct = np.asarray(r.artifacts.CCT)
    assert cct[[1, 5], 500:600].all()
    assert not cct[[0, 2, 3, 4, 6, 7]].any()


# ---------------------------------------------------------------- #18
def test_reference_channels_raw_with_two_references():
    r = quiet(RawAPICE, plain_raw())
    r.artifacts.BT[0, 100:200] = True
    r.artifacts.BCT[2, :] = True
    r.deal_with_reference_channels(["E2", "E4"])
    bct = np.asarray(r.artifacts.BCT)
    for ch in (2, 4):
        assert bct[ch, 100:200].all()
        assert not bct[ch, :100].any() and not bct[ch, 200:].any()


def test_reference_channels_epochs_with_two_references():
    ep = mne.make_fixed_length_epochs(plain_raw(), duration=2.0, preload=True, verbose=False)
    e = quiet(EpochsAPICE, ep)
    e.artifacts.BT[1, 0, 50:80] = True
    e.deal_with_reference_channels(["E2", "E4"])
    bct = np.asarray(e.artifacts.BCT)
    assert bct[1][[2, 4]][:, 50:80].all()
    assert not bct[0][[2, 4]].any()


# ---------------------------------------------------------------- #20
def test_epochs_export_keeps_artifacts_attribute(tmp_path):
    ep = mne.make_fixed_length_epochs(plain_raw(), duration=2.0, preload=True, verbose=False)
    e = quiet(EpochsAPICE, ep)
    quiet(e.export, tmp_path / "x-epo.fif")
    assert hasattr(e, "artifacts")
    # a refused export (file exists, overwrite=False) must not lose it either
    with pytest.raises(FileExistsError):
        quiet(e.export, tmp_path / "x-epo.fif")
    assert hasattr(e, "artifacts")


# ---------------------------------------------------------------- #7
def _segment(r, **kw):
    events = mne.make_fixed_length_events(r, id=1, start=1.0, duration=2.0)
    kwargs = dict(tmin=-0.2, tmax=0.8, baseline=None, verbose=False)
    kwargs.update(kw)
    return events, quiet(r.segment_continuous_data, events, {"stim": 1}, epoching_kwargs=kwargs)


@pytest.mark.parametrize("first_samp", [0, 700])
@pytest.mark.parametrize("decim", [1, 2])
def test_segment_copies_bct_and_bt_at_the_right_samples(first_samp, decim):
    r = quiet(RawAPICE, synthetic_raw(first_samp=first_samp))
    events = mne.make_fixed_length_events(r, id=1, start=1.0, duration=2.0)
    k = 3
    onset = events[k, 0] - r.first_samp
    r.artifacts.BCT[5, onset + 60] = True               # 0.24 s after the event
    r.artifacts.BT[0, onset + 100:onset + 110] = True    # 0.40-0.44 s after the event
    _, e = _segment(r, decim=decim, preload=True)
    t = e.times
    i_bct = np.argmin(np.abs(t - 0.24))
    bct = np.asarray(e.artifacts.BCT)
    assert bct[k, 5, i_bct]
    assert bct.sum() == 1
    bt = np.asarray(e.artifacts.BT)[k, 0]
    expected = (t >= 0.40 - 1e-9) & (t < 0.44 - 1e-9)
    np.testing.assert_array_equal(bt, expected)


def test_segment_with_preload_false():
    r = quiet(RawAPICE, synthetic_raw())
    events, e = _segment(r, preload=False)
    assert len(e) == len(events) - 1 or len(e) == len(events)
