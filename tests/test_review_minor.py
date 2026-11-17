"""Regression tests for the remaining (minor) bug-review findings, 2026-11:
stim channels to annotations, metadata alignment, export side effects,
ICA picks / component count, n_jobs, running_mean, config mutation."""

from datetime import datetime, timezone

import mne
import numpy as np
import pytest

from apice import RawAPICE
from apice.artifacts_detection import running_mean
from apice.artifacts_rejection import ArtifactsConfiguration, run_algorithms
from apice.ica import _fit_span, _picks_to_indices
from apice.pipeline import convert_stim_channels_to_annotations, generate_metadata_for_epochs
from conftest import quiet
from synthetic import SyntheticEEG


def _stim_raw(meas_date):
    sf = 100.0
    info = mne.create_info(["E1", "DIN1", "DIN2", "STI"], sf, ["eeg", "stim", "stim", "stim"])
    d = np.zeros((4, 3000))
    d[1, 500:510] = 1                 # DIN1 at sample 500
    d[2, 500:520] = 1                 # DIN2 at the same time (overlap)
    d[2, 1200:1210] = 1
    d[3, 800:805] = 3                 # a channel with several codes
    d[3, 900:905] = 7
    raw = mne.io.RawArray(d, info, first_samp=1000, verbose="ERROR")
    raw.set_meas_date(meas_date)
    return raw


@pytest.mark.parametrize("meas_date", [None, datetime(2020, 1, 1, tzinfo=timezone.utc)])
def test_stim_channels_to_annotations(meas_date):
    raw = _stim_raw(meas_date)
    quiet(convert_stim_channels_to_annotations, raw)
    ev, ids = mne.events_from_annotations(raw, verbose="ERROR")
    inv = {v: k for k, v in ids.items()}
    got = sorted((int(s - raw.first_samp), inv[c]) for s, _, c in ev)
    assert got == [(500, "DIN1"), (500, "DIN2"), (800, "STI/3"), (900, "STI/7"), (1200, "DIN2")]
    # running it again adds no duplicates
    quiet(convert_stim_channels_to_annotations, raw)
    assert len(raw.annotations) == 5


def test_metadata_has_one_row_per_segmentation_event():
    """Rows are matched to the segmentation events by sample (a safety net:
    make_metadata already requires the segmentation events among the
    metadata events)."""
    sf = 100.0
    raw = mne.io.RawArray(np.zeros((1, 6000)), mne.create_info(1, sf, "eeg"), verbose="ERROR")
    raw.set_annotations(mne.Annotations(
        [5.0, 15.0, 25.0, 35.0, 5.3, 25.3, 35.3], [0.0] * 7,
        ["shape/a", "shape/b", "shape/a", "shape/b", "resp/x", "resp/x", "resp/y"]))
    md = generate_metadata_for_epochs(
        raw,
        kwargs_events_from_annotations_for_metadata=dict(verbose="ERROR"),
        kwargs_events_from_annotations_for_segmentation=dict(regexp="^shape/", verbose="ERROR"),
        tmin=0.0, tmax=1.0)
    assert md["event_name"].tolist() == ["shape/a", "shape/b", "shape/a", "shape/b"]
    assert md["resp/x"].notna().tolist() == [True, False, True, False]


def test_raw_export_leaves_the_object_unchanged(tmp_path):
    raw, _ = SyntheticEEG(32, duration=10, seed=0).build()
    r = quiet(RawAPICE, raw)
    r.artifacts.BCT[2, 100:200] = True
    r.artifacts.BC[3, 0] = True
    n_ann, bads = len(r.annotations), list(r.info["bads"])
    quiet(r.export, tmp_path / "x-raw.fif", overwrite=True)
    assert len(r.annotations) == n_ann and r.info["bads"] == bads
    saved = mne.io.read_raw_fif(tmp_path / "x-raw.fif", verbose="ERROR")
    assert "artifact" in set(saved.annotations.description)


def test_ica_picks_given_as_a_type_string():
    info = mne.create_info(["E1", "E2", "EOG", "STI"], 100.0, ["eeg", "eeg", "eog", "stim"])
    raw = mne.io.RawArray(np.zeros((4, 100)), info, verbose="ERROR")
    assert _picks_to_indices(raw, "eeg") == [0, 1]
    assert _picks_to_indices(raw, ["eeg", "eog"]) == [0, 1, 2]
    assert _picks_to_indices(raw, ["E2", "EOG"]) == [1, 2]
    assert _picks_to_indices(raw, [3]) == [3]


def test_ica_fit_span():
    raw = mne.io.RawArray(np.zeros((1, 1000)), mne.create_info(1, 100.0, "eeg"), verbose="ERROR")
    assert _fit_span(raw, None, None) == (0, 1000)
    assert _fit_span(raw, 100, 300) == (100, 300)
    assert _fit_span(raw, 1.0, 3.0) == (100, 300)


@pytest.mark.parametrize("N", [0, 1])
def test_running_mean_window_below_two_samples(N):
    x = np.arange(10.0)
    np.testing.assert_array_equal(running_mean(x, N), x)


def test_run_algorithms_does_not_modify_the_user_config():
    raw, _ = SyntheticEEG(32, duration=10, seed=0).build()
    r = quiet(RawAPICE, raw)
    cfg = ArtifactsConfiguration()
    cfg.add_algorithm_group("g", max_loops=1, min_rejection=0, define_bcbt=False)
    cfg.add_algorithm("g", "Amplitude", {"thresh_type": "absolute", "thresh": 1.0}, algorithm_name="A")
    before = dict(cfg.cfg["g"]["algorithms"]["A"]["parameters"])
    quiet(run_algorithms, r, cfg.cfg)
    assert cfg.cfg["g"]["algorithms"]["A"]["parameters"] == before


# ------------------------------------------- detection study (2026-11) bugs
def _raw_apice():
    raw, _ = SyntheticEEG(32, duration=10, seed=0).build()
    return quiet(RawAPICE, raw)


def test_zero_threshold_is_a_real_threshold():
    """A threshold of 0 was treated as 'no threshold' (if thresh[i]:)."""
    from apice.artifacts_detection import reject_absolute, reject_relative, reject_relative_per_cha
    r = _raw_apice()
    x = r._data.copy()
    rej, _, _ = reject_absolute(x, [None, 0])
    assert 0.4 < rej.mean() < 0.6                      # about half the samples are > 0
    rej, _, _ = reject_relative(x, r.artifacts, [None, 0])
    assert 0.2 < rej.mean() < 0.3                      # above Q3
    rej, _, _ = reject_relative_per_cha(x, r.artifacts, [None, 0])
    assert 0.2 < rej.mean() < 0.3
    rej, _, _ = reject_absolute(x, [None, None])       # None still means no threshold
    assert not rej.any()


def test_default_artifact_detection_config_exists():
    """detect_artifacts(cfg=None) / run_algorithms(cfg=None) loaded missing files."""
    r = _raw_apice()
    quiet(r.detect_artifacts)
    quiet(run_algorithms, _raw_apice())
