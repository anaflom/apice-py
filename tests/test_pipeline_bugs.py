"""Regression tests for pipeline-level bugs and the detection-loop stopping rule.

Bug-review findings on pipeline.py / filter.py and #9 (2026-11).
"""

import mne
import numpy as np
import pytest

from apice import RawAPICE
from apice.artifacts_rejection import ArtifactsConfiguration, run_algorithms
from apice.filter import Filter
from apice.pipeline import _normalize_evoked_by, get_stim_duration, preprocess_initial_steps
from conftest import quiet
from synthetic import SyntheticEEG


@pytest.mark.parametrize("value, expected", [
    (True, "all"), ("all", "all"), (None, None), (False, None),
    ("tone", ["tone"]), (["a", "b"], ["a", "b"]), (("a",), ["a"]),
])
def test_normalize_evoked_by(value, expected):
    """evoked_by=True (the old run_segmentation default) crashed with `for ev in True`."""
    assert _normalize_evoked_by(value) == expected


def test_get_stim_duration():
    """get_onset_and_duration was called without its time vector."""
    sf = 250.0
    info = mne.create_info(["E0", "STI1", "STI2"], sf, ["eeg", "stim", "stim"])
    data = np.zeros((3, 2500))
    data[1, 100:150] = 1      # 0.2 s pulse
    data[2, 1000:1025] = 1    # 0.1 s pulse
    raw = mne.io.RawArray(data, info, verbose=False)
    assert get_stim_duration(raw) == pytest.approx(0.2)


def test_lowpass_at_or_above_nyquist_is_skipped():
    """The default 100 Hz low-pass crashed on data sampled at 200 Hz."""
    raw, _ = SyntheticEEG(32, duration=10, seed=0, sfreq=200.0).build()
    before = raw.get_data().copy()
    quiet(Filter, raw, h_freq=100, l_freq=None, n_jobs=1)
    np.testing.assert_array_equal(raw.get_data(), before)


def test_initial_steps_apply_crop_and_pick_to_rawapice():
    """Return values of crop/pick/resample were ignored, so they did nothing on RawAPICE."""
    raw, _ = SyntheticEEG(32, duration=20, seed=0).build()
    r = quiet(RawAPICE, raw)
    picks = r.ch_names[:20]
    out, _ = quiet(preprocess_initial_steps, r, picks=picks, crop_times=(2.0, 12.0),
                   resample_freq=125, l_freq=None, h_freq=None, create_report=False,
                   save_report=False, save_cfg=False, save_log=False, n_jobs=1)
    assert out.ch_names == picks
    assert out.info["sfreq"] == 125
    assert out.times[-1] == pytest.approx(10.0, abs=0.02)


def _glitch_raw():
    s = SyntheticEEG(32, duration=20, seed=4)
    raw, _ = s.build()
    raw._data[5, 1000:1010] += 1e-3   # one clear 40 ms glitch
    return quiet(RawAPICE, raw)


def test_detections_of_the_stopping_loop_are_kept():
    """#9: the loop that fell below min_rejection had its detections discarded;
    with min_loops=0 that could be the first loop, leaving nothing marked."""
    r = _glitch_raw()
    cfg = ArtifactsConfiguration()
    cfg.add_algorithm_group("g", min_loops=0, max_loops=3, min_rejection=50)  # 50 %: stops at loop 1
    cfg.add_algorithm("g", "Amplitude", {"thresh_type": "absolute",
                                         "thresh": [-500e-6, 500e-6],
                                         "update_artifacts": False})
    quiet(run_algorithms, r, cfg=cfg.cfg)
    bct = np.asarray(r.artifacts.BCT, bool)
    assert bct[5, 1000:1010].all()
