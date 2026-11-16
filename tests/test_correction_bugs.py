"""Regression tests for artifact-correction bugs.

Bug-review findings #1, #10 and #21 (2026-11).
"""

import numpy as np

from apice import RawAPICE
from apice.artifacts_correction import (SegmentSphericalSplineInterpolation, _modify_rej,
                                        _splice_segments)
from conftest import quiet
from synthetic import SyntheticEEG


def test_splice_is_done_within_each_epoch():
    """#1: every epoch was aligned to the end of the previous one."""
    n = 100
    d = np.zeros(2 * n)
    d[:n] = 0.0
    d[20:30] = 10.0      # corrected segment in epoch 0
    d[n:] = 5.0          # epoch 1: clean, different level
    epoch_if = np.array([[0, n], [n, 2 * n]])
    bad_if = np.array([[20, 30]])
    out = np.squeeze(_splice_segments(d, bad_if, epoch_if, method=1))
    np.testing.assert_array_equal(out[n:], 5.0)        # epoch 1 untouched
    assert out[20] == out[19]                          # segment aligned within epoch 0


def test_modify_rej_on_data_shorter_than_one_segment():
    """#21: IndexError when the data were shorter than one segment."""
    rej = np.zeros((3, 50), dtype=int)
    rej[1, :30] = 1
    out = _modify_rej(rej, segment_samples=100, max_samples=10)
    assert out.shape == rej.shape
    assert out[1].all() and not out[[0, 2]].any()


def test_modify_rej_segments_unchanged_for_long_data():
    rng = np.random.default_rng(0)
    rej = (rng.random((4, 1050)) < 0.05).astype(int)
    out = _modify_rej(rej, segment_samples=100, max_samples=4)
    for s in range(0, 1050, 100):
        blk = rej[:, s:s + 100].sum(axis=1) > 4
        np.testing.assert_array_equal(out[:, s:s + 100].all(axis=1), blk)
        np.testing.assert_array_equal(out[:, s:s + 100].any(axis=1), blk)


def test_segment_spline_without_save_corrected_keeps_bct():
    """#10: with save_corrected=False the BCT mask was still cleared in place."""
    s = SyntheticEEG(32, duration=20, seed=1)
    s.add_motion("C3", onset=8.0, duration=1.0, radius=0.0)
    raw, truth = s.build()
    r = quiet(RawAPICE, raw)
    r.artifacts.BCT[:] = truth.mask
    quiet(r.define_bcbt)
    before = np.asarray(r.artifacts.BCT).copy()
    assert before.any()
    quiet(SegmentSphericalSplineInterpolation(save_corrected=False).correct, r)
    np.testing.assert_array_equal(np.asarray(r.artifacts.BCT), before)
