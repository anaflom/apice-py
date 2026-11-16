"""Segment-wise spherical spline correction: no discontinuities at segment edges.

Detection is bypassed: the ground-truth mask is written into BCT so these tests
only exercise the correction + realignment ("splicing") step.

To make missing realignment visible, the corrupted channels also carry a
channel-specific DC offset over the whole recording (as un-high-passed EEG
does).  The spline estimate comes from neighbours that do not share this
offset, so without realignment every segment edge would show a jump of about
``OFFSET`` volts.
"""

import numpy as np
import pytest

from apice import RawAPICE
from apice.standard_conf import cfg_correction_spline_segments
from conftest import quiet
from synthetic import SyntheticEEG

OFFSET = 200e-6  # >> sample-to-sample steps of the background


def _edges(mask_1d):
    """Sample indices b where mask changes between b-1 and b."""
    return np.where(np.diff(mask_1d.astype(int)) != 0)[0] + 1


def _setup(placement, seed=0):
    s = SyntheticEEG(32, duration=20, seed=seed)
    chs = ["C3", "CP1"]
    if placement == "start":
        target = s.add_glitch(chs, onset="start", duration=0.2)
        target = s.add_motion("C3", after=target, duration=1.0, radius=0.0)
    elif placement == "end":
        target = s.add_motion("C3", onset="end", duration=1.0, radius=0.0)
    elif placement in ("between", "one_bad_boundary"):
        target = s.add_motion("C3", onset=8.0, duration=1.0, radius=0.0)
    else:  # pragma: no cover
        raise ValueError(placement)
    raw, truth = s.build()
    corrupted = sorted({c for ev in truth.events for c in ev.channels})
    raw._data[corrupted] += OFFSET

    r = quiet(RawAPICE, raw)
    r.artifacts.BCT[:] = truth.mask
    good_neighbour_edge = None
    if placement == "one_bad_boundary":
        # all channels bad right after the artifact -> a bad-time (BT) segment
        ev = truth.events[target]
        bt_stop = ev.stop + int(0.5 * r.info["sfreq"])
        r.artifacts.BCT[:, ev.stop:bt_stop] = True
        good_neighbour_edge = "start"
    quiet(r.define_bcbt)
    return r, truth, corrupted, good_neighbour_edge


def _typical_step(x):
    return np.percentile(np.abs(np.diff(x)), 99)


@pytest.mark.parametrize("placement", ["start", "end", "between", "one_bad_boundary"])
def test_spline_segments_no_edge_discontinuity(placement):
    r, truth, corrupted, only = _setup(placement)
    before = r.get_data().copy()
    cfg = cfg_correction_spline_segments(parallelize_mode="channels")
    cfg["n_jobs"] = 1
    quiet(r.correct_spline_segments, cfg=cfg)
    after = r.get_data()
    cct = np.asarray(r.artifacts.CCT, dtype=bool)
    bt = np.asarray(r.artifacts.BT, dtype=bool).ravel()

    # something was corrected on every corrupted channel, and only there (+ mask)
    for ch in corrupted:
        assert cct[ch].any(), f"{r.ch_names[ch]} not corrected"
        art = truth.mask[ch]
        assert np.abs(after[ch, art]).max() < np.abs(before[ch, art]).max()

    for ch in corrupted:
        x = after[ch]
        tol = 2 * _typical_step(before[ch, ~truth.mask[ch]])
        assert OFFSET > 5 * tol, "test would not detect a missing realignment"
        edges = _edges(cct[ch])
        assert len(edges) >= 1
        for b in edges:
            entering = cct[ch, b]  # True: good->corrected edge
            if only == "start" and not entering:
                continue  # this edge borders the bad-time segment
            if bt[b - 1] or bt[b]:
                continue
            jump = abs(x[b] - x[b - 1])
            assert jump <= tol, (
                f"{placement}: {r.ch_names[ch]} jump {jump * 1e6:.1f} uV at sample {b} "
                f"(tolerance {tol * 1e6:.1f} uV)")


def test_spline_segments_clean_data_untouched():
    raw, _ = SyntheticEEG(32, duration=10, seed=1).build()
    r = quiet(RawAPICE, raw)
    before = r.get_data().copy()
    cfg = cfg_correction_spline_segments(parallelize_mode="channels")
    cfg["n_jobs"] = 1
    quiet(r.correct_spline_segments, cfg=cfg)
    np.testing.assert_array_equal(r.get_data(), before)
    assert not np.asarray(r.artifacts.CCT).any()
