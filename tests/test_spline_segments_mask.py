"""Interpolation mask for segment-wise spline correction (bug review #14b).

_modify_rej fills whole fixed-length blocks; it used to put back bad-time
samples that had just been removed, so the spline ran during bad time from
contaminated neighbours.
"""

import numpy as np

from apice import RawAPICE
from apice.artifacts_correction import _build_interpolation_matrix_spline
from conftest import quiet
from synthetic import SyntheticEEG


def _raw():
    raw, _ = SyntheticEEG(32, duration=10, seed=0).build()   # 250 Hz
    return quiet(RawAPICE, raw)


def test_no_interpolation_during_bad_time():
    r = _raw()
    ch = 5
    r.artifacts.BCT[ch, 30:60] = True      # bad stretch on one channel, in block [0, 62)
    r.artifacts.BT[0, 55:100] = True       # bad time overlapping the same block
    mask = np.asarray(quiet(_build_interpolation_matrix_spline, r, min_good_time=1.0,
                            min_intertime=0.05, mask_time=0.1, min_segment_time=0.25), bool)
    bt = np.asarray(r.artifacts.BT, bool).ravel()
    assert mask[ch].any(), "the channel's bad stretch should still be interpolated"
    assert not mask[:, bt].any(), "bad-time samples must not be interpolated"


def test_no_interpolation_on_bad_channels():
    r = _raw()
    r.artifacts.BCT[5, 30:60] = True
    r.artifacts.BCT[7, 30:60] = True
    r.artifacts.BC[7, 0] = True            # channel 7 is a bad channel
    mask = np.asarray(quiet(_build_interpolation_matrix_spline, r, min_good_time=1.0,
                            min_intertime=0.05, mask_time=0.1, min_segment_time=0.25), bool)
    assert mask[5].any()
    assert not mask[7].any()
