"""Spherical-spline interpolation of channels: geometry and neighbour rules.

Bug-review findings #12 (channel order) and #13 (p_neighbors) (2026-11).
"""

import mne
import numpy as np
import pytest

from apice import RawAPICE
from apice.artifacts_correction import ChannelsSphericalSplineInterpolation, _channel_geometry
from conftest import quiet
from synthetic import SyntheticEEG


def _raw(extra_first=False, seed=7):
    raw, _ = SyntheticEEG(32, duration=10, seed=seed).build()
    if extra_first:
        # a channel without a position placed BEFORE the EEG channels
        info = mne.create_info(["EOG1"], raw.info["sfreq"], "eog")
        eog = mne.io.RawArray(np.random.default_rng(0).normal(0, 50e-6, (1, raw.n_times)), info,
                              verbose=False)
        eog.set_meas_date(raw.info["meas_date"])
        montage = raw.get_montage()
        raw = eog.add_channels([raw], force_update_info=True)
        raw.set_montage(montage, on_missing="ignore")
    return raw


def _interpolate(raw, bad, p_neighbors=1.0):
    r = quiet(RawAPICE, raw)
    for ch in bad:
        r.artifacts.BC[r.ch_names.index(ch), 0] = True
    quiet(ChannelsSphericalSplineInterpolation(p=0.9, p_neighbors=p_neighbors, n_jobs=1).correct, r)
    return r


def test_geometry_is_in_data_order():
    raw = _raw(extra_first=True)
    pos, dist, adj = _channel_geometry(raw.info)
    assert pos.shape == (len(raw.ch_names), 3)
    assert np.isnan(pos[0]).all() and np.isinf(dist[0]).all() and not adj[0].any()
    montage_pos = raw.info.get_montage().get_positions()["ch_pos"]
    i = raw.ch_names.index("Cz")
    # positions are relative to the fitted head origin; distances do not depend on it
    j = raw.ch_names.index("Pz")
    assert dist[i, j] == pytest.approx(np.linalg.norm(montage_pos["Cz"] - montage_pos["Pz"]))


def test_interpolation_unaffected_by_a_non_eeg_channel():
    """#12: positions/adjacency were indexed with data indices but built in
    montage order, so an extra channel shifted every neighbour position."""
    bad = ["C3"]
    ref = _interpolate(_raw(), bad).get_data(picks=bad)
    got = _interpolate(_raw(extra_first=True), bad).get_data(picks=bad)
    np.testing.assert_allclose(got, ref, rtol=1e-8, atol=1e-12)


def test_channel_with_too_many_bad_neighbours_is_not_interpolated():
    """#13: p_neighbors applies to the channels to interpolate (as in MATLAB)."""
    raw = _raw()
    pos, _, adj = _channel_geometry(raw.info)
    c3 = raw.ch_names.index("C3")
    neigh = [raw.ch_names[i] for i in np.where(adj[c3])[0] if i != c3]
    bad = ["C3"] + neigh[: max(2, int(0.6 * len(neigh)))]
    before = raw.get_data(picks=["C3"]).copy()
    r = _interpolate(raw, bad, p_neighbors=0.3)
    np.testing.assert_array_equal(r.get_data(picks=["C3"]), before)


def test_bad_channels_are_never_spline_sources():
    """An isolated bad channel gets the same result whatever p_neighbors is,
    since the sources are the same good channels. (The #13 source bug only
    showed in segment mode, where not all bad channels are interpolated; in
    channel mode, tested here, the old code already excluded them all.)"""
    raw = _raw()
    pos, _, adj = _channel_geometry(raw.info)
    c3 = raw.ch_names.index("C3")
    cluster = ["C3"] + [raw.ch_names[i] for i in np.where(adj[c3])[0] if i != c3][:4]
    bad = cluster + ["O2"]
    a = _interpolate(raw, bad, p_neighbors=1.0).get_data(picks=["O2"])
    b = _interpolate(raw, bad, p_neighbors=0.3).get_data(picks=["O2"])
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-14)
