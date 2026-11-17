"""compute_sme: values against theory, bootstrap vs analytic, channel picks."""

import mne
import numpy as np
import pytest

from apice.erp_statistics import compute_sme

SF = 100.0


def _epochs(n=400, seed=0, bads=()):
    rng = np.random.default_rng(seed)
    info = mne.create_info(["E1", "E2", "E3", "EOG"], SF, ["eeg", "eeg", "eeg", "eog"])
    info["bads"] = list(bads)
    data = rng.normal(0, 1, (n, 4, 50))
    events = np.c_[np.arange(n) * 100, np.zeros(n, int), [1, 2] * (n // 2)]
    return mne.EpochsArray(data, info, events=events, event_id={"a": 1, "b": 2},
                           tmin=0.0, verbose="ERROR")


def test_analytic_matches_theory():
    ep = _epochs()
    # window 0.1-0.3 s = 21 iid N(0,1) samples: score SD = 1/sqrt(21)
    sme = compute_sme(ep, start=0.1, stop=0.3, n_iter=None)
    np.testing.assert_allclose(sme, 1 / np.sqrt(21) / np.sqrt(400), rtol=0.15)


def test_bootstrap_matches_analytic():
    ep = _epochs()
    a = compute_sme(ep, n_iter=None)
    b = compute_sme(ep, n_iter=3000, random_state=0)
    np.testing.assert_allclose(b, a, rtol=0.05)


def test_diff_mode():
    ep = _epochs()
    a = compute_sme(ep, condition=["a", "b"], n_iter=None)
    sa = compute_sme(ep, condition="a", n_iter=None)
    sb = compute_sme(ep, condition="b", n_iter=None)
    np.testing.assert_allclose(a, np.sqrt(sa ** 2 + sb ** 2))
    b = compute_sme(ep, condition=["a", "b"], n_iter=3000, random_state=0)
    np.testing.assert_allclose(b, a, rtol=0.05)


def test_roi_returns_a_scalar():
    ep = _epochs()
    s = compute_sme(ep, roi=["E1", "E2"], n_iter=None)
    assert isinstance(s, float)


def test_default_picks_are_eeg_only():
    ep = _epochs()
    sme, names = compute_sme(ep, n_iter=None, return_ch_names=True)
    assert names == ["E1", "E2", "E3"] and sme.shape == (3,)


def test_bad_channels_and_returned_names():
    ep = _epochs(bads=["E2"])
    sme, names = compute_sme(ep, n_iter=None, return_ch_names=True)
    assert names == ["E1", "E3"] and sme.shape == (2,)
    sme, names = compute_sme(ep, picks=["E1", "E2"], n_iter=None, return_ch_names=True)
    assert names == ["E1", "E2"] and sme.shape == (2,)


def test_invalid_condition():
    with pytest.raises(ValueError):
        compute_sme(_epochs(), condition=["a", "b", "a"])
