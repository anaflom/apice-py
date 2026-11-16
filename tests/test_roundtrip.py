"""APICE object -> native MNE / disk -> APICE object must preserve artifact masks.

The masks are compared as plain boolean arrays (BCT, BC, BT, CCT and, for
epochs, BE) through the ``masks`` helper, so these tests only need that helper
updated if the artifact storage is refactored.
"""

import mne
import numpy as np
import pytest

from apice import EpochsAPICE, RawAPICE
from apice.io import load_epochapice, load_rawapice
from conftest import quiet
from synthetic import SyntheticEEG

RAW_KEYS = ("BCT", "BC", "BT", "CCT")
EPOCH_KEYS = ("BCT", "BC", "BT", "CCT", "BE")


def masks(obj, keys):
    out = {}
    for k in keys:
        a = np.asarray(getattr(obj.artifacts, k), dtype=bool)
        out[k] = a.ravel() if k == "BE" else a
    return out


def assert_masks_equal(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].shape == b[k].shape, k
        diff = np.argwhere(a[k] != b[k])
        assert diff.size == 0, f"{k}: {len(diff)} entries differ, first {diff[:5].tolist()}"


def _raw_with_masks(first_samp=0, seed=0):
    s = SyntheticEEG(32, duration=20, seed=seed, first_samp=first_samp)
    s.add_glitch(["Cz"], onset="start", duration=0.05)          # touches sample 0
    s.add_motion("O1", onset=5.0, duration=1.3)
    s.add_glitch(["Fz", "F4"], onset=7.123, duration=0.017)     # odd sample counts
    s.add_motion("Pz", onset="end", duration=0.8)               # touches last sample
    s.add_flat("T8")                                            # whole channel -> BC
    raw, truth = s.build()
    r = quiet(RawAPICE, raw)
    r.artifacts.BCT[:] = truth.mask
    r.artifacts.BCT[:, 2500:2600] = True                        # all channels -> BT
    cct = np.zeros_like(truth.mask)
    cct[[1, 2], 1000:1111] = True
    cct[5, -10:] = True
    r.artifacts.CCT[:] = cct
    quiet(r.define_bcbt)
    assert r.artifacts.BT.any() and r.artifacts.BC.any()
    return r


@pytest.mark.parametrize("first_samp", [0, 1234])
def test_raw_roundtrip_native_mne(tmp_path, first_samp):
    r = _raw_with_masks(first_samp)
    expected = masks(r, RAW_KEYS)
    native = quiet(r.to_mne_raw)
    assert type(native) is mne.io.RawArray
    fname = tmp_path / "x_raw.fif"
    native.save(fname, verbose="ERROR")
    reloaded = mne.io.read_raw_fif(fname, preload=True, verbose="ERROR")
    rebuilt = quiet(RawAPICE, reloaded)
    assert_masks_equal(expected, masks(rebuilt, RAW_KEYS))
    np.testing.assert_allclose(rebuilt.get_data(), r.get_data(), rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("first_samp", [0, 1234])
def test_raw_roundtrip_export_load(tmp_path, first_samp):
    r = _raw_with_masks(first_samp)
    expected = masks(r, RAW_KEYS)
    fname = tmp_path / "x-preproc-raw.fif"
    quiet(r.export, fname)
    rebuilt = quiet(load_rawapice, fname)
    assert_masks_equal(expected, masks(rebuilt, RAW_KEYS))


def test_raw_roundtrip_twice_is_stable(tmp_path):
    r = _raw_with_masks()
    expected = masks(r, RAW_KEYS)
    for i in range(2):
        fname = tmp_path / f"x{i}_raw.fif"
        quiet(r.export, fname)
        r = quiet(load_rawapice, fname)
    assert_masks_equal(expected, masks(r, RAW_KEYS))


@pytest.mark.xfail(strict=True, reason=(
    "BC that is not implied by BCT is lost: RawAPICE.__init__ reads info['bads'] "
    "into BC and then define_bcbt() overwrites BC from BCT"))
def test_raw_roundtrip_bc_not_implied_by_bct(tmp_path):
    r = _raw_with_masks()
    r.artifacts.BC[7, 0] = True  # set without any BCT on channel 7
    expected = masks(r, RAW_KEYS)
    fname = tmp_path / "x_raw.fif"
    quiet(r.export, fname)
    rebuilt = quiet(load_rawapice, fname)
    assert_masks_equal(expected, masks(rebuilt, RAW_KEYS))


# ---------------------------------------------------------------- epochs
def _epochs_with_masks():
    r = _raw_with_masks()
    sf = r.info["sfreq"]
    onsets = np.arange(1.0, 19.0, 1.5)
    events = np.c_[(onsets * sf).astype(int) + r.first_samp, np.zeros(len(onsets), int),
                   np.tile([1, 2], len(onsets))[:len(onsets)]]
    ep = quiet(r.segment_continuous_data, events, {"a": 1, "b": 2},
               epoching_kwargs=dict(tmin=-0.2, tmax=0.8, baseline=None, preload=True,
                                    verbose="ERROR"))
    assert isinstance(ep, EpochsAPICE)
    rng = np.random.default_rng(0)
    ep.artifacts.BCT[3, 4, 10:60] = True
    ep.artifacts.BC[5, 9, 0] = True
    ep.artifacts.BT[6, 0, 100:140] = True
    ep.artifacts.CCT[2, rng.integers(0, 32, 5), 20:30] = True
    # keeppre=False: update_be() on a fresh object broadcasts BE to (n, n, n),
    # see test_epoch_rejection.py::test_update_be_on_fresh_epochs_keeps_shape
    quiet(ep.define_bad_epochs_manual, [i in (1, 4) for i in range(len(ep))],
          rejection_reason="manual", keeppre=False)
    return ep


def test_epochs_masks_transferred_from_raw():
    r = _raw_with_masks()
    sf = r.info["sfreq"]
    events = np.array([[int(5.5 * sf), 0, 1]])
    ep = quiet(r.segment_continuous_data, events, {"a": 1},
               epoching_kwargs=dict(tmin=-0.2, tmax=0.8, baseline=None, preload=True,
                                    verbose="ERROR"))
    start = events[0, 0] - int(0.2 * sf)
    n = len(ep.times)
    np.testing.assert_array_equal(ep.artifacts.BCT[0], r.artifacts.BCT[:, start:start + n])
    np.testing.assert_array_equal(ep.artifacts.CCT[0], r.artifacts.CCT[:, start:start + n])


def test_epochs_roundtrip_export_load(tmp_path):
    ep = _epochs_with_masks()
    expected = masks(ep, EPOCH_KEYS)
    reasons = [set(x) for x in ep.artifacts.rejection_reasons]
    fname = tmp_path / "x-epo.fif"
    quiet(ep.export, fname)
    assert (tmp_path / "x-epo-artifacts.csv").exists()
    rebuilt = quiet(load_epochapice, fname)
    assert_masks_equal(expected, masks(rebuilt, EPOCH_KEYS))
    assert [set(x) for x in rebuilt.artifacts.rejection_reasons] == reasons
    np.testing.assert_allclose(rebuilt.get_data(), ep.get_data(), rtol=1e-6, atol=1e-12)


def test_epochs_roundtrip_native_mne(tmp_path):
    """Plain mne.Epochs saved from EpochsAPICE reload and rewrap with masks via the CSV."""
    ep = _epochs_with_masks()
    expected = masks(ep, EPOCH_KEYS)
    fname = tmp_path / "x-epo.fif"
    quiet(ep.export, fname)
    native = mne.read_epochs(fname, verbose="ERROR")
    assert not hasattr(native, "artifacts")
    rebuilt = quiet(EpochsAPICE, native)
    import pandas as pd
    quiet(rebuilt.dataframe_to_rejection_matrix, pd.read_csv(tmp_path / "x-epo-artifacts.csv"))
    assert_masks_equal(expected, masks(rebuilt, EPOCH_KEYS))
