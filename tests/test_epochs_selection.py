"""EpochsAPICE: artifact masks follow MNE selections (bug review #11).

Each epoch / channel / sample gets a distinct mask pattern, so a mask that
does not follow the data is detected."""

import mne
import numpy as np
import pytest

from apice import EpochsAPICE
from conftest import quiet

SF = 100.0
N_EP, N_CH, N_S = 6, 4, 200          # 2 s epochs


def _epochs():
    rng = np.random.default_rng(0)
    info = mne.create_info([f"E{i}" for i in range(N_CH)], SF, "eeg")
    events = np.c_[np.arange(N_EP) * 300, np.zeros(N_EP, int), [1, 2] * (N_EP // 2)]
    ep = mne.EpochsArray(rng.normal(0, 1e-6, (N_EP, N_CH, N_S)), info, events=events,
                         event_id={"a": 1, "b": 2}, tmin=-0.5, verbose="ERROR")
    e = quiet(EpochsAPICE, ep)
    a = e.artifacts
    # epoch k, channel c: one bad sample at 10*k + c (unique per epoch / channel)
    for k in range(N_EP):
        for c in range(N_CH):
            a.BCT[k, c, 10 * k + c] = True
        a.CCT[k, k % N_CH, 150 + k] = True
        a.BT[k, 0, 100 + k] = True
    a.BC[:, 1, 0] = True
    a.BE[[1, 4]] = True
    a.rejection_reasons[1].add("x")
    a.BCmanual[2] = True
    return e


def _key(e):
    """(epoch, channel) identity from the BCT pattern: first bad sample."""
    return [[int(np.argmax(e.artifacts.BCT[k, c])) for c in range(len(e.ch_names))]
            for k in range(len(e))]


def _check_shapes(e):
    a = e.artifacts
    n, c, s = len(e), len(e.ch_names), len(e.times)
    assert a.BCT.shape == a.CCT.shape == (n, c, s)
    assert a.BC.shape == (n, c, 1) and a.BT.shape == (n, 1, s) and a.BE.shape == (n,)
    assert len(a.rejection_reasons) == n and a.BCmanual.shape == (c,)
    assert (a.n_epochs, a.n_channels, a.n_samples) == (n, c, s)
    assert list(a.ch_names) == list(e.ch_names)


def test_getitem_by_condition_selects_masks():
    e = _epochs()
    sub = e["a"]                                  # epochs 0, 2, 4
    _check_shapes(sub)
    assert _key(sub) == [[10 * k + c for c in range(N_CH)] for k in (0, 2, 4)]
    assert sub.artifacts.BE.tolist() == [False, False, True]
    _check_shapes(e)                              # the original is unchanged
    assert len(e) == N_EP


def test_getitem_slice_and_reasons():
    e = _epochs()
    sub = e[1:3]
    assert _key(sub)[0][0] == 10 and sub.artifacts.rejection_reasons[0] == {"x"}


def test_drop_in_place():
    e = _epochs()
    e.drop([0, 3])
    _check_shapes(e)
    assert [row[0] for row in _key(e)] == [10, 20, 40, 50]


def test_remove_bad_epochs_drops_masks_once():
    e = _epochs()
    quiet(e.remove_bad_epochs)
    _check_shapes(e)
    assert [row[0] for row in _key(e)] == [0, 20, 30, 50]
    assert not e.artifacts.BE.any()


def test_equalize_event_counts():
    e = _epochs()
    e.drop([0])                                   # a: 2, b: 3
    e.equalize_event_counts()
    _check_shapes(e)
    assert len(e) == 4


def test_pick_and_reorder_channels():
    e = _epochs()
    e.pick(["E3", "E1"])
    _check_shapes(e)
    assert _key(e)[0] == [3, 1]
    assert e.artifacts.BC[0, :, 0].tolist() == [False, True]
    assert e.artifacts.BCmanual.tolist() == [False, False]
    e2 = _epochs()
    e2.drop_channels(["E0"])
    assert _key(e2)[2] == [21, 22, 23] and e2.artifacts.BCmanual.tolist() == [False, True, False]
    e3 = _epochs()
    e3.reorder_channels(["E2", "E0", "E1", "E3"])
    assert _key(e3)[1] == [12, 10, 11, 13]


def test_crop():
    e = _epochs()
    e.crop(tmin=-0.4, tmax=1.0)                   # drops the first 10 samples
    _check_shapes(e)
    assert np.allclose(e.artifacts.times, e.times)
    assert e.artifacts.BT[3, 0].nonzero()[0].tolist() == [103 - 10]
    assert not e.artifacts.BCT[0].any()           # epoch 0's marks were at 0..3


def test_decimate_keeps_exact_samples():
    e = _epochs()
    e.decimate(2)
    _check_shapes(e)
    # BT at sample 100 + k: kept only for even samples (100, 102, 104)
    assert [e.artifacts.BT[k, 0].any() for k in range(N_EP)] == [True, False, True, False, True, False]


def test_resample_ors_the_samples_of_each_new_sample():
    e = _epochs()
    quiet(e.resample, 50.0)
    _check_shapes(e)
    # every original bad time sample falls in exactly one new sample; one on a
    # window boundary (odd samples at 50 Hz) goes to the later one
    got = [np.nonzero(e.artifacts.BT[k, 0])[0].tolist() for k in range(N_EP)]
    assert got == [[(100 + k + 1) // 2] for k in range(N_EP)]
    assert e.artifacts.sfreq == 50.0


def test_add_channels_new_channel_is_good():
    e = _epochs()
    info = mne.create_info(["X"], SF, "eeg")
    extra = mne.EpochsArray(np.zeros((N_EP, 1, N_S)), info, events=e.events, event_id=e.event_id,
                            tmin=e.tmin, verbose="ERROR")
    e.add_channels([extra], force_update_info=True)
    _check_shapes(e)
    assert not e.artifacts.BCT[:, -1].any() and not e.artifacts.BC[:, -1].any()


def test_copy_keeps_independent_masks():
    e = _epochs()
    c = e.copy()
    c.drop([0])
    assert len(e.artifacts.rejection_reasons) == N_EP and e.artifacts.BCT.shape[0] == N_EP
