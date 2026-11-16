"""Bad-epoch definition on ``EpochsAPICE``: behaviour + known-bug regressions.

The ``xfail(strict=True)`` tests document known bugs; once a bug is fixed the
test XPASSes, which fails the run as a reminder to drop the marker.
"""

import numpy as np
import pytest

from apice import EpochsAPICE
from conftest import quiet
from synthetic import synthetic_epochs

OUTLIERS = (5, 17, 30)


def make(**kw):
    epochs, bad = synthetic_epochs(**kw)
    return quiet(EpochsAPICE, epochs), bad


def be(ep):
    return np.asarray(ep.artifacts.BE, dtype=bool).ravel()


# ---------------------------------------------------------------- behaviour
def test_dist_rejects_outliers():
    ep, bad = make(outliers=OUTLIERS)
    flagged = quiet(ep.define_bad_epochs_dist, lim_dist=2, keeppre=False)
    assert flagged[bad].all(), np.where(flagged)[0]
    assert np.sum(flagged & ~bad) <= 2
    np.testing.assert_array_equal(be(ep), flagged)
    for k in OUTLIERS:
        assert "distance" in ep.artifacts.rejection_reasons[k]


def test_gfp_rejects_outliers():
    ep, bad = make(outliers=OUTLIERS)
    flagged = quiet(ep.define_bad_epochs_gfp, lim_gfp=2, keeppre=False)
    assert flagged[bad].all(), np.where(flagged)[0]
    assert np.sum(flagged & ~bad) <= 2
    for k in OUTLIERS:
        assert "gfp" in ep.artifacts.rejection_reasons[k]


def test_clean_epochs_mostly_kept():
    ep, _ = make(seed=3)
    quiet(ep.define_bad_epochs, bad_data=1, bad_time=0, bad_channel=0.3, lim_dist=2, lim_gfp=2)
    assert be(ep).mean() <= 0.1


def test_artifacts_criterion_uses_masks():
    ep, _ = make()
    n_ch, n_t = len(ep.ch_names), len(ep.times)
    ep.artifacts.BCT[2, :, : n_t // 2] = True            # 50 % bad data
    ep.artifacts.BT[7, 0, 10:20] = True                  # any bad time
    ep.artifacts.BC[9, : int(0.5 * n_ch), 0] = True      # 50 % bad channels
    flagged = quiet(ep.define_bad_epochs_artifacts, bad_data=0.3, bad_time=0, bad_channel=0.3,
                    keeppre=False)
    assert set(np.where(flagged)[0]) == {2, 7, 9}
    assert all("artifacts" in ep.artifacts.rejection_reasons[k] for k in (2, 7, 9))


def test_define_bad_epochs_then_remove():
    ep, bad = make(outliers=OUTLIERS)
    quiet(ep.define_bad_epochs, bad_data=1, bad_time=0, bad_channel=0.3, lim_dist=2, lim_gfp=2)
    assert be(ep)[bad].all()
    n_bad = be(ep).sum()
    n0 = len(ep)
    quiet(ep.remove_bad_epochs)
    assert len(ep) == n0 - n_bad
    assert ep.artifacts.BCT.shape[0] == len(ep)
    assert not be(ep).any()


# ---------------------------------------------------------------- known bugs
@pytest.mark.xfail(strict=True, reason=(
    "define_bad_epochs_dist: `while (~ok ...)` uses bitwise ~ on a bool "
    "(~True == -2, truthy), so the loop ignores convergence and always runs maxloops"))
def test_dist_loop_stops_on_convergence(monkeypatch):
    ep, _ = make(outliers=OUTLIERS)
    calls = {"n": 0}
    orig = np.nanmean

    def counting(a, *args, **kwargs):
        if np.ndim(a) == 3:  # the per-iteration ERP average
            calls["n"] += 1
        return orig(a, *args, **kwargs)

    monkeypatch.setattr(np, "nanmean", counting)
    quiet(ep.define_bad_epochs_dist, lim_dist=2, maxloops=20, keeppre=False)
    # loop 1 rejects the outliers, loop 2 rejects nothing new -> converged
    assert calls["n"] <= 3, f"{calls['n']} iterations"


@pytest.mark.xfail(strict=True, raises=ValueError, reason=(
    "define_bad_epochs_dist(rmvmean=True): np.tile(mean, (1, 1, n_smpl)) has the "
    "wrong shape and cannot broadcast against the data"))
def test_dist_rmvmean():
    ep, bad = make(outliers=OUTLIERS)
    flagged = quiet(ep.define_bad_epochs_dist, lim_dist=2, rmvmean=True, keeppre=False)
    assert flagged[bad].all()


@pytest.mark.xfail(strict=True, reason=(
    "define_bad_epochs_dist: lim_max_dist thresholds np.mean(D) instead of np.max(D)"))
def test_dist_lim_max_uses_max():
    ep, bad = make(spikes=(12,))
    # disable the time-proportion criterion; only the max criterion is active
    flagged = quiet(ep.define_bad_epochs_dist, lim_dist=1e3, lim_max_dist=3, keeppre=False)
    assert flagged[12]


@pytest.mark.xfail(strict=True, reason=(
    "define_bad_epochs_gfp: lim_max_gfp thresholds np.mean(GFP) instead of np.max(GFP)"))
def test_gfp_lim_max_uses_max():
    ep, bad = make(spikes=(12,))
    flagged = quiet(ep.define_bad_epochs_gfp, lim_gfp=1e3, lim_max_gfp=3, keeppre=False)
    assert flagged[12]


def test_spike_epoch_found_by_time_proportion_control():
    """Control for the two tests above: the spike epoch is separable (here via a
    low time-proportion limit), so the lim_max tests fail only because of the bug."""
    ep, _ = make(spikes=(12,))
    flagged = quiet(ep.define_bad_epochs_gfp, lim_gfp=3, lim_bad_time_gfp=0.01, keeppre=False)
    assert flagged[12]
    assert flagged.sum() <= 3


@pytest.mark.xfail(strict=True, reason=(
    "define_bad_epochs_artifacts: `if not tmin` treats tmin=0 as unset and "
    "evaluates the whole epoch, including the baseline"))
def test_artifacts_tmin_zero():
    ep, _ = make()
    base = ep.times < 0
    ep.artifacts.BT[3, 0, base] = True  # bad only in the baseline
    flagged = quiet(ep.define_bad_epochs_artifacts, bad_data=1, bad_time=0, bad_channel=1,
                    tmin=0, keeppre=False)
    assert not flagged[3]


def test_artifacts_tmin_nonzero_control():
    ep, _ = make()
    ep.artifacts.BT[3, 0, ep.times < 0] = True
    flagged = quiet(ep.define_bad_epochs_artifacts, bad_data=1, bad_time=0, bad_channel=1,
                    tmin=0.004, keeppre=False)
    assert not flagged[3]


@pytest.mark.xfail(strict=True, reason=(
    "ArtifactsEpochs.update_be: BE starts as (n,1,1); OR-ing a (n,) vector into it "
    "broadcasts to (n,n,n), so any keeppre=True rejection before a set_be corrupts BE"))
def test_update_be_on_fresh_epochs_keeps_shape():
    ep, _ = make(n_epochs=10)
    quiet(ep.define_bad_epochs_manual, [i == 3 for i in range(10)], keeppre=True)
    assert np.asarray(ep.artifacts.BE).size == 10
    assert be(ep).tolist() == [i == 3 for i in range(10)]
