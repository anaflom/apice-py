"""Splicing of corrected segments ('trend' default and legacy method 1).

See the 2026-11 splicing study: segment-only trend replacement leaves good
data untouched and is robust to artifact edges missed by detection.
"""

import numpy as np
import pytest

from apice.artifacts_correction import _splice_segments

SF = 250


def _clean(n=SF * 20, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SF
    slow = 15e-6 * np.sin(2 * np.pi * 0.3 * t + 1.0)
    fast = np.convolve(rng.normal(0, 4e-6, n), np.ones(5) / 5, mode="same")
    return slow + fast


def _corrupt(x, I, F, offset=80e-6, slope=40e-6):
    y = x.copy()
    y[I:F] += offset + slope * np.linspace(-1, 1, F - I)
    return y


def _trend(y, I, F, n_ep=None, **kw):
    n = y.size
    epoch_if = np.array([0, n]) if n_ep is None else np.array([[k * n // n_ep, (k + 1) * n // n_ep] for k in range(n_ep)])
    kw.setdefault("window", SF)
    kw.setdefault("guard", 5)
    return np.squeeze(_splice_segments(y, np.array([[I, F]]), epoch_if, method="trend", **kw))


def test_trend_restores_segment_and_leaves_good_data_untouched():
    x = _clean()
    I, F = 2000, 2250
    y = _corrupt(x, I, F)
    out = _trend(y, I, F)
    good = np.ones(x.size, bool)
    good[I:F] = False
    np.testing.assert_array_equal(out[good], y[good])
    err_before = np.sqrt(np.mean((y[I:F] - x[I:F]) ** 2))
    err_after = np.sqrt(np.mean((out[I:F] - x[I:F]) ** 2))
    assert err_after < 0.25 * err_before


def test_trend_is_robust_to_missed_artifact_edges():
    """An undetected 40 ms tail of the artifact next to the segment must not
    drag the segment level (it does with a single-sample anchor)."""
    x = _clean(seed=1)
    I, F = 2000, 2250
    y = _corrupt(x, I, F)
    y[I - 10:I] += 150e-6          # artifact tail outside the marked segment
    y[F:F + 10] += 150e-6
    bct = np.zeros(x.size, bool)   # ...and not detected
    out = _trend(y, I, F, bct=bct)
    err_before = np.sqrt(np.mean((y[I:F] - x[I:F]) ** 2))
    err_after = np.sqrt(np.mean((out[I:F] - x[I:F]) ** 2))
    assert err_after < 0.3 * err_before


def test_trend_uses_only_data_of_the_same_epoch():
    """A segment at the start of epoch 1 must not depend on epoch 0."""
    x = _clean(seed=2)
    n = x.size
    I, F = n // 2, n // 2 + 200
    y = _corrupt(x, I, F)
    y_shifted = y.copy()
    y_shifted[: n // 2] += 300e-6                   # epoch 0 at another level
    a = _trend(y, I, F, n_ep=2)
    b = _trend(y_shifted, I, F, n_ep=2)
    np.testing.assert_allclose(a[I:F], b[I:F], rtol=0, atol=1e-15)
    # one-sided: only the level is matched, to the mean of the next window,
    # so slow activity limits the accuracy (the slope cannot be estimated)
    assert abs(np.mean(a[I:F] - x[I:F])) < 0.3 * abs(np.mean(y[I:F] - x[I:F]))


def test_trend_changes_only_corrected_channels():
    x = np.vstack([_clean(seed=s) for s in range(3)])
    I, F = 2000, 2250
    y = x.copy()
    y[1] = _corrupt(x[1], I, F)
    corrected = np.zeros_like(y, bool)
    corrected[1, I:F] = True
    out = _splice_segments(y, np.array([[I, F]]), np.array([0, y.shape[1]]), method="trend",
                           corrected=corrected, window=SF, guard=5)
    np.testing.assert_array_equal(out[[0, 2]], y[[0, 2]])
    assert not np.array_equal(out[1, I:F], y[1, I:F])


def test_trend_excludes_bad_samples_from_the_fit():
    x = _clean(seed=3)
    I, F = 2000, 2250
    y = _corrupt(x, I, F)
    y[F + 5:F + 200] += 300e-6                      # bad data right after
    bct = np.zeros(x.size, bool)
    bct[F + 5:F + 200] = True
    out = _trend(y, I, F, bct=bct, align_bad_times=False)
    ref = _trend(np.where(bct, x, y), I, F, bct=bct, align_bad_times=False)
    # the bad block does not influence the result at all
    np.testing.assert_allclose(out[I:F], ref[I:F], rtol=0, atol=1e-15)


def test_method1_ramp_stays_within_the_bad_run():
    """Review #2: the ramp ran from the previous boundary and tilted good data."""
    d = np.zeros(400)
    d[100:200] = 500.0                              # bad time
    bt = np.zeros(400, bool)
    bt[100:200] = True
    out = np.squeeze(_splice_segments(d, np.array([[200, 250]]), np.array([0, 400]), bt=bt, method=1))
    np.testing.assert_array_equal(out[:100], 0.0)


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        _splice_segments(np.zeros(10), np.array([[2, 4]]), np.array([0, 10]), method=7)
