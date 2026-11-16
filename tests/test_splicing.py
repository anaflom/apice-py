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


# ------------------------------------------------------------- adaptive
def _splice(y, segs, method, n=None, **kw):
    n = y.size if n is None else n
    kw.setdefault("window", SF)
    kw.setdefault("guard", 5)
    kw.setdefault("step_window", 2 * SF)
    kw.setdefault("null_step", SF // 10)
    return np.squeeze(_splice_segments(y, np.array(segs), np.array([0, n]), method=method, **kw))


def _step_case(seed=4, jump=200e-6, tail=0, corrected_by="spline"):
    """Step artifact as in the user's sketch: the baseline jumps and stays
    high; only the jump itself is marked (corrected). ``tail``: samples of the
    jump left unmarked after the segment (imperfect detection).

    corrected_by='spline': the segment was replaced by an estimate at some
    other level (no jump inside). 'pca': target PCA did not remove the jump,
    which is still inside the segment."""
    x = _clean(seed=seed)
    I, F = 2000, 2025                          # 100 ms marked around the jump
    y = x.copy()
    if corrected_by == "spline":
        y[I:F] += 50e-6                        # interpolated, at a wrong level
    else:
        y[I:F] += np.linspace(0, jump, F - I)  # the jump is still there
    y[F:] += jump                              # the offset that stays
    y[F:F + tail] += 100e-6                    # unmarked end of the jump
    return x, y, I, F


def _rms(a):
    return np.sqrt(np.mean(a ** 2))


def test_adaptive_removes_a_step_like_method1_and_unlike_trend():
    x, y, I, F = _step_case()
    after = slice(F + 50, None)
    m1 = _splice(y, [[I, F]], 1)
    tr = _splice(y, [[I, F]], "trend")
    ad = _splice(y, [[I, F]], "adaptive")
    assert _rms(tr[after] - x[after]) > 150e-6     # trend leaves the offset
    assert _rms(m1[after] - x[after]) < 30e-6
    assert _rms(ad[after] - x[after]) < 15e-6
    np.testing.assert_array_equal(ad[:I], y[:I])   # data before the step untouched


def test_adaptive_removes_a_step_that_target_pca_did_not_correct():
    """The jump is still inside the corrected segment (e.g. target PCA could
    not remove it): method 1 aligns the next data to the segment end, so the
    offset stays; the adaptive method compares the levels outside."""
    x, y, I, F = _step_case(corrected_by="pca")
    after = slice(F + 50, None)
    m1 = _splice(y, [[I, F]], 1)
    ad = _splice(y, [[I, F]], "adaptive")
    assert _rms(m1[after] - x[after]) > 150e-6
    assert _rms(ad[after] - x[after]) < 15e-6


def test_adaptive_step_robust_to_missed_jump_end():
    x, y, I, F = _step_case(tail=8)                # 32 ms of the jump not marked
    after = slice(F + 50, None)
    m1 = _splice(y, [[I, F]], 1)
    ad = _splice(y, [[I, F]], "adaptive")
    assert _rms(ad[after] - x[after]) < 15e-6
    assert _rms(ad[after] - x[after]) < 0.5 * _rms(m1[after] - x[after])


def test_adaptive_transient_artifact_leaves_good_data_untouched():
    """Not a step: only the segment changes; it keeps its own shape and is
    aligned at its edges (single-sample anchors by default)."""
    x = _clean(seed=5)
    I, F = 2000, 2250
    y = _corrupt(x, I, F)
    ad = _splice(y, [[I, F]], "adaptive")
    good = np.ones(y.size, bool)
    good[I:F] = False
    np.testing.assert_array_equal(ad[good], y[good])
    np.testing.assert_allclose([ad[I], ad[F - 1]], [y[I - 1], y[F]], rtol=0, atol=1e-15)
    corr = ad[I:F] - y[I:F]                         # offset + linear ramp only
    np.testing.assert_allclose(np.diff(corr, 2), 0, atol=1e-18)


def _infant_background(n=SF * 20, seed=0):
    """Large delta (0.5-2 Hz, ~70 uV RMS) on top of the default background."""
    rng = np.random.default_rng(seed + 100)
    t = np.arange(n) / SF
    delta = sum(rng.uniform(20e-6, 60e-6) * np.sin(2 * np.pi * f * t + rng.uniform(0, 6.3))
                for f in rng.uniform(0.5, 2.0, 4))
    return _clean(n, seed) + delta


def test_adaptive_keeps_slow_waves_inside_segments():
    """The splicing study found that replacing the segment trend ('trend')
    removes real delta inside segments; edge alignment keeps it."""
    x = _infant_background(seed=7)
    I, F = 2000, 2250                               # 1 s segment
    y = _corrupt(x, I, F)
    ad = _splice(y, [[I, F]], "adaptive")
    tr = _splice(y, [[I, F]], "trend")
    err_ad = _rms(ad[I:F] - x[I:F])
    err_tr = _rms(tr[I:F] - x[I:F])
    assert err_ad < 0.5 * err_tr


@pytest.mark.parametrize("anchor", ["sample", "linear"])
def test_adaptive_anchor_options(anchor):
    x = _clean(seed=8)
    I, F = 2000, 2250
    y = _corrupt(x, I, F)
    ad = _splice(y, [[I, F]], "adaptive", anchor=anchor, anchor_window=SF // 10)
    assert _rms(ad[I:F] - x[I:F]) < 0.5 * _rms(y[I:F] - x[I:F])


def test_adaptive_linear_anchor_robust_to_missed_edges():
    x = _clean(seed=9)
    I, F = 2000, 2250
    y = _corrupt(x, I, F)
    y[I - 8:I] += 150e-6                            # unmarked artifact edges
    y[F:F + 8] += 150e-6
    sample = _splice(y, [[I, F]], "adaptive", anchor="sample")
    linear = _splice(y, [[I, F]], "adaptive", anchor="linear", anchor_window=SF // 10)
    assert _rms(linear[I:F] - x[I:F]) < _rms(sample[I:F] - x[I:F])


def test_adaptive_null_uses_only_good_data():
    """Large artifacts elsewhere, marked as bad, must not change the decision
    (the null distribution is built from good data only)."""
    x = _clean(seed=6)
    I, F = 2000, 2250
    y = _corrupt(x, I, F)
    bct = np.zeros(x.size, bool)
    bct[3000:3500] = True
    y_noisy = y.copy()
    y_noisy[3000:3500] += np.random.default_rng(0).normal(0, 300e-6, 500).cumsum() / 10
    a = _splice(y, [[I, F]], "adaptive", bct=bct)
    b = _splice(y_noisy, [[I, F]], "adaptive", bct=bct)
    good = ~bct
    np.testing.assert_allclose(a[good], b[good], rtol=0, atol=1e-15)


def test_adaptive_reports_its_decisions():
    x, y, I, F = _step_case()
    st = {}
    _splice(y, [[I, F]], "adaptive", stats=st)
    assert st == {"step": 1, "no_step": 0, "undecidable": 0}
    x = _clean(seed=5)
    st = {}
    _splice(_corrupt(x, 2000, 2250), [[2000, 2250]], "adaptive", stats=st)
    assert st == {"step": 0, "no_step": 1, "undecidable": 0}


def _undecidable_step_case():
    """A step with too little data before it to estimate the level (the
    segment starts 1 s into the epoch, the level window is 2 s)."""
    x, y, _, _ = _step_case(seed=10)
    I, F = SF, SF + 25
    y = x.copy()
    y[I:F] += 50e-6
    y[F:] += 200e-6
    return x, y, I, F


def test_undecidable_falls_back_to_method1():
    x, y, I, F = _undecidable_step_case()
    st = {}
    ad = _splice(y, [[I, F]], "adaptive", stats=st)
    m1 = _splice(y, [[I, F]], 1)
    assert st["undecidable"] == 1
    np.testing.assert_allclose(ad, m1, rtol=0, atol=1e-15)


def test_undecidable_edges_option_leaves_the_step():
    x, y, I, F = _undecidable_step_case()
    ad = _splice(y, [[I, F]], "adaptive", undecidable="edges")
    after = slice(F + 50, None)
    np.testing.assert_array_equal(ad[after], y[after])     # good data untouched


def test_null_needs_only_half_good_windows():
    """Dense artifacts (bad every 2 s) must not make every segment
    undecidable: the null windows only need 50 % good samples. The segment
    itself has >1 s of contiguous good data on each side (its level windows
    stop at the neighbouring artifacts and need 50 % of 2 s)."""
    x, y, I, F = _step_case(seed=11)
    n = y.size
    bct = np.zeros(n, bool)
    margin = int(1.3 * SF)
    for s in range(100, n - 100, 2 * SF):             # 0.3 s bad every 2 s
        if s + int(0.3 * SF) <= I - margin or s >= F + margin:
            bct[s:s + int(0.3 * SF)] = True
    st = {}
    ad = _splice(y, [[I, F]], "adaptive", bct=bct, stats=st)
    assert st["step"] == 1
    after = slice(F + 50, None)
    good_after = ~bct[after]
    assert _rms((ad[after] - x[after])[good_after]) < 15e-6


def test_level_windows_do_not_cross_other_artifacts():
    """A bad run shortly BEFORE a step must not be declared a step itself
    (its 'after' window would otherwise include the jump)."""
    x, y, I, F = _step_case(seed=12)
    bct = np.zeros(y.size, bool)
    bct[I - 150:I - 75] = True                         # 0.3 s bad, 0.3 s before the step
    st = {}
    ad = _splice(y, [[I, F]], "adaptive", bct=bct, stats=st)
    ok = ~bct
    ok[I:F] = False
    before = slice(I - 75, I)                          # good data between the two
    np.testing.assert_array_equal(ad[before], y[before])


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        _splice_segments(np.zeros(10), np.array([[2, 4]]), np.array([0, 10]), method=7)
