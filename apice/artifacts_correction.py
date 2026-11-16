"""Artifact correction algorithms for APICE EEG preprocessing.

This module provides correction methods applied after artifact detection,
including target PCA-based correction and spherical spline interpolation at
channel and segment levels.
"""

# %% LIBRARIES
import os
import time
import numpy as np
import progressbar
from progressbar import (ProgressBar, Percentage, Bar)


import mne
from apice.artifacts_detection import (ShortGoodSegments, ShortBadSegments, Mask)

from apice.electrode_positions import _check_origin
from apice.utils import (find_true_segments_edges, mask_bad_segments, include_short_bad_segments, reject_short_good_segments)



def _print_header(header, separator="="):
    """Print a title wrapped by repeated separator lines.

    Parameters
    ----------
    header : str
        Header text to display.
    separator : str, default='='
        Single character (or string) repeated to form the border line.

    Returns
    -------
    None
        Prints formatted text to stdout.
    """
    # Calculate the length of the header text
    header_length = len(header)

    # Create separator lines of the same size as the header text
    separator = separator * header_length

    # Print the separator line
    print(separator)

    # Print the header text
    print(header)

    # Print the separator line below the header
    print(separator + "\n")


# %% FUNCTIONS FOR SPLICING

SPLICE_METHODS = (1, 'trend', 'adaptive')


def _splice_segments(data, bad_if, epoch_if, bct=None, bt=None, bc=None, method=1,
                     corrected=None, window=250, guard=5, order=1, align_bad_times=True,
                     step_window=1000, step_min=250, step_k=2.5, null_step=25,
                     anchor='sample', anchor_window=25, undecidable='method1', stats=None):
    """Splice corrected segments to reduce discontinuities at segment borders.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array with shape (channels, samples) or (samples,).
    bad_if : numpy.ndarray
        Array of segment start/end sample indices to splice (end exclusive).
    epoch_if : numpy.ndarray
        Epoch boundaries as start/end sample indices (end exclusive).
    bct : numpy.ndarray | None, default=None
        Bad-channel-time mask (corrected samples must NOT be marked).
    bt : numpy.ndarray | None, default=None
        Bad-time mask.
    bc : numpy.ndarray | None, default=None
        Bad-channel mask.
    method : {1, 'trend', 'adaptive'} | None, default=1
        Splicing strategy.

        - ``1`` (default): APICE / MATLAB ``eega_splicesgments1``: each
          segment after a boundary is shifted to start at the previous
          sample; the offsets accumulate, so all later data are shifted too
          (a high-pass filter is then needed to remove the resulting drifts).
          This removes "step" artifacts (a permanent jump of the baseline
          where only the jump is marked as bad), at the price of also
          shifting good data after transient artifacts.
        - ``'trend'``: only the corrected samples change. The low-order
          trend of each corrected segment (per channel) is replaced by a
          robust trend fitted to the good data on both sides of it
          (``window`` samples per side, skipping ``guard`` samples next to
          the segment, within the epoch, excluding bad and corrected
          samples). Good data are never modified, so a step artifact is left
          in the data after it. NOT recommended for infant data: with large
          slow waves, replacing the segment's trend removes real activity
          inside it (splicing study 2026-11). See ``_splice_segments_trend``.
        - ``'adaptive'``: decides for each non-good run (corrected, bad or
          bad time) whether it is a step, by comparing the difference between
          the mean of the good data after and before it (windows of the
          largest length w, from ``step_window`` down to ``step_min``, that
          fits on both sides) with a matched reference: the same difference,
          same w and same gap, at all positions of contiguous GOOD data of
          the same channel (threshold: ``step_k`` robust SDs). Steps: the data after the run are
          shifted to the level before it (as method 1, but with robust
          levels, not single samples). Then the corrected samples are aligned
          at their edges, keeping their own shape (as MATLAB method 3; see
          ``anchor``). Good data are modified only after detected steps.
          Undecidable runs (not enough good data): see ``undecidable``. See
          ``_splice_segments_adaptive``.
        - ``None`` / ``0``: no splicing.
    corrected : numpy.ndarray | None, default=None
        Boolean (channels, samples) mask of corrected samples, used by
        ``'trend'`` to know which channels changed in each segment. If None,
        all channels in ``bad_if`` segments are considered corrected.
    window, guard : int, default=250, 5
        Samples per side used for the trend fit, and samples skipped next to
        the segment (``'trend'`` only). Callers pass ~1 s and ~20 ms.
    order : int, default=1
        Order of the trend (``'trend'`` only).
    align_bad_times : bool, default=True
        With ``'trend'`` / ``'adaptive'``, also align bad-time runs to the
        surrounding good data (they stay marked as bad; this only reduces how
        much a later high-pass spreads their offset into good data).
    step_window : int, default=1000
        ``'adaptive'``: longest window (samples per side) for the levels
        before and after a run (callers pass ~4 s); halved until it fits in
        the contiguous good data on both sides and enough reference spans
        exist.
    step_min : int, default=250
        ``'adaptive'``: shortest window allowed (callers pass 1 s; shorter
        windows had too little power with large infant slow waves, so a
        missed step was declared 'no_step'); shorter clean data ->
        'undecidable'.
    undecidable : {'method1', 'edges'}, default='method1'
        ``'adaptive'``, when there is not enough good data to decide:
        'method1' splices that run as method 1 (offset to the data before;
        the data after continue from its last sample); 'edges' aligns it at
        both edges only, leaving any step in the data.
    stats : dict | None, default=None
        ``'adaptive'``: if given, incremented with the number of runs per
        decision ('step', 'no_step', 'undecidable'), counted per channel.
    anchor : {'sample', 'linear'}, default='sample'
        ``'adaptive'``: edge anchors of the corrected segments. 'sample':
        the neighbouring samples (exact continuity; best when edges are clean,
        which the masking of bad segments usually ensures). 'linear': lines
        fitted to ``anchor_window`` samples on each side of the edge
        (skipping ``guard`` outside), extrapolated to it (robust to missed
        artifact edges, less exact otherwise).
    anchor_window : int, default=25
        ``'adaptive'`` with anchor='linear' (callers pass ~100 ms).
    step_k : float, default=2.5
        ``'adaptive'``: a segment is a step if the before/after level
        difference exceeds ``step_k`` robust SDs of the null distribution.
    null_step : int, default=25
        ``'adaptive'``: spacing (samples) of the positions in good data used
        to build the null distribution (callers pass ~0.1 s).

    Returns
    -------
    dN : numpy.ndarray
        Spliced data with reduced boundary jumps.
    """
    if not method:
        dN = data
        print('Not splicing for alingment was performed')
    elif method == 1:
        dN = _splice_segments1(data, bad_if, epoch_if, bct=bct, bt=bt, bc=bc)
    elif method == 'trend':
        dN = _splice_segments_trend(data, bad_if, epoch_if, bct=bct, bt=bt, bc=bc,
                                    corrected=corrected, window=window, guard=guard,
                                    order=order, align_bad_times=align_bad_times)
    elif method == 'adaptive':
        dN = _splice_segments_adaptive(data, bad_if, epoch_if, bct=bct, bt=bt, bc=bc,
                                       corrected=corrected, window=window, guard=guard,
                                       order=order, align_bad_times=align_bad_times,
                                       step_window=step_window, step_k=step_k,
                                       null_step=null_step, step_min=step_min,
                                       anchor=anchor, anchor_window=anchor_window,
                                       undecidable=undecidable, stats=stats)
    else:
        raise ValueError(f"splice method must be one of {SPLICE_METHODS} or None, got {method!r}")
    return dN


def _robust_polyfit(t, v, order, thresh=3.0, niter=4):
    """Polynomial fit that iteratively drops samples with residuals above
    ``thresh`` SDs (as NoiseTools' nt_detrend). Returns None if too few samples."""
    keep = np.isfinite(v)
    p = None
    for _ in range(niter):
        if keep.sum() < order + 1:
            break
        p = np.polyfit(t[keep], v[keep], order)
        r = v - np.polyval(p, t)
        s = np.std(r[keep])
        new = keep & (np.abs(r) < thresh * s) if s > 0 else keep
        if new.sum() == keep.sum():
            break
        keep = new
    return p


def _true_runs(mask):
    """(start, stop) pairs of the True runs of a 1-D boolean vector (stop exclusive)."""
    d = np.diff(np.r_[0, np.asarray(mask, dtype=np.int8), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


def _splice_segments_trend(data, bad_if, epoch_if, bct=None, bt=None, bc=None, corrected=None,
                           window=250, guard=5, order=1, align_bad_times=True, min_side=3):
    """Segment-only splicing by trend replacement (see ``_splice_segments``).

    For each channel and each run of corrected samples [I, F) (split at epoch
    limits), fit a robust polynomial of ``order`` to the good samples in
    [I-guard-window, I-guard) and [F+guard, F+guard+window) (within the epoch),
    fit the same order to the segment itself, and replace the segment's trend
    by the surrounding one. With good data on one side only, a constant level
    is matched. Using a window and a guard, not the single neighbouring
    sample, keeps the alignment robust when the detection misses the edges
    of an artifact (splicing study, 2026-11).
    """
    data_, bad_if, epoch_if, bct, bt, bc, I_all, F_all, Epoch_I, Epoch_F = \
        _prepare_data_for_splicing(data, bad_if, epoch_if, bct, bt, bc)
    n_ch, n_s = data_.shape
    window = max(int(window), min_side)
    guard = max(int(guard), 0)

    if corrected is None:
        corrected = np.zeros((n_ch, n_s), dtype=bool)
        for I, F in zip(np.ravel(I_all), np.ravel(F_all)):
            corrected[:, int(I):int(F)] = True
    else:
        corrected = np.reshape(np.asarray(corrected, dtype=bool), (n_ch, n_s))

    # epoch index of every sample
    epoch_of = np.full(n_s, -1)
    for e, (ei, ef) in enumerate(zip(np.ravel(Epoch_I), np.ravel(Epoch_F))):
        epoch_of[int(ei):int(ef)] = e

    dN = data_.copy()
    for ch in range(n_ch):
        if bc[ch]:
            continue
        bad_ch = np.logical_or(bct[ch], bt) & ~corrected[ch]
        ok = ~bad_ch & ~corrected[ch]
        targets = _true_runs(corrected[ch])
        if align_bad_times:
            targets += _true_runs(bt & ~corrected[ch])
        for I, F in targets:
            # split runs that cross epoch limits
            cuts = [I] + [int(b) for b in np.ravel(Epoch_I) if I < b < F] + [F]
            for i0, f0 in zip(cuts[:-1], cuts[1:]):
                e = epoch_of[i0]
                ei, ef = int(np.ravel(Epoch_I)[e]), int(np.ravel(Epoch_F)[e])
                _trend_replace_run(dN[ch], data_[ch], ok, i0, f0, ei, ef,
                                   window, guard, order, min_side)

    return dN


def _trend_replace_run(out, src, ok, i0, f0, ei, ef, window, guard, order, min_side=3):
    """Replace the trend of src[i0:f0] by the robust trend of the good
    samples (``ok``) around it, within [ei, ef); writes into ``out``."""
    left = np.arange(max(ei, i0 - guard - window), max(ei, i0 - guard))
    right = np.arange(min(ef, f0 + guard), min(ef, f0 + guard + window))
    left = left[ok[left]]
    right = right[ok[right]]
    has_l, has_r = left.size >= min_side, right.size >= min_side
    if not (has_l or has_r):
        return False
    o = order if (has_l and has_r) else 0
    c = (i0 + f0 - 1) / 2.0
    tt = np.concatenate([left, right]) if (has_l and has_r) else (left if has_l else right)
    p_out = _robust_polyfit(tt - c, src[tt], o)
    ts = np.arange(i0, f0)
    p_seg = _robust_polyfit(ts - c, src[i0:f0], min(o, max(f0 - i0 - 2, 0)))
    if p_out is None or p_seg is None:
        return False
    out[i0:f0] = src[i0:f0] - np.polyval(p_seg, ts - c) + np.polyval(p_out, ts - c)
    return True


def _edge_level(src, ok, idx, t0):
    """Value at t0 of a straight line fitted to src[idx] (good samples only);
    a constant if fewer than 6 samples; None if fewer than 1."""
    idx = idx[ok[idx]] if ok is not None else idx
    if idx.size == 0:
        return None
    if idx.size < 6:
        return float(np.mean(src[idx]))
    p = np.polyfit(idx - t0, src[idx], 1)
    return float(p[1])


def _edge_align_run(out, src, ok, i0, f0, ei, ef, anchor='sample', anchor_window=25, guard=5,
                    sides='both'):
    """Segment-only alignment at the edges (MATLAB eega_splicesgments3 style).

    The run src[i0:f0] keeps its own shape; it is offset so that its start
    meets the good data before it, and a linear ramp makes its end meet the
    good data after it (a constant offset if only one side is available).
    Writes into ``out``.

    anchor='sample': the neighbouring samples src[i0-1] and src[f0] (exact
        continuity; best when the bad segment edges are clean, e.g. thanks
        to the masking of bad segments).
    anchor='linear': lines fitted to ``anchor_window`` good samples outside
        (skipping ``guard``) and to the first/last ``anchor_window`` samples
        of the run, extrapolated to the edge (more robust when the edges of
        the artifact were missed, less exact otherwise).
    sides='left': only the start is aligned (a constant offset), as method 1.
    """
    n = f0 - i0
    dL = dR = None
    if anchor == 'sample':
        if i0 - 1 >= ei and ok[i0 - 1]:
            dL = src[i0 - 1] - src[i0]
        if f0 < ef and ok[f0]:
            dR = src[f0] - src[f0 - 1]
    elif anchor == 'linear':
        w = max(int(anchor_window), 1)
        wi = min(w, n)
        lo = _edge_level(src, ok, np.arange(max(ei, i0 - guard - w), max(ei, i0 - guard)), i0 - 0.5)
        if lo is not None:
            li = _edge_level(src, None, np.arange(i0, i0 + wi), i0 - 0.5)
            dL = lo - li
        ro = _edge_level(src, ok, np.arange(min(ef, f0 + guard), min(ef, f0 + guard + w)), f0 - 0.5)
        if ro is not None:
            ri = _edge_level(src, None, np.arange(f0 - wi, f0), f0 - 0.5)
            dR = ro - ri
    else:
        raise ValueError(f"anchor must be 'sample' or 'linear', got {anchor!r}")
    if sides == 'left':
        dR = None
    if dL is None and dR is None:
        return False
    if dL is not None and dR is not None:
        corr = dL + (dR - dL) * (np.arange(n) / max(n - 1, 1))
    else:
        corr = np.full(n, dL if dL is not None else dR)
    out[i0:f0] = src[i0:f0] + corr
    return True


def _good_window_sums(x, ok):
    """Cumulative sums to get, in O(1), the sum and count of good samples of
    any window [a, b): sum = cs_x[b] - cs_x[a], count = cs_n[b] - cs_n[a]."""
    cs_x = np.concatenate([[0.0], np.cumsum(np.where(ok, x, 0.0))])
    cs_n = np.concatenate([[0], np.cumsum(ok.astype(np.int64))])
    return cs_x, cs_n


def _contiguous_good(ok):
    """For every index i in [0, n]: number of contiguous good samples ending
    just before i (``before[i]``) and starting at i (``after[i]``)."""
    n = ok.size
    idx = np.arange(n)
    last_bad = np.maximum.accumulate(np.where(~ok, idx, -1))             # last bad <= j
    next_bad = np.minimum.accumulate(np.where(~ok, idx, n)[::-1])[::-1]  # next bad >= j
    before = np.zeros(n + 1, dtype=np.int64)
    before[1:] = idx - last_bad                                          # i = j + 1
    after = np.zeros(n + 1, dtype=np.int64)
    after[:n] = next_bad - idx
    return before, after


def _step_null_sd(cs_x, good_after, starts, ends, gap, w, guard, null_step, min_null=20):
    """Robust SD of the matched before/after level difference in good data.

    The reference is computed exactly as the observed difference across a
    run of ``gap`` samples: mean of the ``w`` samples after the gap (after a
    ``guard``) minus mean of the ``w`` samples before it (before a
    ``guard``). Positions are taken every ``null_step`` samples within each
    epoch, and only where the whole span (``w``, guard, gap, guard, ``w``) is
    contiguous good data, so no artifact (and no step) is ever included.
    The spread is a robust SD (1.4826 x MAD), which does not assume that
    outliers are present. Returns None if fewer than ``min_null`` positions
    qualify.
    """
    span = 2 * w + 2 * guard + gap
    pos = []
    for ei, ef in zip(starts, ends):
        if ef - ei >= span:
            pos.append(np.arange(ei, ef - span + 1, max(int(null_step), 1)))
    if not pos:
        return None
    a = np.concatenate(pos)
    a = a[good_after[a] >= span]
    if a.size < min_null:
        return None
    r0 = a + w + 2 * guard + gap
    d = (cs_x[r0 + w] - cs_x[r0]) / w - (cs_x[a + w] - cs_x[a]) / w
    mad = np.median(np.abs(d - np.median(d)))
    return 1.4826 * mad if mad > 0 else None


def _matched_windows(w_max, w_min):
    """Window lengths tried for the matched reference: w_max, halved down to w_min."""
    ws = []
    w = int(w_max)
    while w >= max(int(w_min), 1):
        ws.append(w)
        w //= 2
    if not ws or ws[-1] != int(w_min):
        ws.append(max(int(w_min), 1))
    return sorted(set(ws), reverse=True)


SPLICE_STEP_DECISIONS = ('step', 'no_step', 'undecidable')


def _splice_segments_adaptive(data, bad_if, epoch_if, bct=None, bt=None, bc=None, corrected=None,
                              window=250, guard=5, order=1, align_bad_times=True,
                              step_window=1000, step_min=250, step_k=2.5, null_step=25,
                              anchor='sample', anchor_window=25, undecidable='method1',
                              stats=None, min_side=3, min_null=20):
    """Step-aware splicing (see ``_splice_segments``, method ``'adaptive'``).

    Per channel, every run of non-good samples (corrected, bad or bad time),
    split at epoch limits, is examined in time order:

    1. Step decision with a matched reference. Window w: the largest of
       ``step_window``, ``step_window``/2, ... down to ``step_min`` that fits
       in the contiguous good data on BOTH sides of the run (after a
       ``guard``, stopping at neighbouring non-good runs and epoch limits).
       Observed difference: mean of the w samples after the run minus mean
       of the w samples before it. Reference: the same difference, with the
       same w, guard and gap, at all positions of the channel where the
       whole span is contiguous good data (``_step_null_sd``); if fewer than
       ``min_null`` positions exist, w is halved again. Long clean stretches
       thus give sensitive, low-noise windows, while short ones (e.g. many
       short bad marks) still allow a decision with short windows and a
       reference that is noisier in the same way. Windows shorter than
       ``step_min`` (callers: 1 s) are not used: with large slow waves they
       cannot detect relevant steps, so such runs are 'undecidable'.
       |difference| > ``step_k`` robust SDs: 'step', and all the data after
       the run, up to the end of the epoch, are shifted by -difference.
       Otherwise 'no_step'. If the levels or the null cannot be computed:
       'undecidable'.
    2. Corrected samples in the run (and bad-time samples, if
       ``align_bad_times``) are aligned at their edges and keep their own
       shape (``_edge_align_run``: offset at the start, linear ramp to the
       end; ``anchor`` 'sample' or 'linear'), on the (possibly shifted) data.
       For an 'undecidable' run with ``undecidable='method1'``, as method 1
       instead: they are only offset to the data before them, and if the
       run ends with such samples the data after the run are shifted to
       continue from its last sample. With ``undecidable='edges'`` they are
       aligned at both edges as for 'no_step'.

    ``stats`` (dict), if given, is incremented with the number of runs per
    decision ('step', 'no_step', 'undecidable'), counted per channel.
    ``window`` and ``order`` are not used (kept for a common signature).
    """
    if undecidable not in ('method1', 'edges'):
        raise ValueError(f"undecidable must be 'method1' or 'edges', got {undecidable!r}")
    data_, bad_if, epoch_if, bct, bt, bc, I_all, F_all, Epoch_I, Epoch_F = \
        _prepare_data_for_splicing(data, bad_if, epoch_if, bct, bt, bc)
    n_ch, n_s = data_.shape
    step_window = max(int(step_window), min_side)
    guard = max(int(guard), 0)
    starts = [int(v) for v in np.ravel(Epoch_I)]
    ends = [int(v) for v in np.ravel(Epoch_F)]

    if corrected is None:
        corrected = np.zeros((n_ch, n_s), dtype=bool)
        for I, F in zip(np.ravel(I_all), np.ravel(F_all)):
            corrected[:, int(I):int(F)] = True
    else:
        corrected = np.reshape(np.asarray(corrected, dtype=bool), (n_ch, n_s))

    epoch_of = np.full(n_s, -1)
    for e, (ei, ef) in enumerate(zip(starts, ends)):
        epoch_of[ei:ef] = e

    if stats is not None:
        for key in SPLICE_STEP_DECISIONS:
            stats.setdefault(key, 0)

    dN = data_.copy()
    w_list = _matched_windows(step_window, max(min_side, int(step_min)))
    for ch in range(n_ch):
        if bc[ch]:
            continue
        ok = ~(np.logical_or(bct[ch], bt) | corrected[ch])
        x = dN[ch]
        cs_x, cs_n = _good_window_sums(x, ok)
        _, good_after = _contiguous_good(ok)
        null_cache = {}
        runs = _true_runs(~ok)
        for k, (I, F) in enumerate(runs):
            # the level windows stop at the neighbouring non-good runs, so
            # they only use the contiguous good data next to this run (a
            # window crossing another artifact could pick up its step)
            prev_end = runs[k - 1][1] if k > 0 else 0
            next_start = runs[k + 1][0] if k + 1 < len(runs) else n_s
            cuts = [I] + [b for b in starts if I < b < F] + [F]
            for i0, f0 in zip(cuts[:-1], cuts[1:]):
                e = epoch_of[i0]
                ei, ef = starts[e], ends[e]

                # 1. step decision with a matched reference
                decision = 'undecidable'
                lb = i0 - guard                        # left window ends here
                ra = f0 + guard                        # right window starts here
                avail_l = lb - max(ei, prev_end)
                avail_r = min(ef, next_start) - ra
                avail = min(avail_l, avail_r)
                gap = f0 - i0
                for w in w_list:
                    if w > avail:
                        continue
                    key = (gap, w)
                    if key not in null_cache:
                        null_cache[key] = _step_null_sd(cs_x, good_after, starts, ends, gap, w,
                                                        guard, null_step, min_null=min_null)
                    sd = null_cache[key]
                    if sd is None:
                        continue                       # too few clean spans: shorter w
                    diff = (cs_x[ra + w] - cs_x[ra]) / w - (cs_x[lb] - cs_x[lb - w]) / w
                    if abs(diff) > step_k * sd:
                        decision = 'step'
                        x[f0:ef] -= diff
                        # reference spans lie inside single stretches of good
                        # data, so a constant shift after a non-good run never
                        # changes them: the cache stays valid
                        cs_x, cs_n = _good_window_sums(x, ok)
                    else:
                        decision = 'no_step'
                    break
                if stats is not None:
                    stats[decision] += 1

                # 2. alignment of the corrected (and bad-time) samples, which
                # keep their own shape. Anchors only read good samples and the
                # run itself, and are computed before writing, so x can be
                # both source and output.
                targets = corrected[ch, i0:f0].copy()
                if align_bad_times:
                    targets |= bt[i0:f0] & ~corrected[ch, i0:f0]
                fallback = decision == 'undecidable' and undecidable == 'method1'
                for a, b in _true_runs(targets):
                    _edge_align_run(x, x, ok, i0 + a, i0 + b, ei, ef, anchor=anchor,
                                    anchor_window=anchor_window, guard=guard,
                                    sides='left' if fallback else 'both')
                if fallback and targets[-1] and f0 < ef:
                    # method 1: the data after the run continue from its last sample
                    x[f0:ef] += x[f0 - 1] - x[f0]
                    cs_x, cs_n = _good_window_sums(x, ok)

    return dN


def _prepare_data_for_splicing(data, bad_if, epoch_if, bct, bt, bc):
    """Normalize splicing inputs to canonical shapes and masks.

    Parameters
    ----------
    data : numpy.ndarray
        Data as 1D or 2D array.
    bad_if : numpy.ndarray
        Bad segment boundaries.
    epoch_if : numpy.ndarray
        Epoch boundaries.
    bct : numpy.ndarray | None
        Bad-channel-time mask.
    bt : numpy.ndarray | None
        Bad-time mask.
    bc : numpy.ndarray | None
        Bad-channel mask.

    Returns
    -------
    tuple
        Normalized data/masks and segment boundary vectors used by
        splicing functions.
    """
    if np.size(np.shape(data)) == 1:
        data_ = np.reshape(data, (1, np.size(data)))
    elif np.size(np.shape(data)) == 2:
        data_ = data.copy()

    if bct is None or np.size(bct) == 0:
        bct = np.full(np.shape(data_), False)
    else:
        bct = np.reshape(bct, np.shape(data_))
        bct = bct==1

    if bt is None or np.size(bt) == 0:
        bt = np.full(np.shape(data_)[1], False)
    else:
        bt = np.reshape(bt, np.shape(data_)[1])
        bt = bt==1

    if bc is None or np.size(bc) == 0:
        bc = np.full(np.shape(data_)[0], False)
    else:
        bc = np.reshape(bc, np.shape(data_)[0])
        bc = bc==1

    if np.size(np.shape(bad_if)) > 1:
        I_all = bad_if[:, 0].T
        F_all = bad_if[:, 1].T
    else:
        I_all = np.reshape(bad_if[0], (1, 1))
        F_all = np.reshape(bad_if[1], (1, 1))

    if np.size(np.shape(epoch_if)) > 1:
        Epoch_I = epoch_if[:, 0].T
        Epoch_F = epoch_if[:, 1].T
    else:
        Epoch_I = [epoch_if[0]]
        Epoch_F = [epoch_if[1]]
        
    return data_, bad_if, epoch_if, bct, bt, bc, I_all, F_all, Epoch_I, Epoch_F


def _splice_segments1(data, bad_if, epoch_if, bct=None, bt=None, bc=None):
    """Splice by aligning each segment to the previous segment endpoint.

    Parameters
    ----------
    data : numpy.ndarray
        Data to splice.
    bad_if : numpy.ndarray
        Segment start/end boundaries.
    epoch_if : numpy.ndarray
        Epoch start/end boundaries.
    bct : numpy.ndarray | None, default=None
        Bad-channel-time mask.
    bt : numpy.ndarray | None, default=None
        Bad-time mask.
    bc : numpy.ndarray | None, default=None
        Bad-channel mask.

    Returns
    -------
    dN : numpy.ndarray
        Data after segment alignment.
    """
    data, bad_if, epoch_if, bct, bt, bc, I_all, F_all, Epoch_I, Epoch_F = _prepare_data_for_splicing(data, bad_if, epoch_if, bct, bt, bc)
    
    # Splice Segments
    dN = data.copy()
    
    n_epochs = np.size(Epoch_I)
    if np.size(I_all) > 0:
        
        for iep in range(n_epochs):
            
            sample_I = I_all[(I_all >= Epoch_I[iep]) & (I_all <= Epoch_F[iep])]
            sample_F = F_all[(F_all >= Epoch_I[iep]) & (F_all <= Epoch_F[iep])]
            # boundaries of THIS epoch only (as in the MATLAB eega_splicesgments1);
            # using all epochs' limits aligned each epoch to the end of the previous one
            the_I = np.unique(np.hstack([Epoch_I[iep], sample_I, sample_F, Epoch_F[iep]]))
            the_I = np.asarray(the_I, dtype=int)
    
            if np.size(the_I) > 2:
                
                for kk in np.arange(1, np.size(the_I) - 1):
    
                    # Identify which electrodes are fine and which ones are bad in the previous segment
                    if bt[the_I[kk] - 1]:  # If the previous segment is bad, all electrodes are bad
                        id_good_electrode = np.full(np.shape(dN)[0], False)
                        id_bad_electrode = np.full(np.shape(dN)[0], True)
                    else:
                        id_good_electrode = bct[:, the_I[kk] - 1]==False 
                        id_bad_electrode = bct[:, the_I[kk] - 1]
                    
                    # Align the segments of the electrodes with good data in 
                    # previous segment with the previous segment 
                    if np.any(np.logical_and(id_good_electrode,  bc==False)):
                        el = np.logical_and(id_good_electrode,  bc==False)
                        yadd = dN[el, the_I[kk] - 1] - data[el, the_I[kk]]
                        dN[el, the_I[kk]:the_I[kk+1]] = data[el, the_I[kk]:the_I[kk+1]] + np.tile(yadd, (the_I[kk+1]-the_I[kk], 1)).T
    
                    # If the previous segment is bad, align the previous  
                    # segment (the one with bad data, kk-1) to the surronfing  
                    # ones (kk-2 and kk) by fitting a linear trend
                    if np.any(np.logical_and(id_bad_electrode,  bc==False)):
                        el = np.logical_and(id_bad_electrode,  bc==False)
                        el = np.asarray(np.where(el)[0])
                        for i in el:
                            # ramp over the contiguous bad run that ends at
                            # this boundary only (bug review #2: the ramp used
                            # to start at the previous boundary, tilting the
                            # good data before the bad run)
                            bad_i = np.logical_or(bt, bct[i, :])
                            r0 = the_I[kk] - 1
                            while r0 - 1 >= the_I[kk - 1] and bad_i[r0 - 1]:
                                r0 -= 1
                            x = np.arange(r0, the_I[kk])
                            deltaX = x[-1] - x[0]
                            if deltaX != 0:
                                deltaN = dN[i, x[-1]] - dN[i, x[-1]+1]
                                p = deltaN / deltaX
                                dN[i, x] = dN[i, x] - p * (x - x[0])
                                
         
    return dN

# %% FUNCTIONS FOR PCA CORRECTION
def _find_bad_segments_pca(bad_data, intertime, n_epochs, n_samples, mask, maxtime):
    """Find artifact segments eligible for target PCA correction.

    Parameters
    ----------
    bad_data : numpy.ndarray
        Flattened bad-data mask.
    intertime : numpy.ndarray
        Boolean mask indicating time points eligible for interpolation.
    n_epochs : int
        Number of epochs.
    n_samples : int
        Number of samples per epoch.
    mask : int | float
        Mask radius in samples.
    maxtime : int | float
        Maximum allowed segment length in samples.

    Returns
    -------
    bad_if : numpy.ndarray
        Segment boundaries (start/end) in flattened sample indexing.
    """

    bad_if = []
    for ep in np.arange(n_epochs):
        
        temp1 = bad_data[ep * n_samples: (ep + 1) * n_samples]==1
        temp2 = intertime[ep * n_samples: (ep + 1) * n_samples]
        to_interpolate = np.logical_and(temp1==1, temp2==1)

        # mask
        to_interpolate, _ = mask_bad_segments(to_interpolate, mask)
        
        # Beginning and end of each segment
        badi, badf = find_true_segments_edges(to_interpolate)
        badif_ep = np.concatenate( (badi[:,np.newaxis], badf[:,np.newaxis]), axis=1)
                
        if np.shape(badif_ep)[0]>0:

            # Remove too long segments
            bad_duration_ep = badif_ep[:, 1] - badif_ep[:, 0]
            bad_duration_ep = bad_duration_ep - 2 * mask
            idx_to_remove = bad_duration_ep > maxtime
            badif_ep = np.delete(badif_ep, idx_to_remove, axis=0)
                
            if np.shape(badif_ep)[0]>0:
                # Go to total samples
                badif_ep = badif_ep + ep * n_samples
                # append
                bad_if.append(badif_ep)
    
    if bad_if:            
        bad_if = np.vstack(bad_if)
            
    return bad_if


def _target_PCA(data, bad_segment, nSV, vSV, el):
    """Apply target PCA correction on selected bad segments.

    Parameters
    ----------
    data : numpy.ndarray
        Input data matrix with electrodes x samples.
    bad_segment : numpy.ndarray
        Segment boundaries where correction should be applied.
    nSV : int | None
        Number of singular vectors/components to remove.
    vSV : float | None
        Cumulative explained variance threshold for component removal.
    el : int | None
        Electrode index to map corrected output back.

    Returns
    -------
    data : numpy.ndarray
        Corrected data matrix.
    tC : numpy.ndarray
        Boolean mask marking corrected time points.
    """

    if el is None or np.size(el) == 0:
        el = np.full(np.shape(data)[0], True)

    tC = np.full(np.shape(data)[1], False)
        
    # Get the data to correct and remove the mean
    y = data[:,int(bad_segment[0, 0]):int(bad_segment[0, 1])].T
    y = y - np.nanmean(y, axis=0)
    tC[int(bad_segment[0, 0]):int(bad_segment[0, 1])] = True
    for iseg in np.arange(1, np.shape(bad_segment)[0]):
        yi = data[:, int(bad_segment[iseg, 0]):int(bad_segment[iseg, 1])].T
        yi = yi - np.nanmean(yi, axis=0)
        y = np.concatenate((y, yi))
        tC[int(bad_segment[iseg, 0]):int(bad_segment[iseg, 1])] = True
    y = y - np.nanmean(y, axis=0)

    # PCA
    cov_matrix = np.cov(y, rowvar=False)
    [eigenvalues, V] = np.linalg.eigh(cov_matrix)
    eigenvectors = V[:, np.arange(np.shape(cov_matrix)[0] - 1, -1, -1)]
    score = np.dot(y, eigenvectors)
    exp_var = np.var(score, axis=0).T
    exp_var = exp_var / np.sum(exp_var)
    
    n_remove = 0
    if vSV:
        n_remove_var = np.where(np.cumsum(exp_var) >= vSV)[0][0]+1
        n_remove = np.max((n_remove_var, n_remove))
    if nSV:
        n_remove = np.max((nSV, n_remove))

    ev = np.zeros(np.shape(eigenvectors)[0])
    ev[0:n_remove ] = 1
    ev = np.diag(ev)

    yc = y - np.matmul(np.matmul(y, eigenvectors), np.matmul(ev, eigenvectors.T))

    # Store the corrected data
    data[el, tC] = yc[:, el]
        
    return data, tC



# %% FUNCTIONS FOR SPHERICAL SPLINE INTERPOLATION
from joblib import Parallel, delayed
from multiprocessing import Process, Manager


def _channel_geometry(info):
    """Electrode positions, distances and adjacency in DATA channel order.

    The montage and ``find_ch_adjacency`` only cover channels with positions
    (EEG), in their own order. Indexing those arrays with data channel
    indices misaligned them whenever the recording also had other channels
    (EOG, ECG, stim, ...). Here every array has one row per data channel;
    channels without a position get NaN positions, infinite distances and
    no neighbours, so they are never used as spline sources.

    Returns
    -------
    positions : numpy.ndarray, shape (n_channels, 3)
    distance_matrix : numpy.ndarray, shape (n_channels, n_channels)
    adjacency_matrix : numpy.ndarray of bool, shape (n_channels, n_channels)
    """
    from scipy.spatial.distance import cdist

    ch_names = list(info['ch_names'])
    n_ch = len(ch_names)
    ch_pos = info.get_montage().get_positions()['ch_pos']
    origin = _check_origin("auto", info)
    positions = np.full((n_ch, 3), np.nan)
    for i, ch in enumerate(ch_names):
        if ch in ch_pos and np.all(np.isfinite(ch_pos[ch])):
            positions[i] = np.asarray(ch_pos[ch]) - origin
    has_pos = np.all(np.isfinite(positions), axis=1)

    distance_matrix = np.full((n_ch, n_ch), np.inf)
    distance_matrix[np.ix_(has_pos, has_pos)] = cdist(positions[has_pos], positions[has_pos])

    adjacency_matrix = np.zeros((n_ch, n_ch), dtype=bool)
    adj, adj_names = mne.channels.find_ch_adjacency(info, 'eeg')
    idx = np.array([ch_names.index(ch) for ch in adj_names], dtype=int)
    adjacency_matrix[np.ix_(idx, idx)] = adj.toarray().astype(bool)

    return positions, distance_matrix, adjacency_matrix

def _process_bad_channel(bad_idx, ch_names, ch_names_montage, positions, bad_channel_indices, new_exclude_index, data, distances_matrix):
    """Interpolate one bad channel using spherical spline neighbors.

    Parameters
    ----------
    bad_idx : int
        Index of the bad channel to interpolate.
    ch_names : list[str]
        Channel names in data order.
    ch_names_montage : list[str]
        Channel names in montage order.
    positions : numpy.ndarray
        Electrode positions.
    bad_channel_indices : list[int]
        Indices of all bad channels.
    new_exclude_index : list[int]
        Indices to exclude from neighborhood interpolation.
    data : numpy.ndarray
        EEG data matrix (channels x samples).
    distances_matrix : numpy.ndarray
        Pairwise electrode distance matrix.

    Returns
    -------
    tuple[int, numpy.ndarray | None]
        Channel index and interpolated signal, or ``None`` if interpolation
        failed.
    """
    
    #Processes a single bad channel and performs spherical spline interpolation.    
    bad_ch_name = ch_names[bad_idx]
    # positions are in data channel order (see _channel_geometry)
    bad_pos = np.reshape(positions[bad_idx], (1, 3))
    if not np.all(np.isfinite(bad_pos)):
        print(f"Spherical Spline interpolation skipped for channel {bad_ch_name}: no electrode position")
        return bad_idx, None

    distances = distances_matrix[bad_idx,:].copy()
    distances[bad_channel_indices] = np.inf
    distances[new_exclude_index] = np.inf
    
    neighbors_idx = np.where(distances < np.inf)[0]
   
    #neighbor_data = data[good_electrodes, :]
    #neighbor_positions = positions[good_electrodes]

    neighbor_data = data[neighbors_idx, :]
    neighbor_positions = positions[neighbors_idx]
    
    try:
        interpolated_row = _spherical_spline_inter(neighbor_positions, bad_pos, neighbor_data)
        return bad_idx, interpolated_row # Return the bad_idx and the interpolated row.
    except Exception as e:
        print(f"Spherical Spline interpolation failed for channel {bad_ch_name}: {e}")
        return bad_idx, None #return the bad_idx and None if there is an error.

def _parallel_interpolate(bad_channel_indices, ch_names, ch_names_montage, positions, new_exclude_index, data, distances_matrix, n_jobs):
    """Run channel-wise spline interpolation in parallel.

    Parameters
    ----------
    bad_channel_indices : list[int]
        Bad channels to interpolate.
    ch_names : list[str]
        Channel names in data order.
    ch_names_montage : list[str]
        Channel names in montage order.
    positions : numpy.ndarray
        Electrode positions.
    new_exclude_index : list[int]
        Indices to exclude from interpolation neighborhoods.
    data : numpy.ndarray
        EEG data matrix (channels x samples).
    distances_matrix : numpy.ndarray
        Pairwise distance matrix.
    n_jobs : int
        Number of parallel workers.

    Returns
    -------
    result_interpolation : numpy.ndarray
        Data matrix with interpolated bad channels replaced when successful.
    """
    #Parallelizes the spherical spline interpolation for bad channels.
    result_interpolation = np.copy(data) # Create a copy to store the interpolated data

    results = Parallel(n_jobs=n_jobs)(delayed(_process_bad_channel)(bad_idx, ch_names, ch_names_montage, positions, bad_channel_indices, new_exclude_index, data, distances_matrix) for bad_idx in bad_channel_indices)

    for bad_idx, interpolated_row in results:
        if interpolated_row is not None:
            result_interpolation[bad_idx, :] = interpolated_row # Place the row into the copied array.
    
    return result_interpolation

def _do_spherical_spline_interpolation(raw, distances_matrix, positions, adjacency_matrix, bad_neighbor_proportion, bad_channels_to_interpolate, all_bad_channels=None, interpolation_channels=False, n_jobs=-1):
    """Interpolate selected bad channels using spherical spline interpolation.

    Parameters
    ----------
    raw : RawAPICE
        Input recording object containing channel info and data.
    distances_matrix : numpy.ndarray
        Pairwise distance matrix between electrodes.
    positions : numpy.ndarray
        Electrode Cartesian coordinates.
    adjacency_matrix : numpy.ndarray
        Binary adjacency matrix defining neighboring electrodes.
    bad_neighbor_proportion : float
        Maximum allowed proportion of bad neighbors for a channel to be
        considered interpolable.
    bad_channels_to_interpolate : array_like
        Boolean/index mask for channels targeted for interpolation.
    all_bad_channels : array_like | None, default=None
        Boolean/index mask of all currently bad channels.
    interpolation_channels : bool, default=False
        If True, use channel-level interpolation mode.
    n_jobs : int, default=-1
        Number of parallel workers.

    Returns
    -------
    new_interpolated_data : numpy.ndarray
        Data matrix with interpolated channels replaced when successful.
    interpolated_bad_channels : numpy.ndarray
        Boolean vector indicating channels that were interpolated.
    """

    # Mark the bad channels in the raw data information
    bads_list = list(np.asarray(raw.ch_names, dtype=str)[all_bad_channels])

    # Get electrode positions
    spec_ch_pos = raw.info.get_montage().get_positions()['ch_pos']
    ch_names_montage = list(spec_ch_pos.keys()) # Channel names from the montage

    # Get data and channel names
    data = raw.get_data()
    ch_names = raw.ch_names

    # Get common channels names
    common_channels = list(set(ch_names) & set(ch_names_montage))
    montage_indices = [ch_names_montage.index(ch) for ch in common_channels]

    # Find indices of bad channels
    bad_channel_indices = [ch_names.index(ch) for ch in bads_list]

    # Determine which channels to exclude from the interpolation
    exclude_channels = np.logical_xor(all_bad_channels, bad_channels_to_interpolate)
    
    if interpolation_channels:
        exclude_channels = all_bad_channels

    exclude = list(np.asarray(raw.ch_names, dtype=str)[exclude_channels])
    
    # Get the indices of the electrodes to exclude
    exclude_indices = []
    for electrode in exclude:
        exclude_indices.append(raw.ch_names.index(electrode)) 
    
    # Removing the all bad channels
    if not interpolation_channels:
        bad_channel_indices = list(set(bad_channel_indices).symmetric_difference(set(exclude_indices)))
        exclude_channels[bad_channel_indices] = True

    # No bad channel is ever used as a spline source: neither the excluded
    # ones nor those to interpolate (including any dropped just below).
    # Previously, with bad_neighbor_proportion < 1, a bad channel surrounded
    # by bad channels was wrongly kept as a source.
    all_bad_set = set(exclude_indices) | set(bad_channel_indices)
    new_exclude_index = sorted(all_bad_set)

    # Do not interpolate channels with too many bad neighbours (as in the
    # MATLAB eega_InterpSphericSpline 'pneigh'): the proportion of bad
    # neighbours is computed for each channel to interpolate, counting as bad
    # every bad channel (excluded ones and ones being interpolated).
    if bad_neighbor_proportion is not None and bad_neighbor_proportion < 1:
        keep = []
        for el in bad_channel_indices:
            el_neighbors = [i for i in np.where(adjacency_matrix[el])[0] if i != el]
            if len(el_neighbors) == 0:
                continue  # no neighbours: cannot be interpolated reliably
            p_bad = sum(i in all_bad_set for i in el_neighbors) / len(el_neighbors)
            if p_bad <= bad_neighbor_proportion:
                keep.append(el)
            else:
                print(f"Channel {ch_names[el]} not interpolated: {p_bad:.0%} of its neighbours are bad")
        bad_channel_indices = keep


    # Interpolate bad channels in the EEG data
    new_interpolated_data = _parallel_interpolate(bad_channel_indices, ch_names, ch_names_montage, positions, new_exclude_index, data, distances_matrix, n_jobs=n_jobs)
    
    # Identify which channels were successfully interpolated
    interpolated_bad_channels = np.full(len(raw.ch_names), False)
    for el in range(len(raw.ch_names)):
        #interpolated_bad_channels[el] = (raw.ch_names[el] in bads_list)
        interpolated_bad_channels[el] = (raw.ch_names[el] in [ch_names[i] for i in bad_channel_indices])
        raw.info["bads"] = [ch for ch in raw.info["bads"] if ch in [ch_names[i] for i in bad_channel_indices]]

    return new_interpolated_data, interpolated_bad_channels

from numpy.polynomial.legendre import legval
from scipy.linalg import pinv

def _calc_g(cosang, stiffness=4, n_legendre_terms=7):
    """Compute Perrin spherical spline kernel values.

    Parameters
    ----------
    cosang : numpy.ndarray
        Cosine of angular distances between electrode positions.
    stiffness : int, default=4
        Spline stiffness parameter.
    n_legendre_terms : int, default=7
        Number of Legendre polynomial terms.

    Returns
    -------
    numpy.ndarray
        Kernel matrix values used for spherical spline interpolation.
    """
    factors = [
        (2 * n + 1) / (n**stiffness * (n + 1) ** stiffness * 4 * np.pi)
        for n in range(1, n_legendre_terms + 1)
    ]
    return legval(cosang, [0] + factors)

def _normalize_vectors(rr):
    """Normalize surface vertices."""
    size = np.linalg.norm(rr, axis=1)
    mask = size > 0
    rr[mask] /= size[mask, np.newaxis]  # operate in-place
    return size

def _spherical_spline_inter(good_pos, bad_pos, good_data):
    """Interpolate bad positions from good channels using spherical splines.

    Parameters
    ----------
    good_pos : numpy.ndarray
        Cartesian coordinates of good electrodes (n_good, 3).
    bad_pos : numpy.ndarray
        Cartesian coordinates of target electrodes (n_bad, 3).
    good_data : numpy.ndarray
        Data from good electrodes (n_good, n_samples).

    Returns
    -------
    interpdata : numpy.ndarray
        Interpolated data at bad electrode positions.
    """

    _normalize_vectors(good_pos)
    _normalize_vectors(bad_pos)
    
    Gelec = _calc_g(good_pos.dot(good_pos.T)) # from
    Gsph = _calc_g(bad_pos.dot(good_pos.T)) # to

    Gelec.flat[:: len(Gelec) + 1] += 1e-5

    n_from = Gelec.shape[0]
    n_to = Gsph.shape[0]

    C = np.vstack(
        [
            np.hstack([Gelec, np.ones((n_from, 1))]),
            np.hstack([np.ones((1, n_from)), [[0]]]),
        ]
    )
    C_inv = pinv(C)

    interpolation = np.hstack([Gsph, np.ones((n_to, 1))]) @ C_inv[:, :-1]

    interpdata = np.matmul(interpolation, good_data)

    return interpdata.copy() 


def _build_interpolation_matrix_spline(raw, min_good_time, min_intertime, mask_time, min_segment_time):
    """Build a cleaned interpolation mask for segment-wise spline correction.

    Parameters
    ----------
    raw : RawAPICE
        Input recording with artifact matrices.
    min_good_time : float
        Minimum good segment duration (seconds) to preserve.
    min_intertime : float
        Minimum bad segment duration (seconds) required for interpolation.
    mask_time : float
        Temporal mask extension around bad segments (seconds).
    min_segment_time : float
        Minimum segment duration (seconds) used for segment consolidation.

    Returns
    -------
    numpy.ndarray
        Boolean/integer interpolation matrix aligned with ``raw.artifacts.BCT``.
    """

    n_electrodes, n_samples, n_epochs = raw.get_data_size()
    raw_int=raw.copy()

    # remove segments to interpolate that are too short
    shortbad = ShortBadSegments(time_limit=min_intertime, verbose=False)
    shortbad.reject(raw_int)
    
    # mask interpolation segmets
    maskrej = Mask(mask_length=mask_time, verbose=False)
    maskrej.reject(raw_int)

    # mark as interpolation segmets too short no interpolation segmets
    shortgood = ShortGoodSegments(time_limit=min_good_time, verbose=False)
    shortgood.reject(raw_int)

    # remove bad times and bad channels from the interpolation matrix
    if len(raw_int._data.shape)==2:
        # remove bad channels
        raw_int.artifacts.BCT[np.tile(raw.artifacts.BC,(1, n_samples))] = 0
        # remove bad times
        raw_int.artifacts.BCT[np.tile(raw.artifacts.BT,(n_electrodes, 1))] = 0
    elif len(raw_int._data.shape)==3:
        # remove bad channels
        raw_int.artifacts.BCT[np.tile(raw.artifacts.BC,(1, 1, n_samples))] = 0
        # remove bad times
        raw_int.artifacts.BCT[np.tile(raw.artifacts.BT,(1, n_electrodes, 1))] = 0

    # remove segments to interpolate that are too short
    shortbad = ShortBadSegments(time_limit=min_intertime, verbose=False)
    shortbad.reject(raw_int)

    # modify the rejection to avoid having too short separate segments to 
    # interpolate due to different channels to interpolate
    segment_samples = int(np.round(min_segment_time*raw.info['sfreq']))
    max_samples = int(np.round(mask_time/2*raw.info['sfreq']))
    if len(raw_int._data.shape)==2:
        raw_int.artifacts.BCT = _modify_rej(raw_int.artifacts.BCT, segment_samples, max_samples)
    elif len(raw_int._data.shape)==3:
        for iep in range(n_epochs):
            raw_int.artifacts.BCT[iep,:,:] = _modify_rej(raw_int.artifacts.BCT[iep,:,:], segment_samples, max_samples)

    # _modify_rej fills whole blocks, which can put back bad-time samples
    # (and bad channels) removed above; during bad time most channels are bad,
    # so a spline there would be built from contaminated sources. Remove them
    # again (bug review #14b).
    bct_int = np.asarray(raw_int.artifacts.BCT, dtype=bool)
    if len(raw_int._data.shape)==2:
        bct_int[np.tile(np.asarray(raw.artifacts.BC, dtype=bool), (1, n_samples))] = False
        bct_int[np.tile(np.asarray(raw.artifacts.BT, dtype=bool), (n_electrodes, 1))] = False
    elif len(raw_int._data.shape)==3:
        bct_int[np.tile(np.asarray(raw.artifacts.BC, dtype=bool), (1, 1, n_samples))] = False
        bct_int[np.tile(np.asarray(raw.artifacts.BT, dtype=bool), (1, n_electrodes, 1))] = False
    raw_int.artifacts.BCT = bct_int.astype(np.asarray(raw_int.artifacts.BCT).dtype)

    return raw_int.artifacts.BCT
    

def _modify_rej(rej, segment_samples, max_samples):
    """Regularize rejection mask within fixed-length segments.

    Parameters
    ----------
    rej : numpy.ndarray
        Rejection matrix (channels x samples).
    segment_samples : int
        Number of samples per segment.
    max_samples : int
        Maximum rejected samples per segment before marking the full segment
        as rejected for that channel.

    Returns
    -------
    rej_out : numpy.ndarray
        Regularized rejection matrix.
    """
    
    n_samples = np.shape(rej)[1]
    
    # initialize new rejection matrix
    rej_out = np.full(np.shape(rej), 0, dtype=int)
    
    # define segments (the last one may be shorter). Same segments as before,
    # but also valid when the data are shorter than one segment.
    segment_samples = max(1, int(segment_samples))
    seg = np.arange(0, n_samples, segment_samples)

    # iterate over segments
    for segi in seg:
        m = rej[:,segi:segi+segment_samples]
        nreji = np.sum(m,axis=1)
        idx_ch_rej = nreji>max_samples
        rej_out[idx_ch_rej, segi:segi+segment_samples] = 1

    return rej_out

def _find_segments_interpolation_spline(rej, bad_channels):
    """Find contiguous interpolation segments and involved channels.

    Parameters
    ----------
    rej : numpy.ndarray
        Rejection matrix (channels x samples).
    bad_channels : numpy.ndarray
        Boolean mask of channels excluded from interpolation.

    Returns
    -------
    bad_if : numpy.ndarray
        Segment boundaries (start/end indices).
    cha_interpolate : list[numpy.ndarray]
        Channel masks indicating channels to interpolate per segment.
    """
    
    # remove bad channesl from rejection
    rej[bad_channels,:] = 0
    
    # find segments where changes happend
    bad_if = _find_segments_change(rej)
    n_segm = np.shape(bad_if)[0]
    
    # keep the segments with something to interpolate
    segm_rmv = np.full((n_segm), False)
    for i in range(n_segm):
        if not np.any(np.any(rej[:,bad_if[i,0]:bad_if[i,1]]==1, axis=1)):
            segm_rmv[i] = True
    bad_if = bad_if[segm_rmv==False,:]
    n_segm = np.shape(bad_if)[0]
    
    # get the channels to interpolate in each segment
    cha_interpolate = [np.any(rej[:,segmi[0]:segmi[1]]==1, axis=1) for segmi in bad_if]
    # cha_interpolate = [list(np.where(np.any(rej[:,segmi[0]:segmi[1]]==1, axis=1))[0]) for segmi in bad_if]
     
    return bad_if, cha_interpolate

def _find_segments_change(rej):
    """Compute segment boundaries where rejection pattern changes.

    Parameters
    ----------
    rej : numpy.ndarray
        Rejection matrix (channels x samples).

    Returns
    -------
    bad_if : numpy.ndarray
        Segment boundaries (start/end indices).
    """
    
    # Identify the changes in the number of bad channels
    n_samples = np.shape(rej)[1]
    change_channel = np.where(np.any(np.diff(rej, axis=1), axis=0))[0]
    seg_i = np.unique(np.hstack([change_channel + 1, 0]))
    seg_f = np.unique(np.hstack([change_channel + 1, n_samples]))
    bad_if = np.asarray([seg_i.T, seg_f.T]).T
    
    return bad_if


def _interpolate_spline_segment_task(raw, raw_data_2d, segment_start, segment_end,
                                     bad_channels_to_interpolate, all_bad_channels,
                                     distance_matrix, positions, adjacency_matrix,
                                     p_neighbors, n_jobs):
    """Interpolate one segment and return only the updated channels for safe merging."""
    cropped_eeg = raw.copy()
    cropped_eeg._data = raw_data_2d[:, segment_start:segment_end]
    seg_len = int(segment_end - segment_start)
    cropped_eeg._times = np.arange(seg_len, dtype=float) / raw.info['sfreq']

    interpolated_data, interpolated_bad_channels = _do_spherical_spline_interpolation(
        cropped_eeg,
        distance_matrix,
        positions,
        adjacency_matrix,
        p_neighbors,
        bad_channels_to_interpolate,
        all_bad_channels,
        False,
        n_jobs,
    )

    bad_ch = np.where(interpolated_bad_channels)[0]
    if np.size(bad_ch) == 0:
        return segment_start, segment_end, bad_ch, np.empty((0, seg_len))

    return segment_start, segment_end, bad_ch, interpolated_data[bad_ch, :]

# %% CLASSES FOR ARTIFACTS CORRECTION

class ArtCorrection:
    """Base class for artifact correction algorithms.

    Provides shared parameter handling and bookkeeping utilities used by
    concrete correction methods.

    Parameters
    ----------
    verbose : bool, default=True
        Whether to print correction progress details.
    """

    def __init__(self, verbose=True):

        # Arreange parameters
        self.params = dict()
        self.verbose = verbose

    def steps_pre_interpolation(self):
        # print info
        if self.verbose:
            print('\nInterpolation parameters')
            for k in self.params.keys():
                print('-- {}: '.format(k), self.params[k])
        
    def steps_post_interpolation(self, raw, interpolation_matrix):
        
        n_electrodes, n_samples, n_epochs = raw.get_data_size()
        
        # Mark the interpolated data
        if self.params['save_corrected']:
            BCT = np.reshape(raw.artifacts.BCT.copy(), (n_epochs, n_electrodes, n_samples))
            INT = np.reshape(interpolation_matrix.copy(), (n_epochs, n_electrodes, n_samples))
            BCT[np.logical_and(INT, BCT==1)] = 0
            
            # reshape if necessary
            if len(raw._data.shape)==2:
                BCT = np.squeeze(np.transpose(BCT,(1,2,0)), axis=2) 
                INT = np.squeeze(np.transpose(INT,(1,2,0)), axis=2)
                
            # update the rejection
            raw.artifacts.BCT = BCT
            raw.artifacts.CCT[INT] = True
            
        # save interpolated data
        self.interpolation_matrix = interpolation_matrix
        self.interpolated = np.sum(interpolation_matrix.flatten())/np.size(interpolation_matrix)

    def _report_splice_stats(self, raw, stats, label):
        """Print the 'adaptive' splicing decisions and accumulate them in
        ``raw.artifacts.splice_stats`` (read by the preprocessing summary)."""
        self.splice_stats = dict(stats)
        if not stats:
            return
        total = sum(stats.values())
        fallback = self.params.get('splice_undecidable', 'method1')
        print(f"Splicing ({label}, 'adaptive'): {total} runs (per channel): "
              f"{stats.get('step', 0)} steps, {stats.get('no_step', 0)} without step, "
              f"{stats.get('undecidable', 0)} undecidable (spliced with '{fallback}')")
        acc = dict(getattr(raw.artifacts, 'splice_stats', None) or {})
        for k, v in stats.items():
            acc[k] = acc.get(k, 0) + v
        raw.artifacts.splice_stats = acc


class TargetPCA(ArtCorrection):
    """Artifact correction based on target PCA per electrode.

    Parameters
    ----------
    max_time : float, default=0.100
        Maximum segment duration in seconds eligible for correction.
    components_to_remove : int | None, default=None
        Fixed number of components to remove.
    variance_to_remove : float, default=0.98
        Cumulative variance threshold for component removal.
    mask_time : float, default=0.05
        Temporal masking around bad segments in seconds.
    all_time : str, default='all'
        Time selection strategy for PCA.
    all_channel : str, default='no_bad_channel'
        Channel selection strategy for PCA.
    all_epochs : str, default='all'
        Epoch selection strategy for PCA.
    splice_method : {1, 'trend', 'adaptive'} | None, default=1
        Splicing strategy applied after correction (see ``_splice_segments``).
    splice_undecidable : {'method1', 'edges'}, default='method1'
        With 'adaptive', how to splice segments without enough good data to
        decide whether they are steps (see ``_splice_segments``).
    splice_step_k : float, default=2.5
        With 'adaptive', a segment is a step if the before/after level
        difference exceeds this many robust SDs (see ``_splice_segments``).
    save_corrected : bool, default=True
        Whether to update correction masks in ``raw.artifacts``.
    """

    def __init__(self, max_time=0.100, components_to_remove=None, variance_to_remove=0.98, mask_time=0.05,
                 all_time='all', all_channel='no_bad_channel', all_epochs='all', splice_method=1, save_corrected=True,
                 splice_undecidable='method1', splice_step_k=2.5):
        
        super().__init__()
        
        # Arreange parameters for rejection
        self.params = dict(max_time=max_time, components_to_remove=components_to_remove,
                           variance_to_remove=variance_to_remove, mask_time=mask_time,
                           all_time=all_time, all_channel=all_channel, all_epochs=all_epochs,
                           save_corrected=save_corrected, index_to_remove=[], splice_method=splice_method,
                           splice_undecidable=splice_undecidable, splice_step_k=splice_step_k,
                           verbose=True)
        

    def correct(self, raw):         
        """Run target PCA correction and update artifact bookkeeping.

        Parameters
        ----------
        raw : RawAPICE
            Input object to correct in place.

        Returns
        -------
        None
            Data are modified in place.
        """
            
        _print_header('Performing Target PCA per Electrode', separator="-")
        
        # Steps pre interpolation
        self.steps_pre_interpolation()
        
        # Target PCA
        stats = {}
        raw._data, interpolation_matrix = self.apply_target_PCA(raw,
                                                                max_time=self.params['max_time'],
                                                                components_to_remove=self.params['components_to_remove'],
                                                                variance_to_remove=self.params['variance_to_remove'],
                                                                mask_time=self.params['mask_time'],
                                                                all_time=self.params['all_time'],
                                                                all_channel=self.params['all_channel'],
                                                                all_epochs=self.params['all_epochs'],
                                                                splice_method=self.params['splice_method'],
                                                                splice_undecidable=self.params.get('splice_undecidable', 'method1'),
                                                                splice_step_k=self.params.get('splice_step_k', 2.5),
                                                                stats=stats,
                                                                )

        # Steps post interpolation
        self.steps_post_interpolation(raw, interpolation_matrix)
        self._report_splice_stats(raw, stats, 'target PCA')
            
        print('\n')
     
    @staticmethod
    def apply_target_PCA(raw, max_time=0.100, components_to_remove=None, variance_to_remove=0.98, mask_time=0.05,
                 all_time='all', all_channel='no_bad_channel', all_epochs='all', splice_method=1,
                 splice_undecidable='method1', splice_step_k=2.5, stats=None):
        """Apply target PCA correction and return corrected data plus mask.

        Parameters
        ----------
        raw : RawAPICE
            Input raw/epochs object.
        max_time : float, default=0.100
            Maximum segment duration in seconds for PCA correction.
        components_to_remove : int | None, default=None
            Number of components to remove.
        variance_to_remove : float, default=0.98
            Cumulative explained variance threshold for component removal.
        mask_time : float, default=0.05
            Temporal mask in seconds around bad segments.
        all_time : str, default='all'
            Time selection mode.
        all_channel : str, default='no_bad_channel'
            Channel selection mode.
        all_epochs : str, default='all'
            Epoch selection mode.
        splice_method : {1, 'trend', 'adaptive'} | None, default=1
            Splicing method for corrected segments.
        splice_undecidable : {'method1', 'edges'}, default='method1'
            With 'adaptive': splicing of undecidable segments.
        splice_step_k : float, default=2.5
            With 'adaptive': step threshold in robust SDs.
        stats : dict | None, default=None
            With 'adaptive': filled with the counts of step decisions.

        Returns
        -------
        data_pca : numpy.ndarray
            Corrected data.
        interpolation_matrix : numpy.ndarray
            Boolean mask of corrected samples.
        """

        # get data size
        n_electrodes, n_samples, n_epochs_all = raw.get_data_size()
        
        # get the times that are considered for PCA
        if all_time == 'all':
            if len(np.shape(raw._data))==2:
                IT = np.full((1, n_samples), True)
            else:
                IT = np.full((n_epochs_all, 1, n_samples), True)    
        elif all_time == 'no_bad_time':
            IT = raw.artifacts.BT.copy()==False
        elif all_time == 'bad_time':
            IT = raw.artifacts.BT.copy()==True

        # get the channels that are considered for PCA
        if all_channel == 'all':
            if len(np.shape(raw._data))==2:
                IC = np.full(n_electrodes, True)
            else:
                IC = np.full((n_epochs_all, n_electrodes), True)
        elif all_channel == 'no_bad_channel':
            IC = raw.artifacts.BC.copy()==False
            IC = np.reshape(IC, (n_epochs_all, n_electrodes))
        elif all_channel == 'bad_channel':
            IC = raw.artifacts.BC.copy()==True
            IC = np.reshape(IC, (n_epochs_all, n_electrodes))
        
        # get the epochs considered for PCA
        if len(np.shape(raw._data))==2:
            IE = np.full(1, True)
        else:
            if all_epochs == 'all':
                IE = np.full(n_epochs_all, True)
            elif all_epochs == 'no_bad_epoch':
                IE = np.reshape(raw.artifacts.BE, n_epochs_all)==False
            elif all_epochs == 'bad_epoch':
                IE = np.reshape(raw.artifacts.BE, n_epochs_all)==True
        get_some_epochs = np.any(IE==0)
        
        # get the data and rejection matrix
        DATA = raw._data.copy()
        BCT = raw.artifacts.BCT.copy()
        BT = raw.artifacts.BT.copy()
        
        # mark bad times as bad for all channnels
        if len(np.shape(raw._data))==2:
            BCT[np.tile(BT,(n_electrodes,1))] = 1
        else:
            BCT[np.tile(BT,(1, n_electrodes, 1))] = 1
        
        # get epochs to interpolate
        if get_some_epochs and len(np.shape(raw._data))==3:
            DATA = DATA[IE,:,:]
            BCT = BCT[IE,:,:]
            BT = BT[IE,:,:]
            IC = IC[IE,:]
            IT = IT[IE,:,:]
            n_epochs = np.sum(IE)
        else:
            n_epochs = n_epochs_all
        
        # reshape
        if len(np.shape(raw._data))==3:
            DATA = np.reshape(np.transpose(DATA, (1,2,0)), (n_electrodes, n_samples*n_epochs), order='F')
            BCT = np.reshape(np.transpose(BCT, (1,2,0)), (n_electrodes, n_samples*n_epochs), order='F')
            IT = np.reshape(np.transpose(IT, (1,2,0)), (1, n_samples*n_epochs), order='F')
            IC = np.any(IC,axis=0)
            
        IT = np.reshape(IT, n_samples*n_epochs)
        IC = np.reshape(IC, n_electrodes)
        
        # initialize rejection matrix
        interpolation_matrix = np.full((n_electrodes, n_samples*n_epochs), False)

        # normalize the data so all electrodes have equal variance
        norm_electrodes = np.full((n_electrodes, 1), np.nan)
        for el in np.arange(n_electrodes):
            bt = BCT[el, :]==1
            d = DATA[el, :]
            if np.sum(bt==False) > 2:
                norm_electrodes[el] = np.nanstd(d[bt==False])
        idx = (np.isnan(norm_electrodes) | (norm_electrodes < 1e-9)) # 1e-3
        norm_electrodes[idx] = np.nanmean(norm_electrodes[idx==False])
        DATA = np.divide(DATA, np.tile(norm_electrodes, (1, n_samples*n_epochs)))
        DATA[np.isnan(DATA)] = 0

        # Apply PCA
        data_pca = DATA.copy()
        print('\n')
        for el in np.arange(n_electrodes):
    
            if IC[el]:
                
                # print info
                print('Electrode ', el + 1, ': ', end='')
    
                # index for the channel within the included ones
                good_electrode = np.sum(IC[0:el])
    
                # Find the segments to interpolate
                bad_if = _find_bad_segments_pca(BCT[el, :], IT, n_epochs, n_samples,
                                                mask_time * raw.info['sfreq'],
                                                max_time * raw.info['sfreq']
                                                )
    
                if np.size(bad_if) > 0:
    
                    print('Total bad data ', np.sum(BCT[el, :]), '(',
                          np.round(np.sum(BCT[el, :]) / (n_samples * n_epochs) * 100, 2), '%).',
                          'Data to apply PCA ', int(np.sum(bad_if[:, 1] - bad_if[:, 0])),
                          '(', np.round(np.sum(bad_if[:, 1] - bad_if[:, 0]) / (n_samples * n_epochs) * 100, 2),
                          '%).')
    
                    # Target PCA
                    d, tC = _target_PCA(DATA[IC, :], bad_if, components_to_remove, variance_to_remove, good_electrode)
                    d = d[good_electrode, :]
                    interpolation_matrix[el, tC] = True
    
                    # Splice the segments of data together
                    if splice_method:
                        epoch_i = np.arange(0, n_samples*n_epochs, n_samples)
                        epoch_f = epoch_i+n_samples
                        epoch_if = np.concatenate((epoch_i[:,np.newaxis], epoch_f[:,np.newaxis]), axis=1)
                        # BCT of this channel (bad times included) tells the
                        # trend method which samples are not good; corrected
                        # samples are taken from bad_if
                        d = _splice_segments(d, bad_if, epoch_if, bct=BCT[el, :], method=splice_method,
                                             window=int(round(1.0 * raw.info['sfreq'])),
                                             guard=int(round(0.02 * raw.info['sfreq'])),
                                             step_window=int(round(4.0 * raw.info['sfreq'])),
                                             step_min=int(round(1.0 * raw.info['sfreq'])),
                                             step_k=splice_step_k,
                                             null_step=int(round(0.1 * raw.info['sfreq'])),
                                             anchor_window=int(round(0.1 * raw.info['sfreq'])),
                                             undecidable=splice_undecidable, stats=stats)
    
                    # Store the data
                    data_pca[el, :] = d
                else:
                    print('No segments to apply target PCA were found.')
    
            else:
                print('Electrode ', el + 1, ': Bad channel.')

        # Rescale the data back
        data_pca = data_pca * np.tile(norm_electrodes, (1, n_samples * n_epochs))
        DATA = DATA * np.tile(norm_electrodes, (1, n_samples * n_epochs))

        # Reshape the data
        if len(np.shape(raw._data))==3:
            data_pca = np.reshape(data_pca, (n_electrodes, n_samples, n_epochs), order='F')
            data_pca = data_pca.transpose(2,0,1)
            interpolation_matrix = np.reshape(interpolation_matrix, (n_electrodes, n_samples, n_epochs), order='F')
            interpolation_matrix = interpolation_matrix.transpose(2,0,1)
            
        # If only some epochs were corrected put the corrected data in arrays the size of the original data
        if get_some_epochs and len(np.shape(raw._data))==3:
            data_pca_ = data_pca.copy()
            interpolation_matrix_ = interpolation_matrix.copy()
            data_pca = raw._data.copy()
            interpolation_matrix = np.full(np.shape(raw._data), False, dtype=bool)
            data_pca[IE,:,:] = data_pca_
            interpolation_matrix[IE,:,:] = interpolation_matrix_

        return data_pca, interpolation_matrix


class ChannelsSphericalSplineInterpolation(ArtCorrection):
    """Spherical spline interpolation for globally bad channels.

    Parameters
    ----------
    p : float, default=0.3
        Maximum allowed proportion of bad channels to attempt interpolation.
    p_neighbors : float, default=1
        Maximum proportion of bad neighbors accepted for interpolation.
    save_corrected : bool, default=True
        Whether to update correction masks in ``raw.artifacts``.
    verbose : bool, default=True
        Whether to print progress information.
    n_jobs : int, default=-1
        Number of parallel workers.
    """

    def __init__(self, p=0.3, p_neighbors=1, save_corrected=True, verbose=True, n_jobs=-1):
        
        super().__init__(verbose=verbose)
        
        # Arrange parameters for rejection
        self.params = dict(p=p, p_neighbors=p_neighbors, save_corrected=save_corrected, verbose=verbose, n_jobs=n_jobs)

    def correct(self, raw):         
        """Run channel-wise spline interpolation and update bookkeeping.

        Parameters
        ----------
        raw : RawAPICE
            Input object to correct in place.

        Returns
        -------
        None
            Data are modified in place.
        """
            
        _print_header('Performing Spherical Spline Interpolation (Bad channels)', separator="-")
        
        # Steps pre interpolation
        self.steps_pre_interpolation()
        
        # Interpolation channels
        raw._data, interpolation_matrix = self.spherical_spline_interpolation(raw, self.params['p'], self.params['p_neighbors'], n_jobs=self.params['n_jobs'])

        # Steps post interpolation
        self.steps_post_interpolation(raw, interpolation_matrix)
            
        print('\n')

    @staticmethod
    def spherical_spline_interpolation(raw, p, p_neighbors, n_jobs=-1):
        """Interpolate bad channels using spherical splines.

        Parameters
        ----------
        raw : RawAPICE
            Input raw/epochs object.
        p : float
            Maximum proportion of bad channels to allow interpolation.
        p_neighbors : float
            Maximum proportion of bad neighbors for eligible interpolation.
        n_jobs : int, default=-1
            Number of parallel workers.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Corrected data and interpolation mask.
        """

        n_electrodes, n_samples, n_epochs = raw.get_data_size()
        good_channels = np.reshape(raw.artifacts.BC,(n_epochs, n_electrodes, 1), order='F')==False
        if np.size(raw._data) > 0:

            # Positions, distances and adjacency in data channel order (computed once)
            print('\nExtracting electrode adjacency matrix.')
            positions, distance_matrix, adjacency_matrix = _channel_geometry(raw.info)

            # Initialize the interpolation matrix to False
            interpolation_matrix = np.full((n_epochs, n_electrodes, n_samples), False)

            # Copy and reshape the raw EEG data for processing
            eeg_data = raw._data.copy()
            eeg_data = np.reshape(eeg_data, (n_epochs, n_electrodes, n_samples), order='F')
            eeg_copy = raw.copy()
            eeg_copy._data = []
            eeg_copy._times = []
            
            # Reshape CCT to (n_epochs, n_electrodes, n_samples) for uniform indexing
            cct_reshaped = np.reshape(raw.artifacts.CCT, (n_epochs, n_electrodes, n_samples), order='F')

            # Loop through each epoch to handle bad channels
            for epochIndex in np.arange(n_epochs):
                # Identify bad channels and interpolate them
                channelsToInterpolate = ~good_channels[epochIndex][:, :][:,0]

                # Channels fully covered by CCT are unreliable references
                fully_interpolated_ep = np.all(cct_reshaped[epochIndex], axis=1)

                print('\nEpoch :', epochIndex + 1)

                if np.any(channelsToInterpolate):
                    start_time = time.time()
                    
                    print('Bad electrodes: ', [raw.info['ch_names'][i] for i in np.where(channelsToInterpolate)[0]])

                    # Get a copy of the EEG data and info
                    eeg_copy._data = eeg_data[epochIndex][:, :]                    
                    eeg_copy._times = raw.times.copy()

                    # all_bad_channels includes CCT-fully-interpolated channels so they are
                    # excluded from the reference pool of the spline
                    all_bad_channels_ep = channelsToInterpolate | fully_interpolated_ep

                    # Perform spline interpolation
                    if np.sum(all_bad_channels_ep) / n_electrodes <= p:
                        interpolated_data, interpolated_channels = _do_spherical_spline_interpolation(eeg_copy, 
                                                                                                distance_matrix,
                                                                                                positions,
                                                                                                adjacency_matrix, 
                                                                                                p_neighbors, 
                                                                                                channelsToInterpolate, 
                                                                                                all_bad_channels_ep,
                                                                                                True, n_jobs) 
                                                                                              
                        # Store data
                        for badElectrodeIndex in np.where(channelsToInterpolate)[0]:
                            if len(raw._data.shape) == 2:
                                raw._data[badElectrodeIndex, :] = interpolated_data[badElectrodeIndex, :]
                            elif len(raw._data.shape) == 3:
                                raw._data[epochIndex][badElectrodeIndex, :] = interpolated_data[badElectrodeIndex, :]
                            interpolation_matrix[epochIndex][badElectrodeIndex, :] = True

                        if np.all(interpolated_channels[channelsToInterpolate]):
                            print('--- All bad channels were interpolated.')
                            print(
                                  f"--- Elapsed time during whole channel interpolation: {time.time() - start_time} seconds\n"
                            )
                        elif np.any(interpolated_channels[channelsToInterpolate]):
                            print('--- Some channels were interpolated.')
                            print(
                                  f"--- Elapsed time during whole channel interpolation: {time.time() - start_time} seconds\n"
                            )
                    else:
                        print('--- No bad channels could be interpolated.')

                else:
                    print('No bad channels to interpolate.')

        # reshape if necessary         
        if len(raw._data.shape) == 2:
            interpolation_matrix = np.squeeze(np.transpose(interpolation_matrix,(1,2,0)), axis=2)
            
        return raw._data, interpolation_matrix



class SegmentSphericalSplineInterpolation(ArtCorrection):
    """Spherical spline interpolation applied on bad temporal segments.

    Parameters
    ----------
    n_jobs : int, default=-1
        Number of parallel workers.
    p : float, default=0.5
        Maximum proportion of bad channels allowed for interpolation per
        segment.
    p_neighbors : float, default=1
        Maximum proportion of bad neighbors for interpolation eligibility.
    min_good_time : float, default=1.00
        Minimum good segment duration in seconds.
    min_intertime : float, default=0.100
        Minimum bad segment duration in seconds for interpolation.
    mask_time : float, default=0.100
        Mask duration in seconds around bad segments.
    min_segment_time : float, default=0.200
        Minimum segment length in seconds for rejection regularization.
    splice_method : {1, 'trend', 'adaptive'} | None, default=1
        Splicing method after interpolation.
    splice_undecidable : {'method1', 'edges'}, default='method1'
        With 'adaptive', how to splice segments without enough good data to
        decide whether they are steps (see ``_splice_segments``).
    splice_step_k : float, default=2.5
        With 'adaptive', a segment is a step if the before/after level
        difference exceeds this many robust SDs (see ``_splice_segments``).
    save_corrected : bool, default=True
        Whether to update correction masks in ``raw.artifacts``.
    verbose : bool, default=True
        Whether to print progress information.
    parallelize_mode : {'auto', 'channels', 'segments'}, default='auto'
        Parallelization strategy.
    """

    def __init__(self, n_jobs=-1, p=0.5, p_neighbors=1, min_good_time=1.00, min_intertime=0.100, mask_time=0.100, 
                 min_segment_time=0.200, splice_method=1, save_corrected=True, verbose=True,
                 parallelize_mode='auto', splice_undecidable='method1', splice_step_k=2.5):
        
        super().__init__(verbose=verbose)

        valid_parallelize_mode = {'auto', 'channels', 'segments'}
        if parallelize_mode not in valid_parallelize_mode:
            raise ValueError(
                "parallelize_mode must be one of {'auto', 'channels', 'segments'}, "
                f"got {parallelize_mode!r}"
            )
        
        # Arreange parameters for rejection
        self.params = dict(n_jobs=n_jobs, p=p, p_neighbors=p_neighbors, min_good_time=min_good_time,
                           min_intertime=min_intertime, mask_time=mask_time, 
                           min_segment_time=min_segment_time,
                           splice_method=splice_method, splice_undecidable=splice_undecidable,
                           splice_step_k=splice_step_k,
                           save_corrected=save_corrected, verbose=verbose,
                           parallelize_mode=parallelize_mode)
    
    def correct(self, raw):
        """Run segment-wise spline interpolation and update bookkeeping.

        Parameters
        ----------
        raw : RawAPICE
            Input object to correct in place.

        Returns
        -------
        None
            Data are modified in place.
        """
        
        _print_header('Performing Spherical Spline Interpolation per Segment', separator="-")

        # Steps pre interpolation
        self.steps_pre_interpolation()
        
        # Interpolation of Spatial Segments
        self._splice_stats_tmp = {}
        raw._data, interpolation_matrix = self._spherical_spline_interpolation(raw)

        # Steps post interpolation
        self.steps_post_interpolation(raw, interpolation_matrix)
        self._report_splice_stats(raw, self._splice_stats_tmp, 'spline segments')
            
        print('\n')

    def _spherical_spline_interpolation(self, raw):
        """Interpolate bad channel-time segments using spherical splines.

        Parameters
        ----------
        raw : RawAPICE
            Input raw/epochs object.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            Interpolated data and interpolation mask.
        """

        n_channels, n_samples, n_epochs = raw.get_data_size()

        # Find the segments to interpolate
        print('\nDefining data to interpolate...')
        bct_to_interpolate = _build_interpolation_matrix_spline(raw, self.params['min_good_time'], self.params['min_intertime'], self.params['mask_time'], self.params['min_segment_time'])

        # Positions, distances and adjacency in data channel order (computed once)
        print('\nExtracting electrode adjacency matrix.')
        positions, distance_matrix, adjacency_matrix = _channel_geometry(raw.info)


        badIntervals = []
        chaInterpolate = []
        allBadChannels = []
        for ep in np.arange(n_epochs):
            
            # bad channels in that epoch are not interpolated
            if len(np.shape(raw._data))==2:
                badChannelsPerEpoch = np.squeeze(raw.artifacts.BC[:,0]) | raw.artifacts.BCmanual
                # channels fully covered by CCT are unreliable references
                fully_interpolated_ep = np.all(raw.artifacts.CCT, axis=1)
            elif len(np.shape(raw._data))==3:
                badChannelsPerEpoch = np.squeeze(raw.artifacts.BC[ep,:,0]) | raw.artifacts.BCmanual
                # channels fully covered by CCT are unreliable references
                fully_interpolated_ep = np.all(raw.artifacts.CCT[ep], axis=1)
            badChannelsPerEpoch |= fully_interpolated_ep

            # data to interpolate
            if len(np.shape(raw._data))==2:
                bct_to_interpolate_ep = bct_to_interpolate
            elif len(np.shape(raw._data))==3:
                bct_to_interpolate_ep = np.squeeze(bct_to_interpolate[ep,:,:])
                
            # Identify the segments and channels to interpolate
            badIntervals_ep, chaInterpolate_ep = _find_segments_interpolation_spline(bct_to_interpolate_ep, badChannelsPerEpoch)
            for segmentIndex in np.arange(np.shape(badIntervals_ep)[0]):
                allBadChannels_ep = chaInterpolate_ep[segmentIndex] | badChannelsPerEpoch
                # Transform to indexes across epochs
                badIntervals.append((badIntervals_ep[segmentIndex] + ep * n_samples).astype(int))
                chaInterpolate.append(chaInterpolate_ep[segmentIndex])
                allBadChannels.append(allBadChannels_ep)

        if len(badIntervals) == 0:
            interpolation_matrix = np.zeros_like(raw._data, dtype=bool)
            return raw._data.copy(), interpolation_matrix

        badIntervals = np.asarray(badIntervals, dtype=int)
        
        # Initialize matrices
        raw_data = raw._data.copy()
        if len(raw._data.shape) == 3:
            raw_data_2d = np.transpose(raw_data, (1, 2, 0))
            raw_data_2d = np.reshape(raw_data_2d, (n_channels, n_samples * n_epochs), order='F')
        else:
            raw_data_2d = raw_data.copy()

        interpolatedData = raw_data_2d.copy()
        interpolationMatrix = np.full((n_channels, n_samples * n_epochs), False, dtype=bool)

        # Initialize arrays for storing segment indices
        initialSegmentIndices = []
        finalSegmentIndices = []
        
        requested_mode = self.params['parallelize_mode']
        selected_mode = requested_mode
        if requested_mode == 'auto':
            n_segments = int(np.shape(badIntervals)[0])
            n_jobs = self.params['n_jobs']
            n_jobs = os.cpu_count() or 1 if n_jobs in (-1, None) else max(1, int(n_jobs))
            avg_channels_to_interpolate = np.mean([np.sum(x) for x in chaInterpolate]) if n_segments else 0
            max_channels_allowed = self.params['p'] * n_channels
            if n_segments >= max(8, 2 * n_jobs) and avg_channels_to_interpolate <= max_channels_allowed:
                selected_mode = 'segments'
            else:
                selected_mode = 'channels'
            print(f"Auto parallelization selected mode: {selected_mode}")

        # Initialize progress bar
        start_time = time.time()
        if selected_mode == 'channels':
            widgets = ['Interpolating...', Percentage(), Bar()]
            bar = ProgressBar(maxval=int(np.shape(badIntervals)[0] - 1), widgets=widgets)
            bar.start()

            for segmentIndex in np.arange(np.shape(badIntervals)[0]):
                segmentStart = int(badIntervals[segmentIndex, 0])
                segmentEnd = int(badIntervals[segmentIndex, 1])
                bad_channels_to_interpolate = chaInterpolate[segmentIndex]
                all_bad_channels = allBadChannels[segmentIndex]

                if (np.sum(all_bad_channels) / n_channels) <= self.params['p']:
                    seg_start, seg_end, bad_ch, bad_rows = _interpolate_spline_segment_task(
                        raw,
                        raw_data_2d,
                        segmentStart,
                        segmentEnd,
                        bad_channels_to_interpolate,
                        all_bad_channels,
                        distance_matrix,
                        positions,
                        adjacency_matrix,
                        self.params['p_neighbors'],
                        self.params['n_jobs'],
                    )
                    if np.size(bad_ch) > 0:
                        initialSegmentIndices.append(seg_start)
                        finalSegmentIndices.append(seg_end)
                        interpolationMatrix[bad_ch, seg_start:seg_end] = True
                        interpolatedData[bad_ch, seg_start:seg_end] = bad_rows

                bar.update(segmentIndex)

            bar.finish()
        else:
            eligible_segments = [
                idx for idx in np.arange(np.shape(badIntervals)[0])
                if (np.sum(allBadChannels[idx]) / n_channels) <= self.params['p']
            ]

            print(f"Interpolating segments in parallel: {len(eligible_segments)} eligible / {np.shape(badIntervals)[0]} total")
            if len(eligible_segments) > 0:
                results = Parallel(n_jobs=self.params['n_jobs'], prefer='threads')(
                    delayed(_interpolate_spline_segment_task)(
                        raw,
                        raw_data_2d,
                        int(badIntervals[idx, 0]),
                        int(badIntervals[idx, 1]),
                        chaInterpolate[idx],
                        allBadChannels[idx],
                        distance_matrix,
                        positions,
                        adjacency_matrix,
                        self.params['p_neighbors'],
                        1,
                    )
                    for idx in eligible_segments
                )

                for seg_start, seg_end, bad_ch, bad_rows in results:
                    if np.size(bad_ch) == 0:
                        continue
                    initialSegmentIndices.append(seg_start)
                    finalSegmentIndices.append(seg_end)
                    interpolationMatrix[bad_ch, seg_start:seg_end] = True
                    interpolatedData[bad_ch, seg_start:seg_end] = bad_rows

        print('--- Elapsed time during interpolation: ', time.time() - start_time, 'seconds')
        print('--- Percentage of interpolated data: ', np.round(np.sum(interpolationMatrix[:]) / np.size(interpolationMatrix) * 100, 2), '%')
        
        # reshape
        if len(np.shape(raw._data))==3:
            interpolatedData = np.reshape(interpolatedData, (n_channels, n_samples, n_epochs), order='F')
            interpolatedData = np.transpose(interpolatedData, (2, 0, 1))
            interpolationMatrix = np.reshape(interpolationMatrix, (n_channels, n_samples, n_epochs), order='F')
            interpolationMatrix = np.transpose(interpolationMatrix, (2, 0, 1))
            
        if len(np.shape(raw._data))==2:
            interpolatedData = np.reshape(interpolatedData, (n_epochs, n_channels, n_samples), order='F')
            interpolationMatrix = np.reshape(interpolationMatrix, (n_epochs, n_channels, n_samples), order='F')
 

        # Splice the interpolated segments together
        initialSegmentIndices = np.asarray(initialSegmentIndices, dtype=int)
        finalSegmentIndices = np.asarray(finalSegmentIndices, dtype=int)
        print('Splicing segments...')
        for ep in np.arange(n_epochs):
            int_i = interpolationMatrix[ep, :, :]
            # get the segments to splice in that epoch
            segments_i = initialSegmentIndices[(initialSegmentIndices < n_samples*(ep+1)) & (initialSegmentIndices >= n_samples*ep)]
            segments_i = segments_i - ep * n_samples
            segments_f = finalSegmentIndices[(finalSegmentIndices <= n_samples*(ep+1)) & (finalSegmentIndices > n_samples*ep)]
            segments_f = segments_f - ep * n_samples
            bad_if = np.asarray([segments_i, segments_f], dtype=int).T
            if np.size(bad_if)>0:
                d_i = np.squeeze(interpolatedData[ep,:,:])
                epoch_if = np.asarray([0, n_samples])
                # copies: np.squeeze and basic indexing return views, so the
                # line below used to clear raw.artifacts.BCT itself
                if len(np.shape(raw._data))==2:
                    bct_i = np.squeeze(raw.artifacts.BCT).copy()
                    bc_i = raw.artifacts.BC.copy()
                    bt_i = raw.artifacts.BT.copy()
                else:
                    bct_i = np.squeeze(raw.artifacts.BCT[ep,:,:]).copy()
                    bc_i = np.squeeze(raw.artifacts.BC[ep,:,0]).copy()
                    bt_i = np.squeeze(raw.artifacts.BT[ep,0,:]).copy()
                bct_i[int_i] = 0
                # splice_method was accepted but ignored (always 1) before
                sfreq = raw.info['sfreq']
                dspliced = _splice_segments(d_i, bad_if, epoch_if, bct=bct_i, bt=bt_i, bc=bc_i,
                                            method=self.params['splice_method'],
                                            corrected=int_i,
                                            window=int(round(1.0 * sfreq)),
                                            guard=int(round(0.02 * sfreq)),
                                            step_window=int(round(4.0 * sfreq)),
                                            step_min=int(round(1.0 * sfreq)),
                                            step_k=self.params.get('splice_step_k', 2.5),
                                            null_step=int(round(0.1 * sfreq)),
                                            anchor_window=int(round(0.1 * sfreq)),
                                            undecidable=self.params.get('splice_undecidable', 'method1'),
                                            stats=getattr(self, '_splice_stats_tmp', None))
                interpolatedData[ep,:,:] = dspliced
           
        # reshape if necessary         
        if len(np.shape(raw._data))==2:
            interpolationMatrix = np.squeeze(np.transpose(interpolationMatrix,(1,2,0)), axis=2)
            interpolatedData = np.squeeze(np.transpose(interpolatedData,(1,2,0)), axis=2)
        
        return interpolatedData, interpolationMatrix
    
