"""Statistical measures for ERP data quality assessment."""

import numpy as np
from mne import BaseEpochs

try:  # same channel resolution as Epochs.get_data()
    from mne._fiff.pick import _picks_to_idx
except ImportError:  # older MNE
    from mne.io.pick import _picks_to_idx


def compute_sme(
    epochs,
    condition=None,
    start=None,
    stop=None,
    picks=None,
    roi=None,
    n_iter=1000,
    n_samples=None,
    relative=False,
    random_state=None,
    return_ch_names=False,
):
    """Compute the Standardized Measurement Error (SME) via bootstrapping.

    The SME :footcite:`LuckEtAl2021` is the standard error of measurement for
    an ERP amplitude score, estimated by bootstrapping single trials. It
    quantifies data quality (precision) for the specific score being measured:
    the lower the SME, the more precise the estimate.

    For a single condition the SME is the standard deviation of the bootstrap
    distribution of the condition mean. For two conditions the SME is the
    standard deviation of the bootstrap distribution of the difference between
    condition means, which is conceptually equivalent to averaging each
    bootstrap resample and subtracting with
    ``mne.combine_evoked([boot_evoked1, boot_evoked2], weights=[1, -1])``.

    Parameters
    ----------
    epochs : mne.BaseEpochs
        Epochs object containing the data. All conditions should be present
        when ``condition`` is specified.
    condition : str | list of str | None
        Condition(s) to use for the SME computation:

        - ``None``: use all epochs ('mean' mode).
        - ``str``: select ``epochs[condition]`` ('mean' mode).
        - ``list`` of two strings: compute the SME of the difference between
          the two conditions, i.e. ``condition[0]`` minus ``condition[1]``
          ('diff' mode).

    start : float | None
        Start of the scoring time window in seconds. If ``None``, uses the
        start of the epoch.
    stop : float | None
        End of the scoring time window in seconds. If ``None``, uses the end
        of the epoch.
    picks : str | list | slice | None
        Channel selection passed to ``epochs.get_data()``. Used only when
        ``roi`` is ``None``. If ``None``, all EEG channels (``'eeg'``; MNE's
        own default would also take EOG / ECG channels). Ignored when ``roi``
        is set. As in MNE, a channel TYPE (e.g. ``'eeg'``) excludes the
        channels in ``info['bads']``, while channel NAMES do not; use
        ``return_ch_names=True`` to know which channels the values belong to.
    roi : list of str | None
        Channel names defining a region of interest. When set, data are first
        averaged across those channels and a single scalar SME is returned.
        When ``None`` (default), a per-channel SME array is returned.
    n_iter : int | None
        Number of bootstrap iterations. Default is 1000. If ``None``, the
        SME is computed analytically as ``SD / sqrt(N)`` for 'mean' mode, or
        ``sqrt(SD1² / N1 + SD2² / N2)`` for 'diff' mode, where SD is the
        standard deviation of per-trial scores and N is the number of trials.
        This is equivalent to the standard error of the mean.
    n_samples : int | None
        Number of trials to draw (with replacement) per bootstrap iteration.
        If ``None``, the number of trials of the condition (in 'diff' mode,
        each condition uses its own number of trials). Capped at the number
        of available trials per condition. Ignored when ``n_iter`` is None.
    relative : bool
        If ``True``, return the SME relative to the mean score (i.e., SME
        divided by the absolute value of the mean x 100). Default is ``False``.
        The mean score is computed on the original (non-resampled) data, so
        this only affects the final scaling of the SME. The absolute value is
        used in the denominator to avoid sign effects when the mean amplitude
        is negative. Unstable when the mean score is close to zero (e.g. a
        baseline-corrected window without a clear component).
    random_state : int | numpy.random.Generator | None
        Seed or generator for reproducibility. Passed to
        ``numpy.random.default_rng()``.
    return_ch_names : bool
        If ``True`` (per-channel mode only), also return the names of the
        channels the SME values belong to. Default is ``False``.

    Returns
    -------
    sme : float | numpy.ndarray, shape (n_channels,)
        Bootstrapped SME value(s). If ``relative=True``, divided by the mean
        score computed on the original (non-resampled) data. Returns a scalar
        ``float`` when ``roi`` is set, or a ``numpy.ndarray`` of shape
        ``(n_channels,)`` when operating in per-channel mode.
    ch_names : list of str
        Only with ``return_ch_names=True`` (per-channel mode).

    Notes
    -----
    By default the SME is computed by bootstrapping, following
    :footcite:t:`LuckEtAl2021`; for the time-window mean amplitude used here
    the analytic formula (``n_iter=None``, ``SD / sqrt(N)``) gives the same
    value up to bootstrap noise and is much faster.

    All epochs of ``epochs`` are used: with ``EpochsAPICE``, call
    ``remove_bad_epochs()`` (or select the good epochs) first.

    The score used here is the time-window mean amplitude, averaged across the
    time window defined by ``start`` and ``stop``. For per-channel mode each
    channel is scored independently; for ROI mode all ROI channels are
    averaged before scoring.

    **Bootstrapping procedure**

    For 'mean' mode:

    1. Compute per-trial scores: mean amplitude in the time window.
    2. For each of ``n_iter`` iterations:

       a. Sample ``n_samples`` trial indices *with replacement*.
       b. Compute the mean over sampled scores.
       c. Store the result.

    3. SME = standard deviation of the ``n_iter`` stored means.

    For 'diff' mode, step 2b becomes: compute the mean for each condition
    independently and subtract.

    References
    ----------
    .. footbibliography::

    Luck, S. J., Stewart, A. X., Simmons, A. M., & Rhemtulla, M. (2021).
    Standardized measurement error: A universal metric of data quality for
    averaged event-related potentials. *Psychophysiology*, 58, e13793.
    https://doi.org/10.1111/psyp.13793

    Examples
    --------
    Per-channel SME for a single condition over a 300–500 ms time window:

    >>> sme = compute_sme(epochs, condition="oddball", start=0.3, stop=0.5)
    >>> sme.shape  # (n_channels,)

    Scalar SME using a set of channels as a region of interest:

    >>> roi_channels = ["Pz", "P3", "P4"]
    >>> sme = compute_sme(epochs, condition="oddball",
    ...                   start=0.3, stop=0.5, roi=roi_channels)
    >>> float(sme)  # scalar

    SME of the difference between two conditions:

    >>> sme_diff = compute_sme(epochs, condition=["oddball", "standard"],
    ...                        start=0.3, stop=0.5)
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(epochs, BaseEpochs):
        raise TypeError(
            f"epochs must be an mne.BaseEpochs instance, got {type(epochs)}."
        )

    if condition is not None:
        if isinstance(condition, str):
            pass  # validated below when selecting
        elif isinstance(condition, (list, tuple)):
            if len(condition) != 2:
                raise ValueError(
                    "condition must be a string or a list/tuple of exactly "
                    f"two strings, got {len(condition)} elements."
                )
        else:
            raise TypeError(
                "condition must be a str, a list of two str, or None, "
                f"got {type(condition)}."
            )

    if roi is not None and not isinstance(roi, (list, tuple)):
        raise TypeError(
            f"roi must be a list of channel names or None, got {type(roi)}."
        )

    if n_iter is not None and n_iter < 1:
        raise ValueError(f"n_iter must be >= 1 or None, got {n_iter}.")

    # ------------------------------------------------------------------
    # Select epochs per condition
    # ------------------------------------------------------------------
    diff_mode = isinstance(condition, (list, tuple))

    if diff_mode:
        epochs1 = epochs[condition[0]]
        epochs2 = epochs[condition[1]]
    else:
        epochs1 = epochs[condition] if condition is not None else epochs
        epochs2 = None

    # ------------------------------------------------------------------
    # Helper: extract per-trial scores from an Epochs object.
    # Shape returned: (n_trials,) for ROI mode,
    #                 (n_trials, n_channels) for per-channel mode.
    # ------------------------------------------------------------------
    def _get_scores(ep):
        if roi is not None:
            # ROI mode: average across named channels, then across time
            data = ep.get_data(picks=roi, tmin=start, tmax=stop)
            # data: (n_trials, n_roi_channels, n_timepoints)
            return data.mean(axis=1).mean(axis=1)  # (n_trials,)
        else:
            # Per-channel mode: average across time only
            data = ep.get_data(picks=chan_picks, tmin=start, tmax=stop)
            # data: (n_trials, n_channels, n_timepoints)
            return data.mean(axis=2)  # (n_trials, n_channels)

    # resolve the channels once, with the rule epochs.get_data() uses (a
    # channel type excludes info['bads'], names do not), and select by name:
    # the values and the returned names then always match
    if roi is None:
        idx = _picks_to_idx(epochs1.info, "eeg" if picks is None else picks, exclude="bads")
        chan_picks = [epochs1.ch_names[i] for i in idx]
    scores1 = _get_scores(epochs1)
    n_trials1 = scores1.shape[0]
    n_s1 = n_trials1 if n_samples is None else min(n_trials1, int(n_samples))

    if diff_mode:
        scores2 = _get_scores(epochs2)
        n_trials2 = scores2.shape[0]
        n_s2 = (
            n_trials2 if n_samples is None else min(n_trials2, int(n_samples))
        )

    # ------------------------------------------------------------------
    # SME computation: analytic or bootstrap
    # ------------------------------------------------------------------
    if n_iter is None:
        # Analytic SME: standard error of the mean (SD / sqrt(N))
        if diff_mode:
            sme = np.sqrt(
                scores1.var(axis=0, ddof=1) / n_trials1
                + scores2.var(axis=0, ddof=1) / n_trials2
            )
        else:
            sme = scores1.std(axis=0, ddof=1) / np.sqrt(n_trials1)
    else:
        # Pre-compute trial indices (outside the loop — fixes MATLAB
        # efficiency issue where find(cond) was recomputed every iteration)
        idx1 = np.arange(n_trials1)
        if diff_mode:
            idx2 = np.arange(n_trials2)

        rng = np.random.default_rng(random_state)

        if roi is not None:
            # Scalar per iteration → M shape (n_iter,)
            M = np.empty(n_iter)
            for i in range(n_iter):
                sample1 = rng.choice(idx1, size=n_s1, replace=True)
                m1 = scores1[sample1].mean()
                if diff_mode:
                    sample2 = rng.choice(idx2, size=n_s2, replace=True)
                    m2 = scores2[sample2].mean()
                    M[i] = m1 - m2
                else:
                    M[i] = m1
        else:
            # Vector per iteration → M shape (n_iter, n_channels)
            n_channels = scores1.shape[1]
            M = np.empty((n_iter, n_channels))
            for i in range(n_iter):
                sample1 = rng.choice(idx1, size=n_s1, replace=True)
                m1 = scores1[sample1].mean(axis=0)  # (n_channels,)
                if diff_mode:
                    sample2 = rng.choice(idx2, size=n_s2, replace=True)
                    m2 = scores2[sample2].mean(axis=0)  # (n_channels,)
                    M[i] = m1 - m2
                else:
                    M[i] = m1

        # SME = standard deviation of the bootstrap distribution (ddof=1)
        sme = M.std(axis=0, ddof=1)

    # Relative SME: divide by the absolute mean score on the original data
    # (absolute value avoids sign flip when the mean amplitude is negative)
    if relative:
        if diff_mode:
            mean_score = scores1.mean(axis=0) - scores2.mean(axis=0)
        else:
            mean_score = scores1.mean(axis=0)
        sme = 100 * (sme / np.abs(mean_score))

    # Return scalar for ROI mode (M was 1-D, std returns a 0-D array)
    if roi is not None:
        return float(sme)

    if return_ch_names:
        return sme, list(chan_picks)
    return sme
