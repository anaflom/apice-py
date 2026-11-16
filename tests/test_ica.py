"""ICA cleaning removes the artifact components only (bug review, ica.py)."""

import warnings

import mne
import numpy as np
import pytest

from apice.ica import _ica_artifact_estimate

CHS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz",
       "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "PO3", "PO4", "O1",
       "Oz", "O2", "AF3", "AF4"]


@pytest.fixture(scope="module")
def blink_case():
    rng = np.random.default_rng(0)
    sf, n_t = 250.0, 250 * 60
    mont = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(CHS, sf, "eeg")
    info.set_montage(mont)
    # more brain sources than ICA components, so part of the data is not modelled by the ICA
    brain = rng.normal(size=(len(CHS), 60)) @ np.cumsum(rng.normal(size=(60, n_t)), axis=1)
    brain = mne.filter.filter_data(brain, sf, 1.0, 40.0, verbose="ERROR")
    brain *= 10e-6 / brain.std()
    pos = np.array([mont.get_positions()["ch_pos"][c] for c in CHS])
    topo = np.exp(-((pos[:, 1] - pos[:, 1].max()) ** 2) / 0.002)
    blink_t = np.zeros(n_t)
    for on in np.arange(500, n_t - 200, 1000):
        blink_t[on:on + 75] += np.hanning(75)
    raw = mne.io.RawArray(brain + np.outer(topo, blink_t) * 150e-6, info, verbose=False)
    ica = mne.preprocessing.ICA(n_components=15, random_state=0, max_iter=1000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # FastICA convergence warnings are irrelevant here
        ica.fit(raw, verbose=False)
    src = ica.get_sources(raw).get_data()
    bad = [int(np.argmax([abs(np.corrcoef(s, blink_t)[0, 1]) for s in src]))]
    return raw, brain, ica, bad


def test_artifact_estimate_keeps_unmodelled_signal(blink_case):
    raw, brain, ica, bad = blink_case
    clean = raw.get_data() - _ica_artifact_estimate(ica, raw, bad).get_data()
    # the old include=bad estimate kept only ~75 % of the brain variance here
    assert np.var(clean) / np.var(brain) > 0.9
    err = np.sqrt(np.mean((clean - brain) ** 2)) / np.sqrt(np.mean(brain ** 2))
    assert err < 0.4


def test_artifact_estimate_with_no_component_is_zero(blink_case):
    raw, _, ica, _ = blink_case
    np.testing.assert_allclose(_ica_artifact_estimate(ica, raw, []).get_data(), 0, atol=1e-18)
