"""Tests of the synthetic generator itself (no APICE involved)."""

import numpy as np
import pytest

from synthetic import SyntheticEEG, sample_metrics


@pytest.mark.parametrize("n_channels", [4, 32, 128])
def test_montage_and_shape(n_channels):
    raw, truth = SyntheticEEG(n_channels, duration=10, seed=0).build()
    assert len(raw.ch_names) == n_channels
    assert raw.get_montage() is not None
    pos = raw.get_montage().get_positions()["ch_pos"]
    assert set(pos) == set(raw.ch_names)
    assert truth.mask.shape == (n_channels, raw.n_times)
    assert not truth.mask.any()
    rms = raw.get_data().std(axis=1)
    np.testing.assert_allclose(rms, 15e-6, rtol=1e-6)


def test_deterministic():
    def make(seed):
        s = SyntheticEEG(32, duration=10, seed=seed)
        s.add_glitch(["Cz"], onset=2, duration=0.05)
        s.add_motion("O1", onset=5, duration=1.0)
        s.add_blink(onset=8)
        return s.build()[0].get_data()

    np.testing.assert_array_equal(make(3), make(3))
    assert not np.allclose(make(3), make(4))


def test_background_is_spatially_correlated():
    raw, _ = SyntheticEEG(32, duration=30, seed=0).build()
    c = np.corrcoef(raw.get_data())
    np.fill_diagonal(c, np.nan)
    best = np.nanmax(c, axis=0)
    assert np.median(best) > 0.7, "neighbouring channels should correlate"
    assert np.nanmedian(c) < 0.9, "channels should not be copies of each other"


def test_artifacts_confined_to_their_masks():
    """Artifact injection leaves the background untouched outside the mask."""
    clean, _ = SyntheticEEG(32, duration=20, seed=7).build()
    s = SyntheticEEG(32, duration=20, seed=7)
    s.add_flat("T7")
    s.add_noisy("P4", onset=2, duration=3)
    s.add_lowcorr("O2", onset=6, duration=2)
    s.add_glitch(["Cz", "C3"], onset=9, duration=0.03)
    s.add_motion("F4", onset=11, duration=2.0)
    s.add_blink(onset=15)
    raw, truth = s.build()
    diff = raw.get_data() != clean.get_data()
    assert not (diff & ~truth.mask).any()
    # and every artifact actually changed (almost) all of its samples
    for ev in truth.events:
        changed = diff[list(ev.channels), ev.start:ev.stop].mean()
        assert changed > 0.95, (ev.kind, changed)
    assert truth.bad_channels == ["T7"]


def test_placement_controls():
    s = SyntheticEEG(32, duration=10, seed=0)
    i_start = s.add_motion("Cz", onset="start", duration=1.0)
    i_adj = s.add_glitch(["Cz"], after=i_start, duration=0.05)
    i_mid = s.add_blink(onset="middle", duration=0.3)
    i_end = s.add_motion("Pz", onset="end", duration=0.5)
    _, truth = s.build()
    ev = truth.events
    assert ev[i_start].start == 0
    assert ev[i_adj].start == ev[i_start].stop
    assert ev[i_end].stop == truth.n_times
    mid = (ev[i_mid].start + ev[i_mid].stop) / 2
    assert abs(mid - truth.n_times / 2) <= 1
    with pytest.raises(ValueError):
        s.add_glitch(["Cz"], onset=9.99, duration=0.05)


def test_first_samp():
    raw, _ = SyntheticEEG(32, duration=5, seed=0, first_samp=1234).build()
    assert raw.first_samp == 1234


def test_sample_metrics():
    t = np.array([[1, 1, 0, 0]], bool)
    d = np.array([[1, 0, 1, 0]], bool)
    m = sample_metrics(d, t)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["recall"] == m["precision"] == 0.5
