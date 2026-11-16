"""Behavioural tests of artifact detection against synthetic ground truth.

Only the public ``RawAPICE`` API and the packaged default configurations are
used; the tests read ``raw.artifacts.BCT`` / ``BC`` through small accessor
helpers so they only need updating in one place if the storage changes.

Thresholds on recall / precision are deliberately loose: they guard against
regressions, while the exact numbers are recorded with the ``baseline``
fixture and printed at the end of the run.
"""

import numpy as np
import pytest

from apice import RawAPICE
from apice.utils import get_cfg
from conftest import quiet
from synthetic import SyntheticEEG, sample_metrics

MOTION_CFG = "detect_artifacts_motion_config.json"


# ---------------------------------------------------------------- accessors
def bct(raw):
    """Bad channel-time mask, (n_channels, n_samples) bool."""
    return np.asarray(raw.artifacts.BCT, dtype=bool)


def bad_channels(raw):
    return [raw.ch_names[i] for i in np.where(np.asarray(raw.artifacts.BC).ravel())[0]]


def to_apice(raw):
    return quiet(RawAPICE, raw)


def detect_motion(r):
    quiet(r.detect_artifacts, cfg=get_cfg(None, MOTION_CFG))


# ---------------------------------------------------------------- bad channels
@pytest.mark.parametrize("n_channels", [32, 128])
def test_bad_channels_found(n_channels, baseline):
    s = SyntheticEEG(n_channels, duration=60, seed=11)
    names = s.ch_names
    flat, noisy, lowcorr = names[3], names[n_channels // 2], names[-4]
    s.add_flat(flat)
    s.add_noisy(noisy)
    s.add_lowcorr(lowcorr)
    raw, truth = s.build()
    r = to_apice(raw)
    quiet(r.detect_bad_channels)
    found = bad_channels(r)
    for ch in truth.bad_channels:
        assert ch in found, f"{ch} not detected (found {found})"
    false_pos = set(found) - set(truth.bad_channels)
    assert len(false_pos) <= max(1, n_channels // 32), false_pos
    bc = np.isin(r.ch_names, found)
    baseline(f"bad_channels/{n_channels}ch",
             sample_metrics(bc[:, None], truth.bad_channel_mask[:, None]))


@pytest.mark.parametrize("n_channels", [32, 128])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bad_channels_clean_data(n_channels, seed):
    raw, _ = SyntheticEEG(n_channels, duration=60, seed=seed).build()
    r = to_apice(raw)
    quiet(r.detect_bad_channels)
    assert len(bad_channels(r)) <= max(1, n_channels // 32)


@pytest.mark.parametrize("kind", ["flat", "noisy", "lowcorr"])
def test_bad_channel_each_kind(kind):
    s = SyntheticEEG(32, duration=60, seed=4)
    getattr(s, f"add_{kind}")("C4")
    raw, _ = s.build()
    r = to_apice(raw)
    quiet(r.detect_bad_channels)
    assert "C4" in bad_channels(r)


def test_bad_channels_low_density_runs():
    """4-channel montage: detection must run and find a flat channel."""
    s = SyntheticEEG(4, duration=60, seed=0)
    s.add_flat("Pz")
    raw, _ = s.build()
    r = to_apice(raw)
    quiet(r.detect_bad_channels)
    assert "Pz" in bad_channels(r)


# ---------------------------------------------------------------- glitches
def _glitch_sim(n_channels=32, seed=2):
    s = SyntheticEEG(n_channels, duration=60, seed=seed)
    rng = np.random.default_rng(seed + 100)
    durations = [0.01, 0.02, 0.05, 0.1, 0.03, 0.08, 0.015, 0.06]
    for i, d in enumerate(durations):
        n_ch = int(rng.integers(1, 4))
        chs = list(rng.choice(s.ch_names, n_ch, replace=False))
        s.add_glitch(chs, onset=3 + 7 * i, duration=d)
    return s


@pytest.mark.parametrize("n_channels", [32, 128])
def test_glitch_detection(n_channels, baseline):
    raw, truth = _glitch_sim(n_channels).build()
    r = to_apice(raw)
    quiet(r.detect_glitches)
    m = baseline(f"glitch/{n_channels}ch", sample_metrics(bct(r), truth.mask))
    assert m["recall"] >= 0.8
    assert m["precision"] >= 0.02
    # every single glitch is at least partly found
    for ev in truth.events:
        hit = bct(r)[list(ev.channels), ev.start:ev.stop].mean()
        assert hit > 0.5, (ev, hit)


def test_glitch_detection_clean_fpr(baseline):
    raw, truth = SyntheticEEG(32, duration=60, seed=2).build()
    r = to_apice(raw)
    quiet(r.detect_glitches)
    m = baseline("glitch/clean-32ch", sample_metrics(bct(r), truth.mask))
    assert m["fpr"] <= 0.01


# ---------------------------------------------------------------- motion / blinks
def _motion_sim(n_channels=32, seed=3):
    s = SyntheticEEG(n_channels, duration=60, seed=seed)
    s.add_motion("Cz", onset=5, duration=1.0)
    s.add_motion("O1", onset=20, duration=2.5, amplitude=300e-6)
    s.add_motion("F3", onset=40, duration=0.5)
    s.add_blink(onset=30)
    s.add_blink(onset=50)
    return s


@pytest.mark.parametrize("n_channels", [32, 128])
def test_motion_detection(n_channels, baseline):
    raw, truth = _motion_sim(n_channels).build()
    r = to_apice(raw)
    detect_motion(r)
    m = baseline(f"motion+blink/{n_channels}ch", sample_metrics(bct(r), truth.mask))
    assert m["recall"] >= 0.7
    assert m["precision"] >= 0.3
    for kind in ("motion", "blink"):
        baseline(f"motion+blink/{n_channels}ch/{kind}-recall",
                 sample_metrics(bct(r), truth.mask_of(kind), ignore=truth.mask & ~truth.mask_of(kind)))


def test_motion_detection_clean_fpr(baseline):
    raw, truth = SyntheticEEG(32, duration=60, seed=3).build()
    r = to_apice(raw)
    detect_motion(r)
    m = baseline("motion/clean-32ch", sample_metrics(bct(r), truth.mask))
    assert m["fpr"] <= 0.01


@pytest.mark.parametrize("placement", ["start", "end", "between", "adjacent"])
def test_motion_placement(placement, baseline):
    """Artifacts at the recording edges or touching another bad segment are found."""
    s = SyntheticEEG(32, duration=40, seed=5)
    if placement == "start":
        target = s.add_motion("Cz", onset="start", duration=1.5)
    elif placement == "end":
        target = s.add_motion("Cz", onset="end", duration=1.5)
    elif placement == "between":
        s.add_motion("Pz", onset=10, duration=1.0)
        target = s.add_motion("Cz", onset=12, duration=1.5)  # 1 s of good data on each side
        s.add_motion("Fz", onset=14.5, duration=1.0)
    else:  # adjacent: blink immediately followed by a motion artifact
        first = s.add_blink(onset=20)
        target = s.add_motion("Cz", after=first, duration=1.5)
    raw, truth = s.build()
    r = to_apice(raw)
    detect_motion(r)
    ev = truth.events[target]
    t = np.zeros_like(truth.mask)
    t[list(ev.channels), ev.start:ev.stop] = True
    m = baseline(f"motion-placement/{placement}",
                 sample_metrics(bct(r), t, ignore=truth.mask & ~t))
    assert m["recall"] >= 0.7, m
