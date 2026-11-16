"""Synthetic EEG with injectable artifacts and known ground-truth masks.

The generator is deliberately independent of ``apice``: it only produces plain
``mne.io.RawArray`` objects plus boolean ground-truth masks, so it can be
reused unchanged after the APICE artifact-storage refactor and for systematic
evaluation of detection algorithms / thresholds.

Signal model
------------
* Spatially smooth background: ``n_sources`` latent 1/f^beta sources placed
  on a sphere, projected to the electrodes with a Gaussian distance kernel,
  plus a small amount of independent sensor noise.  Neighbouring channels are
  therefore correlated, as in real EEG.
* An alpha-like, amplitude-modulated oscillation with a posterior topography.
* Each channel is scaled to ``rms`` volts.

Artifacts (all positions in seconds, all masks are ``n_channels x n_samples``)
---------------------------------------------------------------------------
``flat``      channel replaced by a constant
``noisy``     high-amplitude broadband noise added to a channel
``lowcorr``   channel replaced by an independent 1/f signal (same RMS)
``glitch``    short rectangular step/jump (10-100 ms) on one or more channels
``motion``    large slow multi-channel transient (0.5-3 s)
``blink``     frontal monophasic deflection (~300 ms)

Placement controls: ``onset`` may be a float (seconds) or one of
``'start'`` (first sample), ``'end'`` (ends at the last sample) or
``'middle'``; ``after=<event index>`` places an artifact immediately after a
previously added one (adjacent bad segments).

Everything is deterministic given ``seed``.

Example
-------
>>> sim = SyntheticEEG(n_channels=32, duration=60, seed=1)
>>> sim.add_flat("T7"); sim.add_glitch(["Cz"], onset=12.0, duration=0.05)
>>> raw, truth = sim.build()
>>> truth.mask.shape == (32, raw.n_times)
True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import mne
import numpy as np

__all__ = [
    "ArtifactEvent",
    "GroundTruth",
    "SyntheticEEG",
    "MONTAGES",
    "pink_noise",
    "sample_metrics",
]

# n_channels -> (montage name, channel subset or None for all)
MONTAGES = {
    4: ("biosemi16", ["Fz", "Cz", "Pz", "Oz"]),
    32: ("biosemi32", None),
    64: ("biosemi64", None),
    128: ("biosemi128", None),
}

WHOLE_CHANNEL_KINDS = ("flat", "noisy", "lowcorr")


# ----------------------------------------------------------------------------
# Ground truth containers
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ArtifactEvent:
    """One injected artifact.

    ``start``/``stop`` are sample indices (``stop`` exclusive); ``channels``
    are channel indices.
    """

    kind: str
    channels: tuple[int, ...]
    start: int
    stop: int
    params: dict = field(default_factory=dict, compare=False)

    @property
    def n_samples(self) -> int:
        return self.stop - self.start


@dataclass
class GroundTruth:
    ch_names: list[str]
    sfreq: float
    n_times: int
    events: list[ArtifactEvent]

    def mask_of(self, kinds: str | Sequence[str] | None = None) -> np.ndarray:
        """Boolean ``(n_channels, n_times)`` mask of the selected artifact kinds."""
        if isinstance(kinds, str):
            kinds = (kinds,)
        m = np.zeros((len(self.ch_names), self.n_times), dtype=bool)
        for ev in self.events:
            if kinds is None or ev.kind in kinds:
                m[np.asarray(ev.channels)[:, None], np.arange(ev.start, ev.stop)[None, :]] = True
        return m

    @property
    def mask(self) -> np.ndarray:
        """Mask of every injected artifact."""
        return self.mask_of(None)

    @property
    def bad_channels(self) -> list[str]:
        """Channels whose artifact covers the whole recording (flat/noisy/lowcorr)."""
        bad = set()
        for ev in self.events:
            if ev.kind in WHOLE_CHANNEL_KINDS and ev.start == 0 and ev.stop == self.n_times:
                bad.update(ev.channels)
        return [self.ch_names[i] for i in sorted(bad)]

    @property
    def bad_channel_mask(self) -> np.ndarray:
        bc = np.zeros(len(self.ch_names), dtype=bool)
        bc[[self.ch_names.index(c) for c in self.bad_channels]] = True
        return bc


# ----------------------------------------------------------------------------
# Signal helpers
# ----------------------------------------------------------------------------
def pink_noise(rng: np.random.Generator, n_signals: int, n_times: int, sfreq: float,
               beta: float = 1.5, f_lo: float = 0.5, f_hi: float = 45.0) -> np.ndarray:
    """Unit-variance 1/f^beta noise, band-limited to ``[f_lo, f_hi]`` with soft edges."""
    freqs = np.fft.rfftfreq(n_times, 1.0 / sfreq)
    spec = rng.standard_normal((n_signals, freqs.size)) + 1j * rng.standard_normal((n_signals, freqs.size))
    f = np.maximum(freqs, freqs[1] if freqs.size > 1 else 1.0)
    amp = f ** (-beta / 2.0)
    amp *= 1.0 / (1.0 + (f_lo / f) ** 4)  # soft high-pass
    amp *= 1.0 / (1.0 + (f / f_hi) ** 8)  # soft low-pass
    amp[0] = 0.0
    x = np.fft.irfft(spec * amp, n=n_times, axis=1)
    x -= x.mean(axis=1, keepdims=True)
    x /= x.std(axis=1, keepdims=True)
    return x


def _tukey(n: int, alpha: float = 0.25) -> np.ndarray:
    from scipy.signal.windows import tukey
    return tukey(n, alpha) if n > 1 else np.ones(n)


def sample_metrics(detected: np.ndarray, truth: np.ndarray,
                   ignore: np.ndarray | None = None) -> dict:
    """Sample-level confusion metrics between two boolean masks.

    ``ignore`` marks samples excluded from the evaluation (e.g. other injected
    artifacts that are not the target of the test).
    """
    detected = np.asarray(detected, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    keep = np.ones_like(truth) if ignore is None else ~np.asarray(ignore, dtype=bool)
    tp = int(np.sum(detected & truth & keep))
    fp = int(np.sum(detected & ~truth & keep))
    fn = int(np.sum(~detected & truth & keep))
    tn = int(np.sum(~detected & ~truth & keep))
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, recall=recall, precision=precision, fpr=fpr)


# ----------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------
class SyntheticEEG:
    """Builder for a synthetic recording with injected artifacts.

    Parameters
    ----------
    n_channels : int
        One of ``MONTAGES`` (4, 32, 64, 128) unless ``montage``/``ch_names`` given.
    duration : float
        Seconds.
    sfreq : float
        Sampling rate.
    seed : int
        Seed; the same seed and the same sequence of ``add_*`` calls give
        bit-identical output.
    rms : float
        Background RMS per channel, in volts.
    alpha_freq, alpha_rel : float
        Frequency and relative amplitude of the alpha-like oscillation.
    sensor_noise : float
        Relative amplitude of independent per-channel noise.
    first_samp : int
        ``first_samp`` of the returned RawArray.
    """

    def __init__(self, n_channels: int = 32, duration: float = 60.0, sfreq: float = 250.0,
                 seed: int = 0, rms: float = 15e-6, alpha_freq: float = 8.0,
                 alpha_rel: float = 0.6, sensor_noise: float = 0.15, n_sources: int = 30,
                 montage: str | None = None, ch_names: Sequence[str] | None = None,
                 first_samp: int = 0):
        if montage is None:
            if n_channels not in MONTAGES:
                raise ValueError(f"n_channels must be one of {sorted(MONTAGES)} or pass montage=")
            montage, subset = MONTAGES[n_channels]
            ch_names = subset if ch_names is None else ch_names
        mont = mne.channels.make_standard_montage(montage)
        if ch_names is None:
            ch_names = list(mont.ch_names)[:n_channels]
        self.ch_names = list(ch_names)
        self.montage = mont
        self.sfreq = float(sfreq)
        self.n_times = int(round(duration * sfreq))
        self.seed = int(seed)
        self.rms = float(rms)
        self.alpha_freq = alpha_freq
        self.alpha_rel = alpha_rel
        self.sensor_noise = sensor_noise
        self.n_sources = n_sources
        self.first_samp = int(first_samp)
        pos = mont.get_positions()["ch_pos"]
        self.pos = np.array([pos[c] for c in self.ch_names])  # (n_ch, 3) in m
        self._specs: list[tuple[str, dict]] = []
        self._events: list[ArtifactEvent] = []  # resolved positions, filled by add_*

    # ---------------------------------------------------------------- helpers
    @property
    def n_channels(self) -> int:
        return len(self.ch_names)

    def ch_index(self, chs: str | int | Sequence[str | int]) -> tuple[int, ...]:
        if isinstance(chs, (str, int, np.integer)):
            chs = [chs]
        return tuple(int(c) if isinstance(c, (int, np.integer)) else self.ch_names.index(c) for c in chs)

    def _resolve_window(self, onset, duration, after) -> tuple[int, int]:
        n = int(round(duration * self.sfreq)) if duration is not None else self.n_times
        n = max(1, min(n, self.n_times))
        if after is not None:
            start = self._events[after].stop
        elif onset is None or onset == "start":
            start = 0
        elif onset == "end":
            start = self.n_times - n
        elif onset == "middle":
            start = (self.n_times - n) // 2
        else:
            start = int(round(float(onset) * self.sfreq))
        stop = start + n
        if start < 0 or stop > self.n_times:
            raise ValueError(f"artifact window [{start}, {stop}) outside recording [0, {self.n_times})")
        return start, stop

    def _add(self, kind: str, channels, start: int, stop: int, **params) -> int:
        ev = ArtifactEvent(kind, tuple(sorted(channels)), start, stop, params)
        self._events.append(ev)
        return len(self._events) - 1

    def nearest(self, label: str) -> str:
        """Channel name for ``label``; 10-20 labels missing from the montage
        (e.g. 'Cz' on biosemi128) map to the nearest electrode."""
        if label in self.ch_names:
            return label
        ref = mne.channels.make_standard_montage("easycap-M1").get_positions()["ch_pos"]
        if label not in ref:
            raise ValueError(f"unknown channel / 10-20 label {label!r}")
        d = np.linalg.norm(self.pos - ref[label], axis=1)
        return self.ch_names[int(np.argmin(d))]

    def neighbours(self, center: str, radius: float = 0.05) -> list[str]:
        """Channel names within ``radius`` metres of ``center`` (inclusive)."""
        c = self.pos[self.ch_names.index(self.nearest(center))]
        d = np.linalg.norm(self.pos - c, axis=1)
        return [self.ch_names[i] for i in np.where(d <= radius)[0]]

    # ---------------------------------------------------------------- artifacts
    def add_flat(self, ch, value: float = 0.0, onset=None, duration=None, after=None) -> int:
        start, stop = self._resolve_window(onset, duration, after)
        return self._add("flat", self.ch_index(ch), start, stop, value=value)

    def add_noisy(self, ch, scale: float = 8.0, onset=None, duration=None, after=None) -> int:
        """Additive white noise with std ``scale * rms``."""
        start, stop = self._resolve_window(onset, duration, after)
        return self._add("noisy", self.ch_index(ch), start, stop, scale=scale)

    def add_lowcorr(self, ch, onset=None, duration=None, after=None) -> int:
        """Replace channel with an independent 1/f signal of the same RMS."""
        start, stop = self._resolve_window(onset, duration, after)
        return self._add("lowcorr", self.ch_index(ch), start, stop)

    def add_glitch(self, chs, onset=None, duration: float = 0.05, amplitude: float = 150e-6,
                   after=None) -> int:
        """Rectangular step/jump of ``amplitude`` volts lasting ``duration`` seconds."""
        if not 0.0 < duration <= 0.2:
            raise ValueError("glitch duration should be short (<= 200 ms)")
        start, stop = self._resolve_window(onset, duration, after)
        return self._add("glitch", self.ch_index(chs), start, stop, amplitude=amplitude)

    def add_motion(self, center: str, onset=None, duration: float = 1.5, amplitude: float = 400e-6,
                   radius: float = 0.06, after=None) -> int:
        """Slow, large, spatially extended transient around ``center``."""
        start, stop = self._resolve_window(onset, duration, after)
        center = self.nearest(center)
        chs = self.ch_index(self.neighbours(center, radius))
        return self._add("motion", chs, start, stop, amplitude=amplitude, center=center, radius=radius)

    def add_blink(self, onset=None, duration: float = 0.3, amplitude: float = 200e-6,
                  min_gain: float = 0.25, after=None) -> int:
        """Monophasic frontal deflection; channels with relative gain < ``min_gain`` untouched."""
        start, stop = self._resolve_window(onset, duration, after)
        gain = self._blink_gain()
        chs = tuple(np.where(gain >= min_gain)[0])
        return self._add("blink", chs, start, stop, amplitude=amplitude, min_gain=min_gain)

    def _blink_gain(self) -> np.ndarray:
        # Source just in front of the frontal pole; gain decays with distance.
        front = self.pos[:, 1].max()
        src = np.array([0.0, front + 0.03, self.pos[:, 2].mean()])
        d = np.linalg.norm(self.pos - src, axis=1)
        g = np.exp(-((d - d.min()) ** 2) / (2 * 0.04 ** 2))
        return g / g.max()

    # ---------------------------------------------------------------- render
    def _background(self, rng: np.random.Generator) -> np.ndarray:
        n_ch, n_t = self.n_channels, self.n_times
        # latent sources on a sphere of radius 0.07 m
        v = rng.standard_normal((self.n_sources, 3))
        v[:, 2] = np.abs(v[:, 2])  # upper hemisphere
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        src_pos = 0.07 * v + np.array([0.0, 0.0, self.pos[:, 2].mean() - 0.04])
        d = np.linalg.norm(self.pos[:, None, :] - src_pos[None, :, :], axis=2)
        gains = np.exp(-(d ** 2) / (2 * 0.045 ** 2))
        sources = pink_noise(rng, self.n_sources, n_t, self.sfreq)
        x = gains @ sources
        x /= x.std(axis=1, keepdims=True)
        # alpha-like posterior oscillation with slow amplitude modulation
        t = np.arange(n_t) / self.sfreq
        env = 1.0 + 0.5 * pink_noise(rng, 1, n_t, self.sfreq, beta=2.0, f_lo=0.05, f_hi=1.0)[0]
        env = np.clip(env, 0.1, None)
        alpha = env * np.sin(2 * np.pi * self.alpha_freq * t + rng.uniform(0, 2 * np.pi))
        post = np.clip(-self.pos[:, 1] / np.abs(self.pos[:, 1]).max(), 0, None) + 0.2
        x += self.alpha_rel * post[:, None] * alpha[None, :]
        x += self.sensor_noise * rng.standard_normal((n_ch, n_t))
        x -= x.mean(axis=1, keepdims=True)
        x /= x.std(axis=1, keepdims=True)
        return x * self.rms

    def build(self) -> tuple[mne.io.RawArray, GroundTruth]:
        """Render the recording. Returns ``(raw, truth)``."""
        ss = np.random.SeedSequence(self.seed)
        bg_seed, *ev_seeds = ss.spawn(1 + len(self._events))
        data = self._background(np.random.default_rng(bg_seed))
        blink_gain = self._blink_gain()
        for ev, s in zip(self._events, ev_seeds):
            rng = np.random.default_rng(s)
            sl = slice(ev.start, ev.stop)
            chs = list(ev.channels)
            n = ev.n_samples
            if ev.kind == "flat":
                data[chs, sl] = ev.params["value"]
            elif ev.kind == "noisy":
                data[chs, sl] += ev.params["scale"] * self.rms * rng.standard_normal((len(chs), n))
            elif ev.kind == "lowcorr":
                data[chs, sl] = self.rms * pink_noise(rng, len(chs), n, self.sfreq)
            elif ev.kind == "glitch":
                sign = rng.choice([-1.0, 1.0], size=(len(chs), 1))
                data[chs, sl] += sign * ev.params["amplitude"]
            elif ev.kind == "motion":
                t = np.arange(n) / self.sfreq
                freqs = rng.uniform(0.5, 3.0, size=3)
                phases = rng.uniform(0, 2 * np.pi, size=3)
                wave = sum(np.sin(2 * np.pi * f * t + p) for f, p in zip(freqs, phases))
                wave = wave / np.abs(wave).max() * _tukey(n, 0.3)
                c = self.pos[self.ch_names.index(ev.params["center"])]
                dist = np.linalg.norm(self.pos[chs] - c, axis=1)
                sigma = max(ev.params["radius"] / 2, 1e-3)
                g = 0.5 + 0.5 * np.exp(-(dist ** 2) / (2 * sigma ** 2))
                sign = rng.choice([-1.0, 1.0], size=(len(chs), 1))
                data[chs, sl] += ev.params["amplitude"] * sign * g[:, None] * wave[None, :]
            elif ev.kind == "blink":
                shape = np.hanning(n + 2)[1:-1]
                data[chs, sl] += ev.params["amplitude"] * blink_gain[chs, None] * shape[None, :]
            else:  # pragma: no cover
                raise ValueError(ev.kind)

        info = mne.create_info(self.ch_names, self.sfreq, ch_types="eeg")
        raw = mne.io.RawArray(data, info, first_samp=self.first_samp, verbose="ERROR")
        raw.set_montage(self.montage, verbose="ERROR")
        truth = GroundTruth(list(self.ch_names), self.sfreq, self.n_times, list(self._events))
        return raw, truth


# ----------------------------------------------------------------------------
# Epochs
# ----------------------------------------------------------------------------
def synthetic_epochs(n_channels: int = 32, n_epochs: int = 40, tmin: float = -0.2,
                     tmax: float = 0.8, sfreq: float = 250.0, seed: int = 0,
                     erp_amplitude: float = 10e-6, outliers: Sequence[int] = (),
                     outlier_scale: float = 5.0, spikes: Sequence[int] = (),
                     spike_amplitude: float = 300e-6, spike_duration: float = 0.02):
    """Epochs = background + a common ERP, with known bad epochs.

    ``outliers``: epochs with broadband noise of ``outlier_scale * rms`` added on
    every channel (large distance to the ERP and large GFP for most samples).
    ``spikes``: epochs with a single short, spatially varying spike (the
    epoch's *maximum* distance/GFP is extreme, its *mean* hardly changes).

    Returns ``(epochs, bad)`` with ``bad`` a boolean vector of length n_epochs.
    """
    n_t = int(round((tmax - tmin) * sfreq)) + 1
    sim = SyntheticEEG(n_channels, duration=n_epochs * n_t / sfreq + 1, sfreq=sfreq, seed=seed)
    raw, _ = sim.build()
    data = raw.get_data()[:, : n_epochs * n_t].reshape(n_channels, n_epochs, n_t).transpose(1, 0, 2).copy()
    t = tmin + np.arange(n_t) / sfreq
    topo = np.cos(np.linspace(0, np.pi, n_channels))  # fixed dipolar-ish topography
    erp = erp_amplitude * np.exp(-((t - 0.3) ** 2) / (2 * 0.05 ** 2))
    data += topo[None, :, None] * erp[None, None, :]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    for k in outliers:
        data[k] += outlier_scale * sim.rms * rng.standard_normal((n_channels, n_t))
    n_spk = max(1, int(round(spike_duration * sfreq)))
    for k in spikes:
        i0 = int(rng.integers(n_t // 4, 3 * n_t // 4))
        w = rng.standard_normal(n_channels)
        data[k, :, i0:i0 + n_spk] += spike_amplitude * w[:, None]
    info = raw.info.copy()
    events = np.c_[np.arange(n_epochs) * n_t + 1000, np.zeros(n_epochs, int), np.ones(n_epochs, int)]
    epochs = mne.EpochsArray(data, info, events=events, tmin=tmin, event_id={"stim": 1},
                             baseline=None, verbose="ERROR")
    bad = np.zeros(n_epochs, dtype=bool)
    bad[list(outliers) + list(spikes)] = True
    return epochs, bad
