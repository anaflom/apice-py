# Architecture notes (revision 2026-11)

Working notes for the planned revision of APICE-Py on `feature/major-update`.
These are requirements and options under discussion, not decisions already
implemented. Update this file when a decision is taken.

## 1. Requirement: provenance of bad channels (BC) and bad times (BT)

Every bad-channel and bad-time mark must keep track of **where it came from**:

- Marks **derived from BCT** (the channel x time artifact matrix) must be
  regenerable at any time, because they change when the derivation parameters
  change (`thresh_bad_channels`, `thresh_bad_times`, `min_bad_time`,
  `mask_time`, `min_good_time`, ...).
- Marks from **other sources** (manual rejection, and possibly others such as
  ICA or external annotations) must be preserved and re-added after every
  regeneration, and must survive export and reload.
- It must be possible to add further sources later.

Principle: store only the **primary** information (BCT plus one layer per
non-derived source), and treat BC and BT as **derived views**:

```
BC = f_BC(BCT, params) | BC_manual | BC_<other sources>
BT = f_BT(BCT, params) | BT_manual | BT_<other sources>
```

On export, the sources must stay distinguishable (for example separate
annotation descriptions for manual and derived bad times, and a separate list
of manual bad channels rather than merging everything into `info['bads']`).

### Current state (as of commit 3a4d49c)

- `BCmanual` exists and `define_bcbt` already computes
  `BC = BCmanual | derived(BCT)`, so manual bad channels survive regeneration
  within a session.
- Export (`RawAPICE.bc_to_bads`) merges BC and `BCmanual` into
  `info['bads']`; on reload `info['bads']` is loaded into BC (not
  `BCmanual`), and `define_bcbt` then overwrites BC from BCT. Manual bad
  channels are therefore **lost on reload**. This is tracked by the strict
  xfail test `tests/test_roundtrip.py::test_raw_roundtrip_bc_not_implied_by_bct`;
  remove the marker when the redesign fixes it.
- There is no `BTmanual`: bad times are always fully derived from BCT.
- Similar provenance already exists for bad epochs: `rejection_reasons` keeps
  a set of reasons per epoch.

## 2. Options for storing artifact masks alongside MNE objects

Problem: `RawAPICE` / `EpochsAPICE` subclass MNE classes and hold an
`Artifacts` object. Every MNE method that changes data shape (crop, pick,
resample, drop, ...) must be overridden to keep the masks in sync, which is
fragile across MNE versions.

Options considered:

1. **Annotations only.** Store all marks as MNE annotations (channel-specific
   annotations via `ch_names` cover BCT) and convert to matrices when needed.
   Fully native, but frequent conversions and many small annotations for
   sample-level BCT patterns.
2. **Native objects in / out of each step.** Each processing step accepts and
   returns native MNE objects and uses matrices internally.
3. **Composition with alignment by key** (current preference to prototype).
   Keep plain `mne.io.Raw` / `mne.Epochs` plus a separate `Artifacts` object
   that records its own channel names, `first_samp`, `sfreq` and epoch
   selection. Functions take and return `(inst, artifacts)`. After any MNE
   operation, `artifacts.align_to(inst)` re-syncs the masks by channel name
   and absolute sample time, so MNE methods need no overriding. Annotations
   (with run-length encoding of BCT) become the export / visualisation
   format; per-epoch information (BE, rejection reasons) maps onto
   `epochs.metadata` and `drop_log`.

Prototype option 3 on one pipeline step before committing to it, and design
the source layers from section 1 into it.

## 3. Other planned work (from the initial plan)

- Systematic evaluation of detection algorithms on semi-synthetic data with
  known artifact positions (`tests/synthetic.py`) and on public / infant data,
  scored by sample-level precision/recall and downstream ERP quality (SME).
- Max-change detector: mark only the samples between the window minimum and
  maximum (plus a margin) instead of the whole window.
- Thresholds: freeze each detector's reference distribution at its first
  estimate and reuse it after correction; robust statistics within sliding
  windows (sleep-state changes); mixture models giving per-sample artifact
  probabilities that can be combined across detectors.
- Reporting improvements; detection methods for low-density recordings.
