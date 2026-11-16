# APICE-Py: notes for Claude sessions

APICE-Py is the Python / MNE port of the APICE EEG preprocessing pipeline
(Fló et al. 2022), designed for infant data but usable with adults.
Active development branch: `feature/major-update` (not `main`).

## Read first

- `docs/architecture_notes.md`: requirements and options for the ongoing
  architecture revision, including the **provenance requirement for bad
  channels / bad times** (derived-from-BCT marks must be regenerable; manual
  and other-source marks must be kept and survive export/reload).
- `docs/pipeline.md`: description of the pipeline steps and classes.

## Environment and tests

- Python env: the conda env named `mne` (find it with `conda env list`).
  The package is imported from the repo; do not install into the env without asking.
- Run tests from the repo root: `python -m pytest -q` (all) or
  `python -m pytest -q -m "not slow"` (skip the end-to-end smoke test).
- From Git Bash, put the env's `Library\bin` (and the env root) on `PATH`
  first, otherwise numpy/BLAS calls crash with exit code 127. PowerShell with
  the same PATH prefix works too.
- `tests/synthetic.py` generates synthetic EEG with known artifact masks;
  use it for new detection / correction tests.
- Strict `xfail` tests document known, unfixed bugs. When a fix makes one
  pass, remove its marker.
- Personal scratch scripts go in `local_tests/` (git-ignored).

## Conventions

- Epoch-level bad mask `BE` has shape `(n_epochs,)`.
- Optional numeric thresholds use `None` for "default"; `0` is a real value
  (e.g. disables a criterion), so never test them with `if not x`.
- Commit only when asked; do not push without asking.
