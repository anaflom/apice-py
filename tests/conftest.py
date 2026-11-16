"""Shared fixtures for the APICE-Py test-suite.

Baseline metrics
----------------
Tests that measure detection quality call the ``baseline`` fixture::

    baseline("glitch/32ch", sample_metrics(detected, truth))

All recorded numbers are printed in a table at the end of the pytest run
(terminal summary) and, if the environment variable ``APICE_BASELINE_JSON``
is set, written to that path as JSON.  They are the reference point for
future algorithm / threshold changes.
"""

from __future__ import annotations

import contextlib
import io
import json
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

_BASELINE: dict[str, dict] = {}


@pytest.fixture
def baseline():
    def _record(name: str, metrics: dict) -> dict:
        _BASELINE[name] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}
        return metrics
    return _record


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    if not _BASELINE:
        return
    tr = terminalreporter
    tr.section("APICE detection baseline (sample level)")
    cols = ["recall", "precision", "fpr", "tp", "fp", "fn"]
    tr.write_line(f"{'case':<44}" + "".join(f"{c:>11}" for c in cols))
    for name in sorted(_BASELINE):
        m = _BASELINE[name]
        row = "".join(f"{m.get(c, ''):>11}" for c in cols)
        tr.write_line(f"{name:<44}{row}")
    out = os.environ.get("APICE_BASELINE_JSON")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(_BASELINE, f, indent=2, sort_keys=True)
        tr.write_line(f"baseline written to {out}")


def quiet(func, *args, **kwargs):
    """Call ``func`` with stdout suppressed (APICE is very chatty)."""
    with contextlib.redirect_stdout(io.StringIO()):
        return func(*args, **kwargs)


@pytest.fixture
def run_quiet():
    return quiet


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture(autouse=True)
def _numpy_errstate():
    # The library divides by zero / takes logs of zeros in a few places on
    # purpose; keep the test output readable.
    with np.errstate(all="ignore"):
        yield
