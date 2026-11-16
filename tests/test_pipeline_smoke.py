"""End-to-end smoke test of ``apice.pipeline.run_preprocessing``.

``run_preprocessing`` catches and prints per-file exceptions, so success is
checked through its outputs, not through the absence of an exception.
"""

import mne
import numpy as np
import pytest

from apice.io import load_rawapice
from apice.pipeline import run_preprocessing
from synthetic import SyntheticEEG


@pytest.mark.slow
def test_run_preprocessing_smoke(tmp_path):
    s = SyntheticEEG(32, duration=40, seed=0)
    s.add_flat("T7")
    s.add_glitch(["Cz"], onset=10, duration=0.03)
    s.add_motion("O1", onset=20, duration=1.5)
    s.add_blink(onset=30)
    raw, truth = s.build()
    raw.set_annotations(mne.Annotations(onset=np.arange(2, 38, 2.0), duration=0.0,
                                        description="stim"))
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    in_dir.mkdir()
    raw.save(in_dir / "sub-01_raw.fif", verbose="ERROR")

    run_preprocessing(in_dir, out_dir, n_jobs=1, save_report=False, show_figures=False)

    out = out_dir / "sub-01_raw-preproc-raw.fif"
    logs = list((out_dir / "logs").glob("*")) if (out_dir / "logs").exists() else []
    assert out.exists(), f"no output; logs: {[p.read_text(errors='ignore')[-2000:] for p in logs]}"
    assert list((out_dir / "summary").glob("*")), "summary not written"

    pre = load_rawapice(out, verbose="ERROR")
    assert pre.ch_names == raw.ch_names
    assert pre.n_times == raw.n_times
    data = pre.get_data()
    assert np.isfinite(data).all()
    bct = np.asarray(pre.artifacts.BCT, bool)
    cct = np.asarray(pre.artifacts.CCT, bool)
    # the flat channel is flagged or interpolated, and something was corrected
    t7 = pre.ch_names.index("T7")
    assert bct[t7].mean() > 0.5 or cct[t7].mean() > 0.5
    assert bct.mean() < 0.5, "most of the data should survive"
