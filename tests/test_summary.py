"""Preprocessing summary: splicing decision counts."""

import numpy as np

from apice import RawAPICE
from apice.pipeline_utils import SummaryPreprocessing
from conftest import quiet
from synthetic import SyntheticEEG


def test_summary_reports_adaptive_splicing_counts(tmp_path):
    raw, _ = SyntheticEEG(32, duration=5, seed=0).build()
    r = quiet(RawAPICE, raw)
    s = quiet(SummaryPreprocessing, tmp_path, "rec", try_loading=False)
    quiet(s.add_to_summary, "before", r)
    r.artifacts.splice_stats = {"step": 3, "no_step": 10, "undecidable": 2, "long": 4}
    quiet(s.add_to_summary, "after", r)
    df = s.summary_df.set_index("step")
    assert np.isnan(df.loc["before", "n_splice_steps"])
    assert df.loc["after", ["n_splice_steps", "n_splice_no_step", "n_splice_undecidable",
                            "n_splice_long"]].tolist() == [3, 10, 2, 4]


def test_summary_loaded_without_new_columns(tmp_path):
    """Summaries written before these columns existed can still be extended."""
    raw, _ = SyntheticEEG(32, duration=5, seed=0).build()
    r = quiet(RawAPICE, raw)
    s = quiet(SummaryPreprocessing, tmp_path, "rec", try_loading=False)
    s.summary_df = s.summary_df.drop(columns=["n_splice_steps", "n_splice_no_step", "n_splice_undecidable",
                                              "n_splice_long"])
    quiet(s.add_to_summary, "x", r)
    assert "n_splice_steps" in s.summary_df.columns and len(s.summary_df) == 1
