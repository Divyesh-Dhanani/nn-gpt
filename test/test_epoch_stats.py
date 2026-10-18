"""Statistical correctness of the per-epoch report (ab/gpt/util/data/EpochStats.py).

Both cases here are regressions of bugs that shipped: a Wilson interval that
divided the wrong term by n, and a baseline that averaged the seed accuracy of
every generation attempt instead of only the ones the LLM modified successfully.
"""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    cs = importlib.util.spec_from_file_location(
        "ab.gpt.util.data.CycleResults", ROOT / "ab/gpt/util/data/CycleResults.py")
    cr = importlib.util.module_from_spec(cs)
    cs.loader.exec_module(cr)
    sys.modules.setdefault("ab.gpt.util.data.CycleResults", cr)
    es = importlib.util.spec_from_file_location(
        "_epoch_stats_under_test", ROOT / "ab/gpt/util/data/EpochStats.py")
    mod = importlib.util.module_from_spec(es)
    es.loader.exec_module(mod)
    return mod


ES = _load()


@pytest.mark.parametrize("k, n, lo, hi", [
    (0, 10, 0.0000, 0.2775),
    (2, 10, 0.0566, 0.5098),
    (4, 10, 0.1682, 0.6873),
    (10, 10, 0.7225, 1.0000),
])
def test_wilson_interval_matches_reference(k, n, lo, hi):
    _, got_lo, got_hi = ES.wilson_ci(k, n)
    assert got_lo == pytest.approx(lo, abs=1e-4)
    assert got_hi == pytest.approx(hi, abs=1e-4)


def test_wilson_interval_is_degenerate_only_for_no_trials():
    assert ES.wilson_ci(0, 0) == (0.0, 0.0, 0.0)
    # zero successes out of n must still bound the rate above zero
    assert ES.wilson_ci(0, 10)[2] > 0.0


def test_mean_ci_uses_t_not_z_for_small_samples():
    vals = [0.5895, 0.41, 0.33, 0.25]
    out = ES.mean_ci(vals)
    assert out["n"] == 4
    half_t = out["ci_high"] - out["mean"]
    half_z = 1.96 * out["std"] / 2.0          # sqrt(n) == 2
    assert half_t > half_z                     # t(3)=3.182 is wider than 1.96


def _make_epoch(tmp_path, valid, failed_seeds):
    d = tmp_path / "A0" / "synth_nn"
    d.mkdir(parents=True)
    pd = pytest.importorskip("pandas")
    for i, (seed, acc) in valid.items():
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": seed}).to_pickle(md / "dataframe.df")
        (md / "eval_info.json").write_text(json.dumps({"eval_results": {"accuracy": acc}}))
    for i, seed in failed_seeds.items():
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": seed}).to_pickle(md / "dataframe.df")
        (md / "error.txt").write_text("compile error")
    return d


def test_baseline_covers_only_successfully_modified_models(tmp_path):
    # Seeds of the failing models are deliberately far from the succeeding ones,
    # so averaging all of them gives a visibly different answer.
    valid = {0: (0.30, 0.5895), 1: (0.32, 0.41), 2: (0.28, 0.33), 3: (0.34, 0.25)}
    failed = {i: 0.90 for i in range(4, 10)}
    stats = ES.collect_epoch_stats(_make_epoch(tmp_path, valid, failed), 0)

    assert stats["models_generated"] == 10
    assert stats["models_valid"] == 4
    assert stats["baseline_accuracy_n"] == 4          # not 10
    assert stats["baseline_accuracy_mean"] == pytest.approx(0.31)
    assert stats["baseline_models"] == ["seed-0", "seed-1", "seed-2", "seed-3"]


def test_improvement_is_paired_per_model(tmp_path):
    valid = {0: (0.30, 0.5895), 1: (0.32, 0.41), 2: (0.28, 0.33), 3: (0.34, 0.25)}
    stats = ES.collect_epoch_stats(_make_epoch(tmp_path, valid, {}), 0)
    expected = sum(a - s for s, a in valid.values()) / len(valid)
    assert stats["paired_n"] == 4
    assert stats["mean_improvement"] == pytest.approx(expected)
    assert stats["improvement_ci_low"] < stats["mean_improvement"] < stats["improvement_ci_high"]


def test_epoch_with_no_valid_models_reports_nulls_not_zeros(tmp_path):
    stats = ES.collect_epoch_stats(_make_epoch(tmp_path, {}, {i: 0.5 for i in range(10)}), 2)
    assert stats["models_valid"] == 0
    assert stats["accuracy_mean"] is None
    assert stats["accuracy_max"] is None
    assert stats["mean_improvement"] is None
    assert stats["valid_rate"] == 0.0
    assert stats["valid_rate_ci_high"] > 0.0


def test_tracker_replaces_rather_than_duplicates_a_rerun_epoch(tmp_path):
    path = tmp_path / "epoch_tracker.json"
    ES.append_epoch_stats(path, {"epoch": 0, "accuracy_max": 0.1})
    ES.append_epoch_stats(path, {"epoch": 1, "accuracy_max": 0.2})
    rows = ES.append_epoch_stats(path, {"epoch": 0, "accuracy_max": 0.9})
    assert [r["epoch"] for r in rows] == [0, 1]
    assert rows[0]["accuracy_max"] == 0.9


# --- finalisation of the per-epoch results file ---------------------------

def _stats(epoch, valid=2, total=10, best=0.45):
    return {"epoch": epoch, "models_valid": valid, "models_generated": total,
            "accuracy_max": best, "accuracy_mean": best, "valid_rate": valid / total}


def test_finalize_adopts_a_generic_file_left_by_the_last_cycle(tmp_path):
    # The run that "kept the name cycle_results.json": NNEval wrote only the
    # generic file, so the epoch has no record of its own.
    generic = tmp_path / "cycle_results.json"
    generic.write_text(json.dumps(
        {"cycle": 3, "evaluation": {"best_accuracy": 0.45}, "cycle_time_minutes": 12.0}))

    ES.finalize_epoch_results(tmp_path, 3, _stats(3))

    per_epoch = tmp_path / "cycle_results_3.json"
    assert per_epoch.is_file()
    payload = json.loads(per_epoch.read_text())
    assert payload["cycle"] == 3
    assert payload["cycle_time_minutes"] == 12.0      # original content preserved
    assert payload["statistics"]["models_valid"] == 2  # statistics merged in


def test_finalize_does_not_adopt_a_generic_file_from_another_epoch(tmp_path):
    (tmp_path / "cycle_results.json").write_text(json.dumps(
        {"cycle": 2, "evaluation": {"best_accuracy": 0.99}, "cycle_time_minutes": 5.0}))

    ES.finalize_epoch_results(tmp_path, 3, _stats(3, best=0.45))

    payload = json.loads((tmp_path / "cycle_results_3.json").read_text())
    assert payload["cycle"] == 3
    assert payload["evaluation"]["best_accuracy"] == 0.45   # rebuilt, not epoch 2's 0.99
    assert payload.get("cycle_time_minutes") is None


def test_finalize_writes_a_record_when_evaluation_never_ran(tmp_path):
    ES.finalize_epoch_results(tmp_path, 4, _stats(4, valid=0, total=10, best=None))
    payload = json.loads((tmp_path / "cycle_results_4.json").read_text())
    assert payload["cycle"] == 4
    assert payload["success"] is False
    assert payload["generation"]["total_generated"] == 10
    assert payload["evaluation"]["best_accuracy"] is None
    # the generic name is refreshed too, so both agree
    assert json.loads((tmp_path / "cycle_results.json").read_text())["cycle"] == 4


def test_update_epoch_report_finalises_and_tracks_a_collapsed_epoch(tmp_path):
    pd = pytest.importorskip("pandas")
    d = tmp_path / "A2" / "synth_nn"
    d.mkdir(parents=True)
    for i in range(10):                      # all generations failed
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": 0.5}).to_pickle(md / "dataframe.df")
        (md / "error.txt").write_text("compile error")

    ES.update_epoch_report(d, 2, tmp_path)

    rows = json.loads((tmp_path / "epoch_tracker.json").read_text())
    assert [r["epoch"] for r in rows] == [2]
    assert rows[0]["models_valid"] == 0 and rows[0]["models_generated"] == 10
    assert (tmp_path / "cycle_results_2.json").is_file()


# --- clearing the previous run's artifacts --------------------------------

def _seed_run_dir(tmp_path, epochs=range(20)):
    (tmp_path / "cycle_results.json").write_text("{}")
    (tmp_path / "epoch_tracker.json").write_text("[]")
    (tmp_path / "epoch_statistics.png").write_bytes(b"\x89PNG")
    for e in epochs:
        (tmp_path / f"cycle_results_{e}.json").write_text("{}")
    # files that belong to the model, not to the statistics, must survive
    (tmp_path / "run_config.json").write_text("{}")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"w")
    keep = tmp_path / "checkpoint"
    keep.mkdir()
    (keep / "adapter_config.json").write_text("{}")


def test_reset_removes_every_statistics_file_of_a_previous_run(tmp_path):
    _seed_run_dir(tmp_path)
    removed = ES.reset_run_artifacts(tmp_path)

    assert len(removed) == 23            # 20 per-epoch + 3 named
    assert not list(tmp_path.glob("cycle_results*.json"))
    assert not (tmp_path / "epoch_tracker.json").exists()
    assert not (tmp_path / "epoch_statistics.png").exists()


def test_reset_leaves_model_and_config_files_alone(tmp_path):
    _seed_run_dir(tmp_path, epochs=range(3))
    ES.reset_run_artifacts(tmp_path)

    assert (tmp_path / "run_config.json").is_file()
    assert (tmp_path / "adapter_model.safetensors").is_file()
    assert (tmp_path / "checkpoint" / "adapter_config.json").is_file()


def test_reset_is_a_noop_on_a_clean_or_missing_directory(tmp_path):
    assert ES.reset_run_artifacts(tmp_path) == []
    assert ES.reset_run_artifacts(tmp_path / "does_not_exist") == []


def test_without_reset_a_short_run_inherits_a_longer_run_s_tail(tmp_path):
    # 20-epoch run, then a 4-epoch run writing over epochs 0..3 only.
    path = tmp_path / "epoch_tracker.json"
    for e in range(20):
        ES.append_epoch_stats(path, {"epoch": e, "accuracy_max": 0.1})
    for e in range(4):
        rows = ES.append_epoch_stats(path, {"epoch": e, "accuracy_max": 0.9})
    assert len(rows) == 20                      # epochs 4..19 are stale leftovers

    ES.reset_run_artifacts(tmp_path)
    for e in range(4):
        rows = ES.append_epoch_stats(path, {"epoch": e, "accuracy_max": 0.9})
    assert [r["epoch"] for r in rows] == [0, 1, 2, 3]


# --- the plot must carry the original networks on both accuracy panels ----

def _plot_rows():
    return [
        {"epoch": 0, "models_generated": 10, "models_valid": 4, "valid_rate": 0.4,
         "valid_rate_ci_low": 0.17, "valid_rate_ci_high": 0.69,
         "accuracy_mean": 0.46, "accuracy_ci_low": 0.28, "accuracy_ci_high": 0.65,
         "accuracy_max": 0.5895, "baseline_accuracy_mean": 0.54,
         "baseline_accuracy_ci_low": 0.44, "baseline_accuracy_ci_high": 0.63,
         "baseline_accuracy_max": 0.61},
        # n=2: the t-interval runs outside [0, 1] and must be clipped when drawn
        {"epoch": 1, "models_generated": 10, "models_valid": 2, "valid_rate": 0.2,
         "valid_rate_ci_low": 0.06, "valid_rate_ci_high": 0.51,
         "accuracy_mean": 0.44, "accuracy_ci_low": 0.29, "accuracy_ci_high": 0.59,
         "accuracy_max": 0.4501, "baseline_accuracy_mean": 0.535,
         "baseline_accuracy_ci_low": -0.04, "baseline_accuracy_ci_high": 1.11,
         "baseline_accuracy_max": 0.58},
        {"epoch": 2, "models_generated": 10, "models_valid": 0, "valid_rate": 0.0,
         "valid_rate_ci_low": 0.0, "valid_rate_ci_high": 0.28,
         "accuracy_mean": None, "accuracy_ci_low": None, "accuracy_ci_high": None,
         "accuracy_max": None, "baseline_accuracy_mean": None,
         "baseline_accuracy_ci_low": None, "baseline_accuracy_ci_high": None,
         "baseline_accuracy_max": None},
    ]


def _labels(ax):
    return ax.get_legend_handles_labels()[1]


def test_plot_series_cover_generated_and_original_on_both_panels(tmp_path, monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["labels"] = [_labels(ax) for ax in fig.axes]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")

    rate, accuracy = captured["labels"]
    assert any("Wilson" in l for l in rate)
    # accuracy panel: both means AND both intervals stay identifiable...
    assert any(l == "generated mean" for l in accuracy)
    assert any(l == "generated 95% CI" for l in accuracy)
    assert any("original" in l and "mean" in l for l in accuracy)
    assert any("original" in l and "95% CI" in l for l in accuracy)
    # ...and the maxima, folded in from the former third panel, are on it too
    assert any(l == "generated best" for l in accuracy)
    assert any(l == "original best" for l in accuracy)


def test_plot_draws_intervals_as_computed_and_frames_on_measured_values(tmp_path, monkeypatch):
    """Bounds are drawn as computed; the VIEW is framed on the measured values.

    A t-interval on n=2 can run past 0 and 1. Clamping it to [0, 1] turned every
    wide interval into a band filling the panel, so no series appeared to move
    between epochs. The band is therefore drawn where it was computed and simply
    leaves the frame, while the axis is framed on the means and best values that
    were actually observed -- which is what a reader is comparing.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            ys = []
            for coll in fig.axes[1].collections:          # the fill_between bands
                for path in coll.get_paths():
                    ys.extend(v[1] for v in path.vertices)
            captured["ys"] = ys
            captured["ylim"] = fig.axes[1].get_ylim()
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    rows = _plot_rows()
    ES.plot_epoch_stats(rows, tmp_path / "p.png")

    ys = captured["ys"]
    assert ys, "no shaded interval was drawn"
    assert min(ys) == pytest.approx(-0.04), "lower bound was altered, not drawn as computed"

    lo, hi = captured["ylim"]
    measured = [v for r in rows
                for v in (r.get("accuracy_mean"), r.get("baseline_accuracy_mean"),
                          r.get("accuracy_max"), r.get("baseline_accuracy_max"))
                if v is not None]
    assert lo > min(ys), "the view was framed on the interval, not the measurements"
    assert lo <= min(measured) and hi >= max(measured), "a measured value fell outside the view"
    # and the stored record keeps the exact, unclipped bounds
    assert rows[1]["baseline_accuracy_ci_low"] == -0.04


# --- a single observation supports no interval ---------------------------

def test_single_observation_yields_no_interval_not_a_zero_width_one():
    """One model cannot support a confidence interval.

    Returning the mean for both bounds draws a zero-width band, which reads as
    perfect certainty and makes two means look significantly different when
    nothing of the sort has been shown.
    """
    out = ES.mean_ci([0.1455])
    assert out["n"] == 1
    assert out["mean"] == pytest.approx(0.1455)
    assert out["ci_low"] is None and out["ci_high"] is None
    assert out["std"] is None
    # two observations are enough for a (wide) interval
    two = ES.mean_ci([0.45, 0.43])
    assert two["ci_low"] is not None and two["ci_high"] > two["ci_low"]


def test_plot_draws_the_full_range_when_the_interval_is_not_estimable(tmp_path, monkeypatch):
    """One model gives an unbounded interval; an accuracy lives in [0, 1].

    Drawing the whole feasible range states that plainly. The failure this
    guards against is the opposite: a band collapsed onto the mean, which
    would assert the mean is known exactly.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    rows = _plot_rows()
    rows[1].update(models_valid=1, accuracy_ci_low=None, accuracy_ci_high=None,
                   baseline_accuracy_ci_low=None, baseline_accuracy_ci_high=None)
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            spans = []
            for coll in fig.axes[1].collections:
                for path in coll.get_paths():
                    at_x1 = [v[1] for v in path.vertices if abs(v[0] - 1.0) < 1e-9]
                    if at_x1:
                        spans.append((min(at_x1), max(at_x1)))
            captured["spans"] = spans
            captured["labels"] = fig.axes[1].get_legend_handles_labels()[1]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / "p.png")

    assert captured["spans"], "no band was drawn at the single-model epoch"
    for low, high in captured["spans"]:
        assert low == pytest.approx(0.0)
        assert high == pytest.approx(1.0)
    # no extra legend entry is needed: the band speaks for itself
    assert not any("estimable" in l for l in captured["labels"])
    assert sum("95% CI" in l for l in captured["labels"]) == 2


# --- the confidence boundary must read as a band edge, not a second series ---


def test_ci_boundary_is_hairline_and_faint_next_to_the_mean_line(tmp_path, monkeypatch):
    """The CI edge used to be as heavy as the mean, so panel 2 showed six lines.

    It is context, not a measurement: thinner than the mean line and well below
    full opacity.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["lines"] = [(round(l.get_linewidth(), 3), l.get_alpha())
                                 for l in fig.axes[1].lines]
            # CI edges are the hairlines drawn at exactly CI_EDGE_WIDTH; the
            # accuracy panel also carries max/best-so-far lines that have an
            # alpha but are not interval boundaries.
            captured["edges"] = [(w, a) for w, a in captured["lines"]
                                 if w == round(ES.CI_EDGE_WIDTH, 3)]
            captured["means"] = [(w, a) for w, a in captured["lines"] if a is None]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")

    edges, means = captured["edges"], captured["means"]
    assert edges, "no CI boundary was drawn"
    assert means, "no mean line was drawn"
    mean_width = max(w for w, _ in means)
    for width, alpha in edges:
        assert width < mean_width, "boundary is as heavy as the mean line"
        assert width <= ES.CI_EDGE_WIDTH
        assert alpha <= ES.CI_EDGE_ALPHA, "boundary is too opaque to read as an edge"


def test_a_single_epoch_still_shows_its_interval(tmp_path, monkeypatch):
    """fill_between needs two x positions, so one cycle drew no interval at all.

    The first cycle of every run is exactly that case. One epoch is rendered as
    a capped whisker instead, with the band kept for two epochs or more.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["segments"] = [
                (l.get_xdata(), l.get_ydata()) for l in fig.axes[1].lines
                if l.get_alpha() is not None
            ]
            captured["labels"] = fig.axes[1].get_legend_handles_labels()[1]
            captured["wilson"] = [
                l for l in fig.axes[0].lines if l.get_alpha() is not None
            ]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows()[:1], tmp_path / "p.png")

    verticals = [(xs, ys) for xs, ys in captured["segments"]
                 if len(ys) == 2 and ys[0] != ys[1]]
    assert verticals, "a single epoch drew no visible interval"
    for _xs, ys in verticals:
        assert ys[1] > ys[0], "the whisker has no extent"
    # The legend is unchanged, so the panel reads the same as a longer run's.
    assert sum("95% CI" in l for l in captured["labels"]) == 2
    assert captured["wilson"], "the valid-rate panel lost its interval too"


def test_many_epochs_keep_the_shaded_band(tmp_path, monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["areas"] = [
                max(v[1] for v in p.vertices) - min(v[1] for v in p.vertices)
                for coll in fig.axes[1].collections for p in coll.get_paths()
            ]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")
    assert captured["areas"] and max(captured["areas"]) > 0


# --- seed models are reported together with the dataset they were trained on --


def test_seed_models_carry_their_dataset(tmp_path):
    pd = pytest.importorskip("pandas")
    d = tmp_path / "A0" / "synth_nn"
    d.mkdir(parents=True)
    for i, ds in enumerate(("cifar-10", "cifar-100")):
        md = d / f"B{i}"
        md.mkdir()
        pd.Series({"nn": f"seed-{i}", "accuracy": 0.3, "dataset": ds}).to_pickle(
            md / "dataframe.df")
        (md / "eval_info.json").write_text(
            json.dumps({"eval_results": {"accuracy": 0.4}}))

    stats = ES.collect_epoch_stats(d, 0)

    assert stats["baseline_models"] == ["seed-0", "seed-1"]
    assert stats["baseline_models_with_dataset"] == [
        "seed-0 (cifar-10)", "seed-1 (cifar-100)"]
    assert stats["datasets"] == ["cifar-10", "cifar-100"]
    assert {s["nn"]: s["dataset"] for s in stats["baseline_seeds"]} == {
        "seed-0": "cifar-10", "seed-1": "cifar-100"}


def test_a_seed_without_a_dataset_is_still_named(tmp_path):
    pd = pytest.importorskip("pandas")
    d = tmp_path / "A0" / "synth_nn"
    (d / "B0").mkdir(parents=True)
    pd.Series({"nn": "seed-0", "accuracy": 0.3}).to_pickle(d / "B0" / "dataframe.df")
    (d / "B0" / "eval_info.json").write_text(
        json.dumps({"eval_results": {"accuracy": 0.4}}))

    stats = ES.collect_epoch_stats(d, 0)
    assert stats["baseline_models_with_dataset"] == ["seed-0"]
    assert stats["datasets"] == []


# --- a new run starts from an empty directory -------------------------------


def test_reset_run_dir_removes_the_whole_previous_run(tmp_path):
    run = tmp_path / "nngpt"
    run.mkdir()
    _seed_run_dir(run, epochs=range(3))
    (run / "generation_seeds.json").write_text('{"nn": ["a"]}')
    (run / "llm" / "epoch" / "A0" / "synth_nn" / "B0").mkdir(parents=True)
    (run / "llm" / "epoch" / "A0" / "synth_nn" / "B0" / "new_nn.py").write_text("x = 1")

    assert ES.reset_run_dir(run) == run

    assert run.is_dir()                      # recreated, so the run can write
    assert list(run.iterdir()) == []         # and nothing of the old run is left


def test_reset_run_dir_is_a_noop_on_a_missing_directory(tmp_path):
    assert ES.reset_run_dir(tmp_path / "absent") is None


# --- the caption: what was run, stated once ------------------------------

_FULL_TRAINING = {
    "learning_rate": 1e-05, "lr_scheduler": "cosine", "warmup_steps": 5,
    "num_train_epochs": 1, "cycles": 4, "optimizer": "paged_adamw_8bit",
    "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.05,
    "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8,
    "max_prompts": 512, "sft_max_length": 4096, "context_length": 8192,
    "max_new_tokens": 8192, "test_nn": 16, "load_in_4bit": True,
    "unsloth": True, "vllm_sleep": True,
}


def test_param_block_states_every_hyper_parameter_on_one_line():
    """All of them, once -- a setting left out cannot be recovered by a reader."""
    lines = ES._param_block(_FULL_TRAINING)
    assert len(lines) == 1, f"configuration spread over {len(lines)} lines"
    text = lines[0]
    for key, value in _FULL_TRAINING.items():
        if key in ES.PARAM_SKIP or isinstance(value, bool):
            continue
        assert f"{ES.PARAM_LABEL.get(key, key)}={ES._fmt_value(value)}" in text, key
    assert len(text) <= ES.PARAM_MAX_CHARS


def test_param_block_omits_what_the_figure_already_shows():
    """cycles is the x axis and test_nn is every label's denominator."""
    text = " ".join(ES._param_block(_FULL_TRAINING))
    for key in ("cycles", "test_nn", "unsloth", "vllm_sleep"):
        assert ES.PARAM_LABEL.get(key, key) not in text, f"{key} is still shown"


def test_a_flag_is_named_when_set_and_absent_when_not():
    assert "4bit" in " ".join(ES._param_block(_FULL_TRAINING))
    off = dict(_FULL_TRAINING, load_in_4bit=False)
    assert "4bit" not in " ".join(ES._param_block(off))


def test_named_values_are_collected_after_the_numbers():
    """Numbers are compared against their neighbours; a name breaks the run."""
    cells = ES._param_block(_FULL_TRAINING)[0].split()
    named = [i for i, c in enumerate(cells)
             if not ES._is_numeric(c.split("=", 1)[-1])]
    assert named, "no named setting in the fixture"
    assert named == list(range(min(named), len(cells))), \
        f"a named setting sits among the numbers: {cells}"


def test_param_block_keeps_an_unrecognised_setting():
    """A setting added to run_config.json later still reaches the figure."""
    assert "rope_scaling=yarn" in " ".join(
        ES._param_block(dict(_FULL_TRAINING, rope_scaling="yarn")))


def test_param_block_wraps_rather_than_truncating():
    lines = ES._param_block(_FULL_TRAINING, max_chars=40)
    assert len(lines) == 2
    assert "opt=paged_adamw_8bit" in " ".join(lines)


def test_study_sentence_uses_the_names_the_dataset_records():
    """Verbatim, so the figure can be matched against the database."""
    assert ES._study_sentence({"task": "img-classification", "dataset": "cifar-10"}, []) == (
        "Automated generation of neural network architectures "
        "for img-classification on cifar-10")


def test_study_sentence_takes_the_dataset_from_the_rows_when_absent():
    sentence = ES._study_sentence({"task": "img-classification"},
                                  [{"datasets": ["cifar-10"]}, {"datasets": ["cifar-10"]}])
    assert sentence.endswith("for img-classification on cifar-10")


def test_study_sentence_survives_a_task_it_has_never_seen():
    sentence = ES._study_sentence({"task": "depth-estimation", "dataset": "nyu-v2"}, [])
    assert "depth-estimation" in sentence and "nyu-v2" in sentence


def test_study_sentence_without_a_configuration_still_reads_as_a_sentence():
    assert ES._study_sentence({}, []) == \
        "Automated generation of neural network architectures"


def test_legend_corners_are_square(tmp_path, monkeypatch):
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["rounded"] = [
                ax.get_legend().legendPatch.get_boxstyle().__class__.__name__
                for ax in fig.axes if ax.get_legend() is not None]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")
    assert captured["rounded"], "no legend was drawn"
    assert all(name == "Square" for name in captured["rounded"]), captured["rounded"]


def test_brand_sits_in_the_top_right_corner(tmp_path, monkeypatch):
    """Top-right, so a crop that removes it also removes the run's identity.

    The title and the configuration occupy the same band; the mark cannot be
    cropped away without taking part of them with it.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "texts"):
            captured["marks"] = [(t.get_position(), t.get_ha(), t.get_va())
                                 for t in fig.texts if t.get_text() == ES.BRAND_TEXT]
        return real_close(fig)

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(_plot_rows(), tmp_path / "p.png")

    marks = captured.get("marks")
    assert marks and len(marks) == 1, "the project mark was not drawn exactly once"
    (x, y), ha, va = marks[0]
    assert x > 0.9 and y > 0.9, f"mark at {(x, y)}, not the top-right corner"
    assert ha == "right" and va == "top"


@pytest.mark.parametrize("pair, suffix", [
    (("accuracy_mean", "baseline_accuracy_mean"), "mean"),
    (("accuracy_max", "baseline_accuracy_max"), "best"),
])
def test_both_series_stay_visible_where_their_values_coincide(
        pair, suffix, tmp_path, monkeypatch):
    """Equal values must not render as a single marker.

    Both series are accuracies on one axis, so they can land within a marker
    width of each other -- which is what a cycle that changed nothing produces,
    the case most worth seeing. At equal size the marker drawn second covered
    the first outright and one series vanished while its legend entry stayed.
    The generated value takes the larger marker and the original is stacked on
    top of it, so a coincidence reads as a small marker centred on a large one.
    """
    plt = pytest.importorskip("matplotlib.pyplot")
    generated_key, original_key = pair
    captured = {}
    real_close = plt.close

    def _capture(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["drawn"] = {
                l.get_label(): (l.get_markersize(), l.get_zorder(),
                                tuple(l.get_ydata()))
                for l in fig.axes[1].lines if l.get_label().endswith(suffix)}
        return real_close(fig)

    rows = _plot_rows()[:1]
    rows[0][original_key] = rows[0][generated_key]

    monkeypatch.setattr(plt, "close", _capture)
    ES.plot_epoch_stats(rows, tmp_path / "p.png")

    drawn = captured["drawn"]
    original = next(v for k, v in drawn.items() if k.startswith("original"))
    generated = next(v for k, v in drawn.items() if k.startswith("generated"))
    assert original[2] == generated[2], "the fixture no longer makes them coincide"
    assert generated[0] > original[0], "the generated marker is not the larger one"
    assert original[1] > generated[1], \
        "the smaller marker is stacked underneath, so the larger one hides it"


def test_the_legend_still_reads_original_before_generated():
    """Stacking is settled by zorder, so the listed order is free to stay."""
    assert ES.MEAN_MARKER[0][0] < ES.MEAN_MARKER[1][0]
    assert ES.BEST_MARKER[0][0] < ES.BEST_MARKER[1][0]