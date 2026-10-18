"""Per-epoch statistics for the NNGPT generate -> evaluate -> fine-tune loop.

For one epoch this collects, straight from the artifacts NNEval leaves in the
epoch's ``synth_nn`` directory:

* how many models were generated and how many are *valid* (ran to completion and
  produced an accuracy), plus the valid rate with a Wilson 95% interval;
* accuracy of the generated models -- mean with a 95% interval, and the maximum;
* accuracy of the seed models those were derived from, read back from the
  ``dataframe.df`` pickle each generated model carries. That is the "before
  modification" number from the LEMUR database, with its own mean and interval,
  so an epoch can be compared against what it started from.

One row per epoch is appended to ``epoch_tracker.json`` and a plot is refreshed.

The previous implementation lived at the end of ``nn_gen()`` and read whatever
``cycle_results.json`` happened to contain at that moment. Generation runs
*before* evaluation, so the file still held the previous epoch's numbers and
every accuracy was reported one epoch late. Collecting here, after evaluation,
from this epoch's own directory removes the lag entirely.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path

Z_95 = 1.96

# t critical values at 95% two-sided, indexed by degrees of freedom (n-1).
# Only used when scipy is unavailable; the normal 1.96 is far too narrow for the
# handful of valid models a single epoch usually yields.
_T_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
         8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
         14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t_critical(df, conf=0.95):
    """Two-sided t critical value for ``df`` degrees of freedom."""
    if df <= 0:
        return float('nan')
    try:
        from scipy import stats
        return float(stats.t.ppf(1 - (1 - conf) / 2, df))
    except Exception:
        pass
    if df in _T_95:
        return _T_95[df]
    for key in sorted(_T_95):
        if df <= key:
            return _T_95[key]
    return Z_95


def wilson_ci(k, n, z=Z_95):
    """Wilson score interval for a binomial proportion.

    Returns ``(p, lower, upper)``. Preferred over the normal approximation
    because an epoch evaluates ~10 models and k is often 0 or n, where the
    normal interval collapses to zero width or runs outside [0, 1].
    """
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    # Only the p(1-p) term is divided by n; z^2/(4n^2) is not. The copy in
    # plot_wilson_ci.py divided the whole bracket by n, which put that term over
    # n^3 and produced intervals that were both too narrow and misplaced -- at
    # k=0,n=10 it reported [0.095, 0.183], claiming at least 9.5% of generations
    # were valid when none were. That function now imports this one.
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def mean_ci(values, conf=0.95):
    """Mean of ``values`` with a two-sided t interval.

    Returns a dict with n/mean/ci_low/ci_high/std/min/max; every numeric field
    is None when there is nothing to report, so the JSON stays well formed.
    """
    vals = [float(v) for v in values if v is not None]
    n = len(vals)
    out = {"n": n, "mean": None, "ci_low": None, "ci_high": None,
           "std": None, "min": None, "max": None}
    if n == 0:
        return out
    mean = sum(vals) / n
    out.update(mean=mean, min=min(vals), max=max(vals))
    if n == 1:
        # No interval is estimable from a single observation: the sample
        # standard deviation is undefined and the t distribution has zero
        # degrees of freedom, so the interval is unbounded. Returning the mean
        # for both bounds -- as this did -- draws a zero-width band, which
        # reads as perfect certainty and makes two means look significantly
        # different when nothing of the sort has been shown. Left as None; the
        # plot marks these epochs as having no estimable interval.
        return out
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    std = math.sqrt(var)
    half = t_critical(n - 1, conf) * std / math.sqrt(n)
    out.update(std=std, ci_low=mean - half, ci_high=mean + half)
    return out


def _seed_accuracy(model_dir):
    """Pre-modification accuracy, name and dataset of the seed, from dataframe.df.

    The dataset travels with the name because an accuracy is meaningless without
    it: 0.66 on cifar-10 and 0.66 on imagenet are not the same result, and a
    seed list naming only architectures cannot be checked or reproduced. The
    same row also carries the task and metric the number was measured under.
    """
    df_path = Path(model_dir) / "dataframe.df"
    if not df_path.is_file():
        return None, None, None
    try:
        import pandas as pd
        row = pd.read_pickle(df_path)
        acc = row.get("accuracy")
        name = row.get("nn")
        dataset = row.get("dataset")
        return (float(acc) if acc is not None else None,
                str(name) if name is not None else None,
                str(dataset) if dataset is not None else None)
    except Exception:
        return None, None, None


def _label(name, dataset):
    """How a seed model is named in reports: ``architecture (dataset)``."""
    if not name:
        return None
    return f"{name} ({dataset})" if dataset else str(name)


def collect_epoch_stats(models_dir, epoch):
    """Build the statistics record for one epoch from its model directories.

    The baseline is PAIRED: only the seed models whose LLM-modified descendant
    actually ran and produced an accuracy are included. Averaging the seeds of
    every attempt, including the ones that failed to compile, would compare the
    generated mean (computed over valid models only) against a different and
    larger population, and the difference would partly reflect which seeds
    happened to fail rather than what the LLM did to them.
    """
    from ab.gpt.util.data.CycleResults import collect_cycle_metrics

    models_dir = Path(models_dir)
    eval_results, model_dirs, successful, failed = collect_cycle_metrics(models_dir, models_dir)

    generated = len(model_dirs)
    valid = len(successful)

    # A directory with neither eval_info.json nor error.txt counted as neither
    # valid nor failed, so the three numbers did not add up and a silently
    # empty generation looked like nothing had gone wrong -- epoch 1 of one run
    # reported 10 generated, 2 valid, 4 failed. Anything not valid has failed.
    failed = max(len(failed), generated - valid)

    accuracies, seed_accs, seed_names, deltas = [], [], [], []
    seed_pairs, datasets = [], []
    for r in eval_results:
        acc = r.get("accuracy")
        if acc is None:
            continue
        accuracies.append(acc)
        seed_acc, seed_name, seed_dataset = _seed_accuracy(r["model_dir"])
        if seed_acc is None:
            continue                      # no seed row on disk: cannot pair this one
        seed_accs.append(seed_acc)
        deltas.append(acc - seed_acc)
        if seed_name:
            seed_names.append(seed_name)
            seed_pairs.append({"nn": seed_name, "dataset": seed_dataset,
                               "baseline_accuracy": seed_acc, "accuracy": acc})
        if seed_dataset:
            datasets.append(seed_dataset)

    p, lo, hi = wilson_ci(valid, generated)
    gen = mean_ci(accuracies)
    base = mean_ci(seed_accs)
    # Per-model differences, so the interval accounts for the pairing instead of
    # treating the two means as independent samples.
    delta = mean_ci(deltas)

    return {
        "epoch": epoch,
        "timestamp": datetime.now().isoformat(),
        "models_generated": generated,
        "models_valid": valid,
        "models_failed": failed,
        # Wilson interval over the `models_generated` Bernoulli trials of this
        # epoch -- the rate is one number, but it rests on n attempts.
        "valid_rate": p,
        "valid_rate_ci_low": lo,
        "valid_rate_ci_high": hi,
        "valid_rate_percent": 100.0 * p,
        # accuracy of this epoch's valid generated models
        "accuracy": gen["max"],          # kept: the previous schema's "accuracy" was the best one
        "accuracy_max": gen["max"],
        "accuracy_mean": gen["mean"],
        "accuracy_ci_low": gen["ci_low"],
        "accuracy_ci_high": gen["ci_high"],
        "accuracy_std": gen["std"],
        "accuracy_n": gen["n"],
        # accuracy the SAME models had before the LLM modified them (LEMUR),
        # restricted to the valid ones so the two means are comparable
        "baseline_accuracy_mean": base["mean"],
        "baseline_accuracy_ci_low": base["ci_low"],
        "baseline_accuracy_ci_high": base["ci_high"],
        "baseline_accuracy_max": base["max"],
        "baseline_accuracy_n": base["n"],
        # Architecture names only, as before, so anything already reading this
        # field keeps working...
        "baseline_models": sorted(set(seed_names)),
        # ...with the dataset each was trained on alongside it. An accuracy
        # without its dataset cannot be interpreted or reproduced.
        "baseline_models_with_dataset": sorted(
            {_label(p["nn"], p["dataset"]) for p in seed_pairs}),
        "baseline_seeds": sorted(seed_pairs, key=lambda p: (p["nn"], p["dataset"] or "")),
        "datasets": sorted(set(datasets)),
        # paired generated-minus-seed difference
        "paired_n": delta["n"],
        "mean_improvement": delta["mean"],
        "improvement_ci_low": delta["ci_low"],
        "improvement_ci_high": delta["ci_high"],
    }


def append_epoch_stats(tracker_path, stats):
    """Append (or replace) this epoch's row in the tracker, keeping it sorted."""
    tracker_path = Path(tracker_path)
    rows = []
    if tracker_path.exists():
        try:
            with open(tracker_path) as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                rows = loaded
        except Exception:
            rows = []
    # A re-run of the same epoch replaces its row rather than duplicating it.
    rows = [r for r in rows if r.get("epoch") != stats["epoch"]]
    rows.append(stats)
    rows.sort(key=lambda r: (r.get("epoch") is None, r.get("epoch")))
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tracker_path, "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def _run_config(base_dir):
    """The run's model and fine-tuning settings, from run_config.json."""
    try:
        with open(Path(base_dir) / "run_config.json") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


#: Display order for the fine-tuning settings. Anything in run_config.json that
#: is not listed is appended, so a newly recorded setting still appears.
PARAM_ORDER = (
    "learning_rate", "lr_scheduler", "warmup_steps", "num_train_epochs", "cycles",
    "optimizer", "lora_r", "lora_alpha", "lora_dropout",
    "per_device_train_batch_size", "gradient_accumulation_steps",
    "max_prompts", "sft_max_length", "context_length", "max_new_tokens",
    "test_nn", "load_in_4bit", "unsloth", "vllm_sleep",
)

#: Short labels so the whole configuration fits on one line; the settings
#: themselves are unabridged.
PARAM_LABEL = {
    "learning_rate": "lr",
    "lr_scheduler": "sched",
    "warmup_steps": "warmup",
    "warmup_ratio": "warmup",
    "num_train_epochs": "ep/cyc",
    "optimizer": "opt",
    "lora_r": "r",
    "lora_alpha": "alpha",
    "lora_dropout": "dropout",
    "per_device_train_batch_size": "batch",
    "gradient_accumulation_steps": "accum",
    "max_prompts": "prompts",
    "context_length": "ctx",
    "max_new_tokens": "new_tok",
    "sft_max_length": "sft_len",
    "load_in_4bit": "4bit",
}

#: Recorded settings that are not hyper-parameters of the fine-tuning, or that
#: the figure already states. ``cycles`` is the length of the x axis and
#: ``test_nn`` is the denominator of every label on the upper panel, so naming
#: them again in the caption only costs line width. The runtime switches say
#: nothing about the optimisation that produced these curves.
PARAM_SKIP = ("cycles", "test_nn", "unsloth", "vllm_sleep")

#: Settings whose value is a plain on/off. Shown by name when enabled and left
#: out entirely when not: "4bit=no" is the default state and reads as a
#: parameter that was chosen, which it was not.
PARAM_FLAGS = ("load_in_4bit",)

PARAM_MAX_CHARS = 170

#: Clearance between the caption and the first panel's title. Enough to
#: separate them, not enough to read as a break in the figure.
HEADER_GAP = 0.006


def _fmt_value(value):
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _is_numeric(value):
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    try:
        float(str(value))
        return True
    except (TypeError, ValueError):
        return False


def _param_block(training, max_chars=PARAM_MAX_CHARS):
    """The fine-tuning hyper-parameters, on one line.

    Settings with a numeric value come first, in the order of PARAM_ORDER, and
    the named ones -- the schedule, the optimiser, the quantisation -- are
    collected at the end. A number is read by comparing it with its neighbours,
    which only works while the numbers sit together; a name interrupting the
    run forces the eye to re-anchor on every value after it.
    """
    if not training:
        return []
    keys = [k for k in PARAM_ORDER if k in training and k not in PARAM_SKIP]
    keys += [k for k in training if k not in PARAM_ORDER and k not in PARAM_SKIP]

    numeric, named = [], []
    for k in keys:
        value = training[k]
        label = PARAM_LABEL.get(k, k)
        if k in PARAM_FLAGS or isinstance(value, bool):
            if value:
                named.append(label)
            continue
        cell = f"{label}={_fmt_value(value)}"
        (numeric if _is_numeric(value) else named).append(cell)

    cells = numeric + named
    if not cells:
        return []
    line = " ".join(cells)
    if len(line) <= max_chars:
        return [line]
    half = -(-len(cells) // 2)
    return [" ".join(cells[:half]), " ".join(cells[half:])]


def _dataset_phrase(names):
    """The dataset identifiers exactly as the dataset records them.

    The caption has to be matched against the database by whoever reads it, so
    the keys are carried through unchanged rather than prettified: a figure
    naming "CIFAR-10" cannot be looked up, one naming "cifar-10" can.
    """
    seen = []
    for n in names:
        n = str(n).strip()
        if n and n not in seen:
            seen.append(n)
    if not seen:
        return ""
    if len(seen) == 1:
        return seen[0]
    return ", ".join(seen[:-1]) + " and " + seen[-1]


def _study_sentence(config, rows):
    """One-sentence statement of what the figure reports.

    A reader meeting the figure in a paper or a report needs the object of study
    before the axes mean anything: which problem the architectures are generated
    for, and on which data they are measured.
    """
    task = str(config.get("task") or "").strip()
    datasets = config.get("dataset")
    datasets = [datasets] if isinstance(datasets, str) else list(datasets or [])
    if not datasets:
        for r in rows:
            datasets.extend(r.get("datasets") or [])
    data = _dataset_phrase(datasets)

    subject = "Automated generation of neural network architectures"
    if task and data:
        return f"{subject} for {task} on {data}"
    if task:
        return f"{subject} for {task}"
    if data:
        return f"{subject} evaluated on {data}"
    return subject


def _run_caption(config, rows):
    """Model name, the statement of the study, and the fine-tuning configuration.

    A plot of accuracies is not interpretable without the configuration: the
    same pipeline at a different learning rate, rank or prompt count produces a
    different curve, and nothing else on the figure records which one this is.
    Every setting recorded in run_config.json is shown rather than a chosen few,
    so two runs can be told apart by looking at them.
    """
    model = config.get("base_model_name") or config.get("llm_conf") or ""
    model = str(model).rstrip("/").split("/")[-1].replace(".json", "")
    return (model, _study_sentence(config, rows),
            _param_block(config.get("training") or {}))


#: Project mark set in the top-right corner, on the band occupied by the title
#: and the configuration. Letterspaced and in a light grey so it reads as a
#: quiet imprint rather than a watermark over the data, and placed where it
#: cannot be cropped away without also removing the run's identification.
BRAND_TEXT = "N N G P T"
BRAND_COLOR = "#a9b0b8"
BRAND_ACCENT = "#2ca02c"


def _draw_brand(fig, text=BRAND_TEXT):
    """Quiet project imprint, top-right, clear of the axes."""
    if not text:
        return
    from matplotlib.patches import Rectangle
    right, top = 0.9885, 0.9955
    fig.text(right, top, text, ha="right", va="top",
             fontsize=7.5, fontweight="bold", color=BRAND_COLOR,
             family="sans-serif", zorder=0)
    rule = 0.047
    fig.patches.append(Rectangle(
        (right - rule, top - 0.0135), rule, 0.0016,
        transform=fig.transFigure, facecolor=BRAND_ACCENT,
        edgecolor="none", alpha=0.55, zorder=0))


def _value_limits(values, lo_bound=0.0, hi_bound=1.0, headroom=0.18):
    """View limits framed on the MEASURED VALUES, with room for point labels.

    Framing on confidence bounds instead compresses every line into the middle
    of the panel: a 95% interval over a handful of models can be several times
    wider than any value actually observed. The intervals are still drawn as
    computed -- they simply leave the frame, and where each one exits stays
    visible.

    ``headroom`` is extra padding at the top only, so a label sitting above the
    highest point is never clipped by the spine.
    """
    values = [v for v in values if v is not None]
    if not values:
        return lo_bound, hi_bound
    lo, hi = min(values), max(values)
    span = hi - lo
    pad = max(0.02 * (hi_bound - lo_bound), 0.08 * span)
    return (max(lo_bound, lo - pad),
            min(hi_bound, hi + pad + headroom * max(span, pad)))


# Confidence bands are context, not a second measurement: the boundary is drawn
# hairline and faint so it reads as the edge of a band rather than as another
# series. At the previous 1.1pt / 0.95 alpha the two bounds were as heavy as the
# mean line itself and the panel showed six lines where it should show two plus
# two shaded regions.
CI_EDGE_WIDTH = 0.5
CI_EDGE_ALPHA = 0.35
#: Fill weight of every confidence band, on every panel. Both accuracy
#: intervals get the SAME alpha so neither is washed out by the other: a light
#: grey at high alpha under a green band at low alpha renders as green, which is
#: why the original interval used to vanish wherever the two overlapped -- and
#: they overlap almost everywhere. The valid-rate band uses it too, so an
#: interval reads the same way on both panels.
CI_ALPHA = 0.22

#: Marker size and stacking order of the two accuracy series, in the order the
#: series are listed (original first, then generated), for the mean and for the
#: best model.
#:
#: The generated networks are the result the figure is about, so they carry the
#: larger marker; the networks they were derived from carry the smaller one and
#: are stacked ON TOP of it. Both values are accuracies on one axis and can land
#: within a marker width of each other -- which is precisely what a cycle that
#: changed nothing produces, the case most worth seeing -- and at equal size the
#: marker drawn second covered the first outright, so one series simply vanished
#: while its legend entry stayed. Sized and stacked this way, a coincidence
#: reads as a small grey marker centred on a larger coloured one.
MEAN_MARKER = ((5.5, 6), (9.0, 5))
BEST_MARKER = ((5.0, 7), (8.0, 6))
MARKER_EDGE = dict(markeredgecolor="white", markeredgewidth=0.7)


def _draw_band(ax, xs, lo, hi, line_colour, fill_colour, alpha, label):
    """Shade a confidence interval and outline it with a hairline boundary.

    ``fill_between`` needs two x positions to have any area, so a run with a
    single epoch -- the first cycle of every run, and any run of one cycle --
    drew nothing at all and the interval was invisible exactly when it matters
    most. One point is therefore rendered as a capped vertical whisker instead:
    the same colour, the same faint weight, the same meaning, and it appears
    from the first epoch. From two epochs on the shaded band is drawn as before,
    so nothing changes about how a normal run looks.
    """
    from matplotlib.colors import to_rgba

    if len(xs) == 1:
        x = xs[0]
        ax.plot([x, x], [lo[0], hi[0]], color=line_colour,
                linewidth=1.0, alpha=0.55, solid_capstyle="butt", zorder=3)
        cap = 0.06
        for bound in (lo[0], hi[0]):
            ax.plot([x - cap, x + cap], [bound, bound], color=line_colour,
                    linewidth=CI_EDGE_WIDTH + 0.3, alpha=0.55, zorder=3)
        # A proxy artist keeps the legend identical to the multi-epoch case.
        ax.fill_between([x, x], lo, hi, facecolor=to_rgba(fill_colour, alpha),
                        edgecolor="none", label=label, zorder=2)
        return

    # Put the alpha in the face colour, not in the call: a call-level alpha is
    # applied to the edge as well and the boundary disappears exactly where the
    # two bands cross.
    ax.fill_between(xs, lo, hi, facecolor=to_rgba(fill_colour, alpha),
                    edgecolor="none", label=label, zorder=2)
    for bound in (lo, hi):
        ax.plot(xs, bound, color=line_colour, linewidth=CI_EDGE_WIDTH,
                linestyle="-", alpha=CI_EDGE_ALPHA, zorder=3)


def plot_epoch_stats(rows, out_path, base_dir=None):
    """Two stacked panels: valid generation rate, and accuracy (mean CI + max).

    The former third panel (best model per epoch) is folded into the accuracy
    panel: max is the same quantity on the same scale as the mean, so splitting
    them forced a reader to compare across axes with different y ranges. Colour
    stays with the entity -- green generated, grey original -- and the statistic
    is carried by marker and line style, so the two never rely on colour alone.
    """
    import matplotlib
    matplotlib.use("Agg")          # cluster nodes have no display
    import matplotlib.pyplot as plt

    rows = [r for r in rows if r.get("epoch") is not None]
    if not rows:
        return None
    ep = [r["epoch"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)

    # Which model and which fine-tuning settings produced these numbers, and on
    # which dataset. Without them the curves cannot be compared between runs.
    head, tail, params = _run_caption(_run_config(base_dir) if base_dir else {}, rows)
    y = 0.995
    if head:
        # Drawn as figure text rather than through suptitle: tight_layout
        # reserves a band of its own for a suptitle, which reappeared as a gap
        # between the caption and the first panel that no spacing value here
        # could close.
        fig.text(0.5, y, head, ha="center", va="top", fontsize=11,
                 fontweight="bold")
        y -= 0.026
    if tail:
        fig.text(0.5, y, tail, ha="center", va="top", fontsize=8.8,
                 color="#44474b", style="italic")
        y -= 0.0275
    if params:
        fig.text(0.5, y, "\n".join(params), ha="center", va="top",
                 fontsize=7.0, family="monospace", color="#333333", linespacing=1.5)
        y -= 0.013 * len(params)
    header_bottom = y - HEADER_GAP

    # 1 -- percentage of correctly generated models, Wilson 95%
    pct = [100.0 * (r.get("valid_rate") or 0.0) for r in rows]
    lo = [100.0 * (r.get("valid_rate_ci_low") or 0.0) for r in rows]
    hi = [100.0 * (r.get("valid_rate_ci_high") or 0.0) for r in rows]
    axes[0].plot(ep, pct, marker="o", color="#1f77b4", label="valid models", zorder=5)
    # Same fill weight as the accuracy bands below, so an interval reads the
    # same way on both panels.
    _draw_band(axes[0], ep, lo, hi, "#1f77b4", "#1f77b4", CI_ALPHA, "Wilson 95% CI")
    for x, r in zip(ep, rows):
        axes[0].annotate(f"{r.get('models_valid', 0)}/{r.get('models_generated', 0)}",
                         (x, 100.0 * (r.get("valid_rate") or 0.0)),
                         textcoords="offset points", xytext=(0, 7),
                         ha="center", fontsize=8)
    axes[0].set_ylabel("correctly generated (%)")
    # Framed on the observed rates, not the Wilson band, for the same reason as
    # the accuracy panel below: at 4/16 the interval runs 10-49% and swamps the
    # movement in the rate itself. Headroom leaves room for the n/N labels.
    axes[0].set_ylim(*_value_limits(pct, 0.0, 100.0))
    axes[0].set_title("Valid generation rate per epoch")
    axes[0].legend(loc="best", fontsize=8, framealpha=0.75, fancybox=False)
    axes[0].grid(alpha=0.3)

    # Grey means "the original networks, before the LLM changed them" on every
    # panel; green means the networks the LLM produced. Both are measurements
    # of the same kind, so both are drawn solid and filled -- only the colour
    # separates them.
    GENERATED, ORIGINAL = "#2ca02c", "#7f7f7f"
    # The best generated network is the headline number, so it gets its own
    # colour rather than sharing green with the mean. A mid red sits ~ΔE 4 from
    # this green under deuteranopia -- the classic red/green confusion -- so the
    # red is taken dark enough to clear the threshold: #9d0208 separates at
    # ΔE 10.5, leaving grey-vs-green as the worst pair on the panel.
    BEST = "#9d0208"

    def _pick(*keys):
        """Rows where the first key is present, as parallel tuples."""
        pts = [tuple([r["epoch"]] + [r.get(k) for k in keys])
               for r in rows if r.get(keys[0]) is not None]
        return list(zip(*pts)) if pts else None

    # 2 -- mean accuracy with 95% CI. Both series are measurements of the same
    # kind -- the networks before and after the LLM changed them -- so both get
    # a solid line and a filled interval, distinguished by colour alone.
    # Equal alpha plus an opaque boundary keeps both readable through overlap.
    ORIGINAL_FILL = "#4d4d4d"   # darker than the mean line, so it reads as grey
    series_spec = (
        ("original (before modification)", ORIGINAL, ORIGINAL_FILL, CI_ALPHA,
         ("baseline_accuracy_mean", "baseline_accuracy_ci_low", "baseline_accuracy_ci_high")),
        ("generated", GENERATED, GENERATED, CI_ALPHA,
         ("accuracy_mean", "accuracy_ci_low", "accuracy_ci_high")),
    )

    drawn = {}
    for label, line_colour, fill_colour, alpha, keys in series_spec:
        picked = _pick(*keys)
        if picked is None:
            continue
        xs, ms, ls, hs = picked
        # Bounds are plotted AS COMPUTED. Clamping them to [0, 1] -- as this
        # did -- turned every wide interval into a band filling the panel with
        # a flat line along each edge, indistinguishable from an interval that
        # genuinely spans 0 to 1, so neither series appeared to change from
        # epoch to epoch. The view is cropped instead (see set_ylim below): a
        # band running past an edge leaves the frame, and where each boundary
        # exits stays visible.
        # An epoch with a single model is the one exception: no degrees of
        # freedom means no interval is estimable, so it spans the whole
        # feasible range, saying the epoch pins the mean down no further.
        lo = [0.0 if l is None else l for l in ls]
        hi = [1.0 if h is None else h for h in hs]
        _draw_band(axes[1], xs, lo, hi, line_colour, fill_colour, alpha,
                   f"{label} 95% CI")
        drawn[label] = (xs, ms, line_colour)

    # Mean lines last so neither band hides them. Size and stacking come from
    # MEAN_MARKER: the series are plotted in their listed order so the legend
    # keeps reading original-then-generated, and which one ends up on top is
    # settled by zorder rather than by call order.
    for i, (label, *_) in enumerate(series_spec):
        if label not in drawn:
            continue
        xs, ms, colour = drawn[label]
        size, z = MEAN_MARKER[min(i, len(MEAN_MARKER) - 1)]
        axes[1].plot(xs, ms, marker="o", markersize=size, **MARKER_EDGE,
                     linestyle="-", linewidth=1.6, color=colour,
                     label=f"{label} mean", zorder=z)

    # Best model per epoch, formerly its own panel. Same quantity, same scale as
    # the mean, so it belongs on the same axes: the question "does the best
    # generated network beat the best it started from" is read against the means
    # rather than across two panels with different y ranges. Colour still marks
    # the entity; the square marker and dashed line mark the statistic, so mean
    # and max never depend on colour alone to be told apart.
    # The y view is settled BEFORE the best-model label is placed, so the label
    # can be positioned against limits that are already known and is never
    # clipped by a spine.
    accuracy_lo, accuracy_hi = _value_limits([
        v for r in rows
        for v in (r.get("accuracy_mean"), r.get("baseline_accuracy_mean"),
                  r.get("accuracy_max"), r.get("baseline_accuracy_max"))
    ])

    picked = _pick("baseline_accuracy_max")
    if picked is not None:
        xs, ys = picked
        axes[1].plot(xs, ys, marker="s", markersize=BEST_MARKER[0][0], **MARKER_EDGE,
                     linestyle="--", linewidth=1.1,
                     color=ORIGINAL, alpha=0.9, zorder=BEST_MARKER[0][1],
                     label="original best")
    picked = _pick("accuracy_max")
    if picked is not None:
        xs, ys = picked
        axes[1].plot(xs, ys, marker="s", markersize=BEST_MARKER[1][0], **MARKER_EDGE,
                     linestyle="--", linewidth=1.3,
                     color=BEST, alpha=0.95, zorder=BEST_MARKER[1][1],
                     label="generated best")
        running, best = [], float("-inf")
        for y in ys:
            best = max(best, y)
            running.append(best)
        axes[1].plot(xs, running, linestyle=":", linewidth=1.2, color=BEST,
                     alpha=0.6, zorder=4, label="generated best so far")

        # The best network of the whole run: the one number a reader looks for.
        # Placed against the settled limits so it stays on the panel wherever
        # the peak falls -- below the marker when the peak is near the top,
        # pushed inward at the first and last epoch where a centred label would
        # cross a spine.
        peak = max(range(len(ys)), key=lambda i: ys[i])
        span = max(accuracy_hi - accuracy_lo, 1e-9)
        near_top = (ys[peak] - accuracy_lo) / span > 0.82
        if peak == len(xs) - 1:
            dx, ha = -9, "right"
        elif peak == 0:
            dx, ha = 9, "left"
        else:
            dx, ha = 0, "center"
        dy, va = (-10, "top") if near_top else (9, "bottom")
        axes[1].annotate(
            f"best {ys[peak]:.4f}", (xs[peak], ys[peak]),
            textcoords="offset points", xytext=(dx, dy), ha=ha, va=va,
            fontsize=8, color=BEST, fontweight="bold", zorder=7,
            annotation_clip=False)

    axes[1].set_ylabel("accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylim(accuracy_lo, accuracy_hi)
    axes[1].set_title("Accuracy per epoch: mean with 95% CI, and best model "
                      "(generated vs. original networks)")
    if axes[1].get_legend_handles_labels()[0]:   # nothing to label on an all-failed epoch
        axes[1].legend(loc="best", fontsize=8, framealpha=0.75, ncol=2, fancybox=False)
    axes[1].grid(alpha=0.3)

    if ep:
        axes[1].set_xticks(sorted(set(ep)))
        if len(set(ep)) == 1:
            # A single epoch leaves matplotlib to invent an x range; fix it so
            # the whiskers drawn by _draw_band sit at a sensible width.
            only = ep[0]
            axes[1].set_xlim(only - 0.5, only + 0.5)
    _draw_brand(fig)
    fig.tight_layout(rect=(0, 0.022, 1, max(0.70, header_bottom)), pad=0.6)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# Run artifacts produced by the statistics layer. Everything matching these is
# a record OF A SINGLE RUN, so carrying it into the next one silently mixes two
# experiments: epoch_tracker.json is keyed by epoch, and a shorter run only
# overwrites the epochs it reaches, leaving the tail of a longer previous run
# in the file and on the plot.
RUN_ARTIFACT_NAMES = ("cycle_results.json", "epoch_tracker.json",
                      "epoch_statistics.png", "generation_seeds.json")
RUN_ARTIFACT_GLOBS = ("cycle_results_*.json",)


def reset_run_dir(base_dir, dry_run=False):
    """Delete the whole run directory so a new run starts from nothing.

    The per-file list below is what this used to do, and keeping it correct
    meant extending RUN_ARTIFACT_NAMES every time the pipeline learned to write
    something new -- generation_seeds.json was added late, epoch directories
    were never covered at all and were removed separately inside tune(). Every
    file under ``out/nngpt`` belongs to one run, so removing the directory is
    both simpler and complete: statistics, pinned seeds, per-epoch model
    directories and adapters all go in one step, and nothing from a previous
    run can be mistaken for this one's.

    Returns the path when it was removed, else None. Never raises.
    """
    base_dir = Path(base_dir)
    if not base_dir.exists():
        print(f"[RUN RESET] Nothing to remove: {base_dir} does not exist")
        return None
    if dry_run:
        return base_dir
    import shutil
    from ab.gpt.util.data.LemurSubmission import archive_submission
    archive_submission(base_dir)
    from ab.gpt.util.data.RunArchive import archive_run_results
    archive_run_results(base_dir)
    try:
        shutil.rmtree(base_dir)
        print(f"[RUN RESET] Removed the previous run directory: {base_dir}")
    except Exception as exc:
        print(f"[RUN RESET] Could not remove {base_dir}: {exc}")
        return None
    try:
        base_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"[RUN RESET] Could not recreate {base_dir}: {exc}")
    return base_dir


def reset_run_artifacts(base_dir, dry_run=False):
    """Delete the previous run's statistics files from ``base_dir``.

    Returns the list of paths removed. Never raises: a file that cannot be
    deleted is reported and the run continues.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return []

    targets = []
    for name in RUN_ARTIFACT_NAMES:
        p = base_dir / name
        if p.is_file():
            targets.append(p)
    for pattern in RUN_ARTIFACT_GLOBS:
        targets.extend(sorted(p for p in base_dir.glob(pattern) if p.is_file()))

    removed = []
    for p in dict.fromkeys(targets):          # de-duplicate, keep order
        if dry_run:
            removed.append(p)
            continue
        try:
            p.unlink()
            removed.append(p)
        except Exception as exc:
            print(f"[RUN RESET] Could not remove {p}: {exc}")

    if removed:
        print(f"[RUN RESET] Removed {len(removed)} file(s) from a previous run in {base_dir}:")
        for p in removed:
            print(f"[RUN RESET]   {p.name}")
    else:
        print(f"[RUN RESET] No previous run artifacts found in {base_dir}")
    return removed


def _cycle_results_from_stats(stats):
    """Build a cycle_results payload from the collected statistics."""
    valid = stats.get("models_valid", 0)
    total = stats.get("models_generated", 0)
    return {
        "cycle": stats.get("epoch"),
        "success": bool(valid),
        "training": {
            "data_dir": None,
            "total_examples": None,
            "new_examples_added": None,
            "training_time_minutes": None,
        },
        "generation": {
            "total_generated": total,
            "successful": valid,
            "novel": valid,
        },
        "evaluation": {
            "models_trained": valid,
            "best_accuracy": stats.get("accuracy_max"),
            "avg_accuracy": stats.get("accuracy_mean"),
            "success_rate": stats.get("valid_rate"),
        },
        "cycle_time_minutes": None,
        "statistics": stats,
    }


def finalize_epoch_results(base_dir, epoch, stats):
    """Make sure this epoch's results are on disk under their own name.

    Called as soon as the validation phase for ``epoch`` finishes, so the record
    is complete before the next fine-tuning cycle starts rather than at some
    later point that may never arrive. Three cases are covered:

    * NNEval already wrote ``cycle_results_<epoch>.json`` -- the statistics
      block is merged into it;
    * only the generic ``cycle_results.json`` exists and it describes this
      epoch -- it is adopted under the per-epoch name (this is the run that
      "kept the name cycle_results.json");
    * nothing was written, because evaluation was skipped or interrupted -- the
      record is rebuilt from the statistics so the epoch is still represented.
    """
    base_dir = Path(base_dir)
    per_epoch = base_dir / f"cycle_results_{epoch}.json"
    generic = base_dir / "cycle_results.json"

    payload = None
    for candidate in (per_epoch, generic):
        if not candidate.is_file():
            continue
        try:
            with open(candidate) as f:
                loaded = json.load(f)
        except Exception:
            continue
        # Only adopt a file that really describes this epoch; the generic one
        # may still hold a previous cycle.
        if isinstance(loaded, dict) and loaded.get("cycle") in (epoch, None):
            payload = loaded
            payload["cycle"] = epoch
            break

    if payload is None:
        payload = _cycle_results_from_stats(stats)
    else:
        payload["statistics"] = stats

    base_dir.mkdir(parents=True, exist_ok=True)
    for path in (per_epoch, generic):
        try:
            with open(path, "w") as f:
                json.dump(payload, f, indent=2, default=str)
        except Exception as exc:
            print(f"[EPOCH STATS] Could not write {path}: {exc}")
    print(f"[EPOCH STATS] Epoch {epoch} results finalised -> {per_epoch}")
    return per_epoch


def update_epoch_report(models_dir, epoch, base_dir):
    """Collect, persist and plot this epoch's statistics.

    Never raises: a reporting failure must not abort a training run. Returns the
    stats dict, or None if collection itself failed.
    """
    base_dir = Path(base_dir)
    try:
        stats = collect_epoch_stats(models_dir, epoch)
    except Exception as exc:
        print(f"[EPOCH STATS] Could not collect statistics for epoch {epoch}: {exc}")
        return None

    tracker_path = base_dir / "epoch_tracker.json"
    try:
        rows = append_epoch_stats(tracker_path, stats)
    except Exception as exc:
        print(f"[EPOCH STATS] Could not write {tracker_path}: {exc}")
        rows = [stats]

    def fmt(v, pct=False):
        if v is None:
            return "n/a"
        return f"{100 * v:.2f}%" if pct else f"{v:.4f}"

    print(
        f"[EPOCH STATS] epoch {epoch}: "
        f"valid {stats['models_valid']}/{stats['models_generated']} "
        f"({fmt(stats['valid_rate'], pct=True)}, "
        f"95% CI {fmt(stats['valid_rate_ci_low'], pct=True)}"
        f"-{fmt(stats['valid_rate_ci_high'], pct=True)}) | "
        f"accuracy mean {fmt(stats['accuracy_mean'])} "
        f"[{fmt(stats['accuracy_ci_low'])}, {fmt(stats['accuracy_ci_high'])}] "
        f"max {fmt(stats['accuracy_max'])} | "
        f"before modification mean {fmt(stats['baseline_accuracy_mean'])} "
        f"[{fmt(stats['baseline_accuracy_ci_low'])}, {fmt(stats['baseline_accuracy_ci_high'])}] "
        f"(n={stats['baseline_accuracy_n']})"
        + (f" | datasets {', '.join(stats.get('datasets') or [])}"
           if stats.get("datasets") else "")
    )

    seeds = stats.get("baseline_models_with_dataset") or []
    if seeds:
        print(f"[EPOCH STATS] epoch {epoch} seed models (architecture + dataset "
              f"they were trained on): {', '.join(seeds)}")

    try:
        finalize_epoch_results(base_dir, epoch, stats)
    except Exception as exc:
        print(f"[EPOCH STATS] Could not finalise epoch results: {exc}")

    try:
        png = plot_epoch_stats(rows, base_dir / "epoch_statistics.png", base_dir=base_dir)
        if png:
            print(f"[EPOCH STATS] Plot updated -> {png}")
    except Exception as exc:
        print(f"[EPOCH STATS] Plot skipped: {exc}")

    return stats