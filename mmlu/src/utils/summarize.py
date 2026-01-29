#!/usr/bin/env python3
import os
import re
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FOLDER_RE = re.compile(
    r"checkpoint-(?P<step>\d+)_t(?P<temperature>[-+]?\d*\.?\d+)_n(?P<test_sample_size>\d+)_s(?P<shot>\d+)_seed(?P<seed>\d+)",
    re.IGNORECASE,
)
BASE_METRICS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]
CANONICAL_METRICS = BASE_METRICS + [m + "_se" for m in BASE_METRICS]
METRIC_KEYS = {
    re.sub(r"[\s_]+", "", metric).lower(): metric
    for metric in CANONICAL_METRICS
}

def parse_folder_name(name):
    match = FOLDER_RE.fullmatch(name)
    if not match:
        return None
    info = {
        "step": int(match.group("step")),
        "shot": int(match.group("shot")),
        "temperature": float(match.group("temperature")),
        "test_sample_size": int(match.group("test_sample_size")),
        "seed": int(match.group("seed")),
    }
    return info

def get_color(shot, max_shot, cmap_name="viridis"):
    cmap = plt.get_cmap(cmap_name)
    if max_shot and max_shot > 0:
        norm_shot = 1.0 - (shot / max_shot)
        return cmap(norm_shot)
    else:
        return cmap(0.0)

def parse_results_log(path):
    if not os.path.isfile(path):
        return None
    vals = {m: None for m in BASE_METRICS}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.match(r"^\s*([^:]+)\s*:\s*([-+]?[0-9]*\.?[0-9]+)", line)
            if not m:
                continue
            label, num = m.group(1).strip(), m.group(2)
            canon = METRIC_KEYS.get(re.sub(r"[\s_]+", "", label).lower())
            vals[canon] = float(num)
    return vals


def iter_result_rows(base_dir):
    abs_base = os.path.abspath(base_dir)
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        info = parse_folder_name(name)
        if not info:
            continue
        parsed = parse_results_log(os.path.join(full, "results.log"))
        if parsed is None:
            continue
        row = {"source_dir": abs_base, **info}
        for metric in BASE_METRICS:
            row[metric] = parsed.get(metric)
        yield row

def make_dataframe(base_dirs):
    frames = []
    for base_dir in base_dirs:
        rows = list(iter_result_rows(base_dir))
        df = pd.DataFrame(rows).sort_values(["shot", "step"]).reset_index(drop=True)
        frames.append(df)
    return pd.concat(frames, ignore_index=True).sort_values(["source_dir", "shot", "step"]).reset_index(drop=True)

def aggregate_metrics(df):
    grouped = df.groupby(["shot", "step", "temperature", "test_sample_size"], dropna=False)

    summary = grouped[BASE_METRICS].mean()
    summary["seed_count"] = grouped["seed"].nunique()

    std_df = grouped[BASE_METRICS].std(ddof=0).fillna(0)
    for metric in BASE_METRICS:
        summary[f"{metric}_se"] = std_df[metric] / np.sqrt(summary["seed_count"])

    return summary.reset_index().sort_values(["shot", "step"]).reset_index(drop=True)

def plot_lines(metric, df, out, include_shots=None, ylim=None, compare_df=None):
    if include_shots is None:
        include_shots = df["shot"].unique()
    max_shot = max(include_shots)
    se_col = f"{metric}_se"

    def _plot_line(df_to_plot, plot_kwargs=None, fill_std=False):
        df_to_plot_sorted = df_to_plot.sort_values("step")
        line_kwargs = {"marker": "o"}
        if plot_kwargs:
            line_kwargs.update(plot_kwargs)

        plt.plot(df_to_plot_sorted["step"], df_to_plot_sorted[metric], **line_kwargs)
        if fill_std and se_col in df_to_plot_sorted.columns:
            std_vals = df_to_plot_sorted[se_col].fillna(0).to_numpy()
            if np.any(std_vals > 0):
                lower = df_to_plot_sorted[metric] - std_vals
                upper = df_to_plot_sorted[metric] + std_vals
                plt.fill_between(df_to_plot_sorted["step"], lower, upper, color=color, alpha=0.2)

    plt.figure()
    for shot in include_shots:
        color = get_color(shot, max_shot)
        df_org = df.groupby("shot", dropna=False).get_group(shot)
        if compare_df is None:
            _plot_line(df_org, plot_kwargs={"label": f"{shot}-shot", "color": color}, fill_std=True)
        else:
            df_ours = compare_df.groupby("shot", dropna=False).get_group(shot)
            _plot_line(df_org, plot_kwargs={"label": f"{shot}-shot", "linestyle": "--", "color": color}, fill_std=False)
            _plot_line(df_ours, plot_kwargs={"label": f"{shot}-shot (Ours)", "color": color}, fill_std=True)

    plt.xlabel("step")
    plt.ylabel("accuracy")
    plt.title(metric)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        plt.legend(handles, labels)
    else:
        plt.legend()
    plt.grid(True, ls=":")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description="Summarize results and produce plots.")
    p.add_argument("--base_dir", type=str, nargs="+", required=True, help="One or more directories with checkpoint-* subfolders")
    p.add_argument("--output_dir", type=str, default="plots")
    p.add_argument("--compare_dir", type=str, nargs="+", help="Optional directories to compare against base_dir.")
    p.add_argument("--shots", type=int, nargs="+", help="Only plot the specified shot counts (e.g. --shots 1 3 5).")
    p.add_argument("--ylim", type=float, nargs=2, metavar=("YMIN", "YMAX"), help="Set y-axis limits for plots.")
    args = p.parse_args()

    raw_df = make_dataframe(args.base_dir)
    agg_df = aggregate_metrics(raw_df)

    os.makedirs(args.output_dir, exist_ok=True)
    raw_df.to_csv(args.output_dir+"/results.csv", index=False)
    agg_df.to_csv(args.output_dir+"/results_summary.csv", index=False)

    if args.compare_dir:
        compare_raw = make_dataframe(args.compare_dir)
        compare_df = aggregate_metrics(compare_raw)
    else:
        compare_df = None

    target_shots = sorted(set(args.shots)) if args.shots else None
    target_ylim = tuple(args.ylim) if args.ylim else None

    for metric in BASE_METRICS:
        plot_lines(metric, agg_df, args.output_dir+f"/{metric}.png", target_shots, target_ylim, compare_df)

    with pd.option_context("display.width", 120, "display.max_columns", None):
        print("\nPreview of aggregated results:")
        print(agg_df.head(20))

if __name__ == "__main__":
    main()
