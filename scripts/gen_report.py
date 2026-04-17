#!/usr/bin/env python3
"""
gen_report.py — Parse RT benchmark QEMU output and generate comparison charts.

Handles three benchmark types from CSV_BEGIN/CSV_END blocks:
  BENCH,  — rtbench (realistic workload, U=1.90)
  VESTAL, — rtvestal (Vestal 2007 inverted-priority MC, U=1.33)
  MALA,   — rtmaladalen (Mälardalen WCET ports, U=1.16)

Usage:
  cat results.txt | python3 scripts/gen_report.py
  cat results.txt | python3 scripts/gen_report.py --suite vestal
  python3 scripts/gen_report.py --file scripts/bench_results.csv
"""

import sys
import os
import argparse

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Task criticality maps per suite
CRIT_MAP = {
    "BENCH": {
        "sensor_read": True, "control_loop": True, "display_render": True,
        "network_send": False, "data_logging": False, "background_sync": False,
    },
    "VESTAL": {
        "flight_ctrl": True, "safety_mon": True, "actuator_cmd": True,
        "sensor_poll": False, "telemetry": False, "log_write": False,
    },
    "MALA": {
        "matmul": True, "bsort100": True, "crc": True,
        "prime": False, "cnt": False, "fibcall": False,
    },
    "GUI": {
        "input_poll": False, "render": True, "blit": True,
        "flood_fill": False, "crc_save": False, "undo_snap": False,
    },
    "DRONE": {
        "imu_read": True, "ahrs_filter": True, "pid_control": True,
        "actuator_upd": False, "telemetry": False, "data_log": False,
    },
}

SUITE_LABELS = {
    "BENCH":  "rtbench (U=1.90)",
    "VESTAL": "Vestal MC (U=1.33)",
    "MALA":   "Mälardalen (U=1.16)",
    "GUI":    "rtgui/MS Paint (U=1.04)",
    "DRONE":  "rtdrone Q10 (U=1.50)",
}


def parse_input(lines):
    """
    Returns {suite: {mode: {task_name: (completions, misses, is_critical)}}}
    """
    results = {"BENCH": {}, "VESTAL": {}, "MALA": {}, "GUI": {}, "DRONE": {}}
    in_csv = False

    for line in lines:
        line = line.strip()
        if line == "CSV_BEGIN":
            in_csv = True
            continue
        if line == "CSV_END":
            in_csv = False
            continue
        if not in_csv:
            continue
        parts = line.split(",")
        if len(parts) != 6:
            continue
        suite, mode, _tid, name, comp, miss = parts
        suite = suite.strip()
        mode  = mode.strip() or "NN"
        name  = name.strip()
        if suite not in results:
            continue
        try:
            c, m = int(comp), int(miss)
        except ValueError:
            continue
        crit = CRIT_MAP.get(suite, {}).get(name, False)
        if mode not in results[suite]:
            results[suite][mode] = {}
        results[suite][mode][name] = (c, m, crit)

    # Remove empty suites
    return {s: d for s, d in results.items() if d}


def totals(task_data):
    tc = sum(c for c, m, _ in task_data.values())
    cm = sum(m for c, m, k in task_data.values() if k)
    sm = sum(m for c, m, k in task_data.values() if not k)
    return tc, cm, sm, cm + sm


def print_suite_summary(suite, data):
    label = SUITE_LABELS.get(suite, suite)
    print(f"\n{'='*52}")
    print(f"  {label}")
    print(f"{'='*52}")
    print(f"  {'Mode':4s}  {'Completions':>11s}  {'HI Misses':>9s}  "
          f"{'LO Misses':>9s}  {'Total':>5s}")
    print(f"  {'-'*4}  {'-'*11}  {'-'*9}  {'-'*9}  {'-'*5}")
    for mode in ["NN", "EDF", "RMS", "RR", "MLFQ"]:
        if mode not in data:
            continue
        tc, cm, sm, tm = totals(data[mode])
        hi_marker = " **" if mode == "NN" and cm == 0 else ""
        print(f"  {mode:4s}  {tc:>11d}  {cm:>9d}  {sm:>9d}  {tm:>5d}{hi_marker}")
    print()


def latex_table(suite, data, caption_extra=""):
    modes = [m for m in ["NN", "EDF", "RMS", "RR", "MLFQ"] if m in data]
    label = SUITE_LABELS.get(suite, suite)
    header = " & ".join(f"\\textbf{{{m}}}" for m in modes)
    rows = {
        "HI-critical misses": lambda d: totals(d)[1],
        "LO-soft misses":     lambda d: totals(d)[2],
        "Total misses":       lambda d: totals(d)[3],
        "Completions":        lambda d: totals(d)[0],
    }
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        f"\\begin{{tabular}}{{l{'c'*len(modes)}}}",
        "\\hline",
        f"Metric & {header} \\\\",
        "\\hline",
    ]
    for row_label, fn in rows.items():
        vals = " & ".join(str(fn(data[m])) if m in data else "—" for m in modes)
        lines.append(f"{row_label} & {vals} \\\\")
    lines += [
        "\\hline",
        "\\end{tabular}",
        f"\\caption{{{label} — xv6-riscv RT scheduler comparison{caption_extra}}}",
        f"\\label{{tab:sched_{suite.lower()}}}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def generate_chart(all_data, outdir="report_figures"):
    os.makedirs(outdir, exist_ok=True)
    suites = [s for s in ["BENCH", "VESTAL", "MALA", "GUI", "DRONE"] if s in all_data]
    modes  = ["NN", "EDF", "RMS", "RR", "MLFQ"]

    # ── Figure 1: HI-critical misses across all suites ──────────────
    fig, axes = plt.subplots(1, len(suites), figsize=(5 * len(suites), 5),
                             sharey=False)
    if len(suites) == 1:
        axes = [axes]
    fig.suptitle("HI-Critical Task Deadline Misses\n"
                 "NN (trained) vs classical schedulers — xv6-riscv",
                 fontsize=13, fontweight="bold")

    colors = {"NN": "#2ecc71", "EDF": "#e74c3c", "RMS": "#e67e22", "RR": "#95a5a6", "MLFQ": "#9b59b6"}

    for ax, suite in zip(axes, suites):
        data = all_data[suite]
        present = [m for m in modes if m in data]
        crit_m  = [totals(data[m])[1] for m in present]
        x = range(len(present))
        bars = ax.bar(x, crit_m,
                      color=[colors.get(m, "#3498db") for m in present],
                      alpha=0.88, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, padding=3, fontsize=10, fontweight="bold")
        ax.set_title(SUITE_LABELS.get(suite, suite), fontsize=10)
        ax.set_xticks(list(x))
        ax.set_xticklabels(present, fontsize=11)
        ax.set_ylabel("HI-Critical Misses" if ax is axes[0] else "")
        ax.set_ylim(0, max(crit_m) * 1.4 + 1)
        ax.axhline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    out1 = os.path.join(outdir, "crit_misses_comparison.png")
    plt.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"[gen_report] Saved: {out1}")

    # ── Figure 2: side-by-side stacked misses per suite ─────────────
    if len(suites) >= 2:
        fig2, axes2 = plt.subplots(1, len(suites), figsize=(5 * len(suites), 5),
                                   sharey=False)
        if len(suites) == 1:
            axes2 = [axes2]
        fig2.suptitle("Deadline Misses: HI-Critical vs LO-Soft\n"
                      "xv6-riscv RT Scheduler Benchmark",
                      fontsize=13, fontweight="bold")

        for ax, suite in zip(axes2, suites):
            data = all_data[suite]
            present = [m for m in ["NN", "EDF", "RMS", "RR", "MLFQ"] if m in data]
            cm = [totals(data[m])[1] for m in present]
            sm = [totals(data[m])[2] for m in present]
            x  = np.arange(len(present))
            w  = 0.6
            b1 = ax.bar(x, cm, w, label="HI-critical", color="#e74c3c", alpha=0.85)
            b2 = ax.bar(x, sm, w, bottom=cm, label="LO-soft", color="#f39c12", alpha=0.75)
            ax.bar_label(b1, label_type="center", fmt="%d", color="white",
                         fontsize=9, fontweight="bold")
            ax.set_title(SUITE_LABELS.get(suite, suite), fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(present, fontsize=11)
            ax.set_ylabel("Deadline Misses" if ax is axes2[0] else "")
            ax.legend(fontsize=9)

        plt.tight_layout()
        out2 = os.path.join(outdir, "stacked_misses.png")
        plt.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"[gen_report] Saved: {out2}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Input file (default: stdin)")
    parser.add_argument("--suite", choices=["bench", "vestal", "mala"],
                        help="Print LaTeX only for this suite")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    all_data = parse_input(lines)
    if not all_data:
        print("[gen_report] No CSV data found. Run rtvestal/rtbench/rtmaladalen "
              "and pipe output here.")
        sys.exit(1)

    for suite, data in all_data.items():
        print_suite_summary(suite, data)

    # LaTeX tables
    filter_suite = args.suite.upper() if args.suite else None
    for suite, data in all_data.items():
        if filter_suite and suite != filter_suite:
            continue
        print(f"\n% ── {suite} ──")
        print(latex_table(suite, data))

    if HAS_MATPLOTLIB:
        generate_chart(all_data)
    else:
        print("\n[gen_report] Install matplotlib: uv add matplotlib")


if __name__ == "__main__":
    main()
