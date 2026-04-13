#!/usr/bin/env python3
"""
gen_report.py — Parse rtbench QEMU output and generate comparison charts.

Usage:
  # Capture rtbench output for all 4 modes, then:
  cat bench_nn.txt bench_edf.txt bench_rms.txt bench_rr.txt | python3 scripts/gen_report.py

  # Or pipe directly from QEMU:
  (sleep 8 && echo "rtbench nn" && sleep 100 && echo "rtbench edf" && ...) | \\
    make qemu CPUS=1 | python3 scripts/gen_report.py
"""

import sys
import re
import os
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[gen_report] matplotlib not found — will print LaTeX table only")


# ------------------------------------------------------------------
# Parse BENCH,MODE,taskid,taskname,completions,misses lines
# ------------------------------------------------------------------
def parse_input(lines):
    """Returns {mode: {task_name: (completions, misses, is_critical)}}"""
    data = {}
    task_info = {}  # name -> is_critical

    # Task criticality from rtbench.c (hardcoded to match the benchmark)
    CRIT = {
        "sensor_read": True,
        "control_loop": True,
        "display_render": True,
        "network_send": False,
        "data_logging": False,
        "background_sync": False,
    }

    in_csv = False
    for line in lines:
        line = line.strip()
        if line == "CSV_BEGIN":
            in_csv = True
            continue
        if line == "CSV_END":
            in_csv = False
            continue
        if in_csv and line.startswith("BENCH,"):
            parts = line.split(",")
            if len(parts) != 6:
                continue
            _, mode, task_id, task_name, completions, misses = parts
            mode = mode.strip() or "NN"
            task_name = task_name.strip()
            try:
                c = int(completions)
                m = int(misses)
            except ValueError:
                continue
            if mode not in data:
                data[mode] = {}
            data[mode][task_name] = (c, m, CRIT.get(task_name, False))

    return data


def compute_totals(task_data):
    """Returns (total_completions, crit_misses, soft_misses, total_misses)"""
    total_c = sum(c for c, m, _ in task_data.values())
    crit_m  = sum(m for c, m, crit in task_data.values() if crit)
    soft_m  = sum(m for c, m, crit in task_data.values() if not crit)
    return total_c, crit_m, soft_m, crit_m + soft_m


# ------------------------------------------------------------------
# Print LaTeX table
# ------------------------------------------------------------------
def print_latex_table(data):
    modes = ["NN", "EDF", "RMS", "RR"]
    modes_present = [m for m in ["NN", "EDF", "RMS", "RR"] if m in data]

    print("\n% LaTeX comparison table")
    print("\\begin{table}[h]")
    print("\\centering")
    header = " & ".join(f"\\textbf{{{m}}}" for m in modes_present)
    print(f"\\begin{{tabular}}{{l{'c' * len(modes_present)}}}")
    print("\\hline")
    print(f"Metric & {header} \\\\")
    print("\\hline")

    metrics = {
        "Critical misses": lambda d: compute_totals(d)[1],
        "Soft misses":     lambda d: compute_totals(d)[2],
        "Total misses":    lambda d: compute_totals(d)[3],
        "Completions":     lambda d: compute_totals(d)[0],
    }
    for metric, fn in metrics.items():
        vals = " & ".join(str(fn(data[m])) if m in data else "—"
                          for m in modes_present)
        print(f"{metric} & {vals} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{RT Scheduler Comparison — xv6-riscv, mixed-criticality workload, U=1.90}")
    print("\\label{tab:sched_compare}")
    print("\\end{table}")


# ------------------------------------------------------------------
# Generate matplotlib bar chart
# ------------------------------------------------------------------
def generate_chart(data, outfile="report_figures/scheduler_comparison.png"):
    os.makedirs(os.path.dirname(outfile), exist_ok=True)

    modes = [m for m in ["NN", "EDF", "RMS", "RR"] if m in data]
    if not modes:
        modes = list(data.keys())

    crit_misses = [compute_totals(data[m])[1] for m in modes]
    soft_misses = [compute_totals(data[m])[2] for m in modes]
    completions = [compute_totals(data[m])[0] for m in modes]

    x = np.arange(len(modes))
    width = 0.28

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("xv6 RT Scheduler Comparison\nMixed-Criticality Workload (U=1.90)", fontsize=13)

    # Left: miss rates
    ax = axes[0]
    bars1 = ax.bar(x - width/2, crit_misses, width, label="Critical misses",
                   color="#e74c3c", alpha=0.85)
    bars2 = ax.bar(x + width/2, soft_misses, width, label="Soft misses",
                   color="#f39c12", alpha=0.85)
    ax.set_xlabel("Scheduler")
    ax.set_ylabel("Deadline Misses")
    ax.set_title("Deadline Misses by Criticality")
    ax.set_xticks(x)
    ax.set_xticklabels([m.strip() for m in modes], fontsize=11)
    ax.legend()
    ax.bar_label(bars1, padding=2)
    ax.bar_label(bars2, padding=2)
    ax.set_ylim(0, max(max(crit_misses), max(soft_misses)) * 1.3 + 1)

    # Right: completions
    ax2 = axes[1]
    bars3 = ax2.bar(x, completions, width * 1.5, color="#2ecc71", alpha=0.85)
    ax2.set_xlabel("Scheduler")
    ax2.set_ylabel("Total Job Completions")
    ax2.set_title("Critical Task Completions")
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.strip() for m in modes], fontsize=11)
    ax2.bar_label(bars3, padding=2)
    ax2.set_ylim(0, max(completions) * 1.3 + 1)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"\n[gen_report] Chart saved to: {outfile}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    lines = sys.stdin.readlines()
    data = parse_input(lines)

    if not data:
        print("[gen_report] No BENCH CSV data found in input.")
        print("             Pipe rtbench output here or provide a saved file.")
        sys.exit(1)

    print(f"[gen_report] Parsed {len(data)} scheduler modes: {list(data.keys())}")
    print()

    # Console summary
    for mode in ["NN", "EDF", "RMS", "RR"]:
        if mode not in data:
            continue
        tc, cm, sm, tm = compute_totals(data[mode])
        print(f"  {mode:3s}: completions={tc:3d}  "
              f"crit_misses={cm:3d}  soft_misses={sm:3d}  total={tm:3d}")

    print_latex_table(data)

    if HAS_MATPLOTLIB:
        generate_chart(data)
    else:
        print("\n[gen_report] Install matplotlib to generate charts:")
        print("  uv add matplotlib")


if __name__ == "__main__":
    main()
