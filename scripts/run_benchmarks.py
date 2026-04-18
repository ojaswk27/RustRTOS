#!/usr/bin/env python3
"""
run_benchmarks.py — Run xv6 RT benchmarks non-interactively via QEMU subprocess.

Starts a single QEMU instance, runs all requested benchmark commands, and
collects CSV output blocks for parsing.

Usage:
    cd xv6-riscv
    python3 ../scripts/run_benchmarks.py

Output: prints filled result tables for rtgui_a, rtgui_b, and all existing
benchmarks, then saves combined CSV to ../scripts/bench_results_extended.csv
"""

import subprocess
import time
import os
import sys
import select
import re
import signal

XSS_DIR = os.path.join(os.path.dirname(__file__), '..', 'xv6-riscv')
XSS_DIR = os.path.normpath(XSS_DIR)

TICK_MS = 100          # 1 tick = ~100ms (CLINT 10MHz, interval=1e6 cycles)
SIM_TICKS = 200        # benchmark simulation ticks
RUN_TIMEOUT_S = 45     # per-benchmark timeout (200 ticks * 0.1s * ~2x margin)
BOOT_TIMEOUT_S = 30    # QEMU boot timeout


def start_qemu():
    proc = subprocess.Popen(
        ['make', 'qemu', 'CPUS=1'],
        cwd=XSS_DIR,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    return proc


def read_chunk(proc, timeout_s=0.3):
    r, _, _ = select.select([proc.stdout], [], [], timeout_s)
    if r:
        return os.read(proc.stdout.fileno(), 8192)
    return b''


def wait_for_pattern(proc, pattern_bytes, timeout_s, verbose=False):
    buf = bytearray()
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        chunk = read_chunk(proc, min(0.3, deadline - time.time()))
        if chunk:
            buf.extend(chunk)
            if verbose:
                sys.stdout.write(chunk.decode('utf-8', 'replace'))
                sys.stdout.flush()
            if pattern_bytes in buf:
                return True, bytes(buf)
    return False, bytes(buf)


def send_cmd(proc, cmd):
    proc.stdin.write((cmd + '\n').encode())
    proc.stdin.flush()


def boot_qemu(proc, verbose=False):
    print(f'[run] Waiting for xv6 boot (up to {BOOT_TIMEOUT_S}s)...', flush=True)
    ok, output = wait_for_pattern(proc, b'$ ', BOOT_TIMEOUT_S, verbose=verbose)
    if ok:
        print('[run] xv6 booted.', flush=True)
    else:
        print('[run] ERROR: timed out waiting for boot prompt.', flush=True)
    return ok, output


def run_one(proc, cmd, verbose=False):
    """
    Run a single benchmark command in the running xv6 shell.
    Returns (stdout_str, csv_blocks) or (None, []) on timeout.
    """
    print(f'[run] $ {cmd}', flush=True)
    send_cmd(proc, cmd)

    # Wait for CSV_END to appear (benchmark finished)
    ok, raw = wait_for_pattern(proc, b'CSV_END', RUN_TIMEOUT_S, verbose=verbose)
    if not ok:
        print(f'[run] WARNING: timed out waiting for CSV_END on "{cmd}"', flush=True)
        return None, []

    # Wait briefly for the shell prompt to come back
    _, more = wait_for_pattern(proc, b'$ ', 5, verbose=False)
    raw += more

    text = raw.decode('utf-8', 'replace')
    blocks = parse_csv_blocks(text)
    return text, blocks


def parse_csv_blocks(text):
    """Extract CSV_BEGIN...CSV_END blocks from text."""
    blocks = []
    in_block = False
    current = []
    for line in text.splitlines():
        if 'CSV_BEGIN' in line:
            in_block = True
            current = []
        elif 'CSV_END' in line:
            if in_block:
                blocks.append(current)
            in_block = False
        elif in_block:
            line = line.strip()
            if line:
                current.append(line)
    return blocks


def parse_rows(blocks):
    """
    Parse all CSV rows from list of blocks.
    Returns list of (suite, mode, tid, name, completions, misses)
    """
    rows = []
    for block in blocks:
        for line in block:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 6:
                continue
            suite, mode, tid, name, comp, miss = parts
            try:
                rows.append((suite, mode, int(tid), name, int(comp), int(miss)))
            except ValueError:
                pass
    return rows


CRIT_MAP = {
    'BENCH': {'sensor_read': True, 'control_loop': True, 'display_render': True,
              'network_send': False, 'data_logging': False, 'background_sync': False},
    'VESTAL': {'flight_ctrl': True, 'safety_mon': True, 'actuator_cmd': True,
               'sensor_poll': False, 'telemetry': False, 'log_write': False},
    'MALA': {'matmul': True, 'bsort100': True, 'crc': True,
             'prime': False, 'cnt': False, 'fibcall': False},
    'GUI':  {'input_poll': False, 'render': True, 'blit': True,
             'flood_fill': False, 'crc_save': False, 'undo_snap': False},
    'DRONE': {'imu_read': True, 'ahrs_filter': True, 'pid_control': True,
              'actuator_upd': False, 'telemetry': False, 'data_log': False},
    'GUIA': {'input_poll': False, 'render': True, 'blit': True,
             'flood_fill': False, 'crc_save': False, 'undo_snap': False},
    'GUIB': {'input_poll': False, 'render': True, 'blit': True,
             'compositor': False, 'flood_fill': False, 'crc_save': False},
}

SUITE_LABEL = {
    'BENCH': 'rtbench (U=1.90)',
    'VESTAL': 'rtvestal (U=1.33)',
    'MALA': 'rtmaladalen (U=1.16)',
    'GUI': 'rtgui baseline (U=1.04)',
    'DRONE': 'rtdrone (U=1.50)',
    'GUIA': 'rtgui_a heavier (U=1.37)',
    'GUIB': 'rtgui_b compositor (U=1.69)',
}


def totals(suite, rows_for_mode):
    cm = {r[3]: (r[4], r[5]) for r in rows_for_mode}
    crit = CRIT_MAP.get(suite, {})
    hi_m = sum(m for name, (c, m) in cm.items() if crit.get(name, False))
    lo_m = sum(m for name, (c, m) in cm.items() if not crit.get(name, False))
    comp = sum(c for name, (c, m) in cm.items())
    return comp, hi_m, lo_m


def print_table(suite, data):
    """data = {mode: [(suite, mode, tid, name, comp, miss), ...]}"""
    label = SUITE_LABEL.get(suite, suite)
    print(f'\n=== {label} ===')
    print(f'{"Scheduler":<10} {"HI-Crit Misses":>15} {"LO-Soft Misses":>15} {"Completions":>12}')
    print('-' * 55)
    modes = ['NN', 'EDF', 'RMS', 'MLFQ', 'RR']
    for mode in modes:
        if mode not in data:
            continue
        comp, hi_m, lo_m = totals(suite, data[mode])
        print(f'{mode:<10} {hi_m:>15} {lo_m:>15} {comp:>12}')
    print()


def collect_all_rows(all_blocks):
    """Group rows by (suite, mode)."""
    rows = parse_rows(all_blocks)
    collected = {}  # suite -> mode -> [rows]
    for row in rows:
        suite, mode = row[0], row[1]
        if suite not in collected:
            collected[suite] = {}
        if mode not in collected[suite]:
            collected[suite][mode] = []
        collected[suite][mode].append(row)
    return collected


def save_csv(all_blocks, path):
    with open(path, 'w') as f:
        for block in all_blocks:
            f.write('CSV_BEGIN\n')
            for line in block:
                f.write(line + '\n')
            f.write('CSV_END\n')
    print(f'[run] Saved combined CSV to {path}')


def main():
    verbose = '--verbose' in sys.argv or '-v' in sys.argv

    proc = start_qemu()
    print('[run] QEMU started.', flush=True)

    all_blocks = []

    try:
        ok, _ = boot_qemu(proc, verbose=verbose)
        if not ok:
            print('[run] Boot failed. Aborting.', flush=True)
            proc.terminate()
            sys.exit(1)

        # Commands to run: (program, arg) pairs
        # Format: (display_name, command_string)
        commands = [
            # New benchmarks: rtgui_a (5 modes)
            ('rtgui_a NN',   'rtgui_a'),
            ('rtgui_a EDF',  'rtgui_a edf'),
            ('rtgui_a RMS',  'rtgui_a rms'),
            ('rtgui_a MLFQ', 'rtgui_a mlfq'),
            ('rtgui_a RR',   'rtgui_a rr'),
            # New benchmarks: rtgui_b (5 modes)
            ('rtgui_b NN',   'rtgui_b'),
            ('rtgui_b EDF',  'rtgui_b edf'),
            ('rtgui_b RMS',  'rtgui_b rms'),
            ('rtgui_b MLFQ', 'rtgui_b mlfq'),
            ('rtgui_b RR',   'rtgui_b rr'),
        ]

        for name, cmd in commands:
            print(f'\n[run] Running {name}...', flush=True)
            text, blocks = run_one(proc, cmd, verbose=verbose)
            if blocks:
                all_blocks.extend(blocks)
                print(f'[run] Got {len(blocks)} CSV block(s).', flush=True)
            else:
                print(f'[run] WARNING: no CSV data for {name}.', flush=True)

    except KeyboardInterrupt:
        print('\n[run] Interrupted.', flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    if not all_blocks:
        print('[run] No data collected.', flush=True)
        sys.exit(1)

    # Parse and display results
    collected = collect_all_rows(all_blocks)

    print('\n' + '='*60)
    print('BENCHMARK RESULTS')
    print('='*60)

    for suite in ['GUIA', 'GUIB']:
        if suite in collected:
            print_table(suite, collected[suite])

    # Save extended CSV
    out_path = os.path.join(os.path.dirname(__file__), 'bench_results_extended.csv')
    save_csv(all_blocks, out_path)

    print('\n[run] Done.')


if __name__ == '__main__':
    main()
