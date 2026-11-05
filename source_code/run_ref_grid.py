#!/usr/bin/env python3
"""Run ref_sim_v3 over the experiment grid; quick and full results are isolated."""
import argparse
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# (label, h, sojourn time). Keep MMPP sojourn time constant when comparing h.
ARRIVALS = [
    ("Poisson", 1.0, 0.05),
    ("MMPP-h2", 2.0, 0.02),
    ("MMPP-h3", 3.0, 0.02),
    ("MMPP-h5", 5.0, 0.02),
]
RHOS = [0.5, 0.7, 0.9]
KS = [10, 20, 50]
PHIS = [-0.6, 0.0, 0.3, 0.6, 0.9]
REPS, SIMTIME, WARM = 20, 5000, 100


def tag(x):
    return str(x).replace("-", "m").replace(".", "p")


def make_jobs(binary, raw_dir, arrivals, rhos, ks, phis, reps, simtime, warm):
    jobs = []
    for (label, h, soj), rho, K, phi in itertools.product(arrivals, rhos, ks, phis):
        name = f"{label}_rho{tag(rho)}_K{K}_phi{tag(phi)}.csv"
        output = raw_dir / name
        command = [
            binary, "--mode=corr", f"--rho={rho}", f"--K={K}",
            f"--h={h}", f"--soj={soj}", f"--phi={phi}",
            f"--simTime={simtime}", f"--warm={warm}", f"--reps={reps}",
        ]
        jobs.append((command, output))
    return jobs


def run_one(job):
    command, output = job
    part = Path(str(output) + ".part")
    if output.exists():
        return "skipped", str(output), ""
    part.unlink(missing_ok=True)
    try:
        result = subprocess.run(
            command + [f"--csv={part}"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        return "failed", str(output), str(exc)
    if result.returncode != 0 or not part.is_file():
        detail = (result.stderr or result.stdout or "No simulator output")[-2000:]
        part.unlink(missing_ok=True)
        return "failed", str(output), detail
    try:
        frame = pd.read_csv(part)
        if frame.empty:
            raise ValueError("CSV has no data rows")
        if "lost" not in frame.columns:
            raise ValueError("CSV is missing the expected 'lost' column")
    except Exception as exc:
        part.unlink(missing_ok=True)
        return "failed", str(output), f"Invalid simulator CSV: {exc}"
    os.replace(part, output)
    return "completed", str(output), ""


def main():
    parser = argparse.ArgumentParser(description="Run the finite-buffer simulator grid")
    parser.add_argument("--bin", default="./ref_sim", help="Simulator executable")
    parser.add_argument("--out", default="ref_results", help="Output directory")
    parser.add_argument("--quick", action="store_true", help="Run a small smoke test")
    parser.add_argument("--workers", type=int, default=2, help="Parallel processes (default: 2)")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    binary = os.path.abspath(args.bin)
    if not os.path.isfile(binary) or not os.access(binary, os.X_OK):
        parser.error(f"Simulator executable not found or not executable: {binary}")

    arrivals, rhos, ks, phis = ARRIVALS, RHOS, KS, PHIS
    reps, simtime, warm = REPS, SIMTIME, WARM
    out_name = args.out
    if args.quick:
        out_name = "ref_results_quick" if args.out == "ref_results" else args.out.rstrip("/") + "_quick"
        rhos, ks, phis = [0.7], [20], [0.0, 0.6]
        reps, simtime, warm = 2, 400, 20

    out_dir = Path(out_name)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "binary": binary,
        "arrivals": [[label, h, soj] for label, h, soj in ARRIVALS],
        "rhos": rhos,
        "Ks": ks,
        "phis": phis,
        "reps": reps,
        "simTime": simtime,
        "warm": warm,
        "quick": args.quick,
    }
    manifest = out_dir / "run_manifest.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous != config:
            parser.error(
                f"{out_dir} contains results from different settings. Use a new --out "
                "directory or move the old results before changing the experiment."
            )
    else:
        old_csvs = list(raw_dir.glob("*.csv"))
        if old_csvs:
            parser.error(
                f"{raw_dir} contains CSVs but no run_manifest.json. Use a new output "
                "directory or move the old raw CSVs first."
            )
        manifest.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    print("High-state offered load: rhoHigh = 2*rho*h/(1+h); values > 1 are overloaded.")
    print("arrival      " + "  ".join(f"rho={r}" for r in rhos))
    for label, h, soj in arrivals:
        print(f"{label:<12} " + "  ".join(f"{2*r*h/(1+h):8.3f}" for r in rhos))

    jobs = make_jobs(binary, raw_dir, arrivals, rhos, ks, phis, reps, simtime, warm)
    print(f"\n{len(jobs)} configurations x {reps} replications; {args.workers} workers")
    counts = {"completed": 0, "skipped": 0, "failed": 0}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            status, filename, detail = future.result()
            counts[status] += 1
            if status == "failed":
                failures.append((filename, detail))
                print(f"\nFAILED: {filename}\n{detail}\n")

    # Merge only this run's expected filenames, never every arbitrary CSV in raw/.
    expected = [job[1] for job in jobs]
    present = [p for p in expected if p.is_file()]
    if present:
        labels = {(float(h), float(soj)): label for label, h, soj in arrivals}
        frames = []
        for path in present:
            frame = pd.read_csv(path)
            if not {"h", "soj"}.issubset(frame.columns):
                raise RuntimeError(f"{path} is missing h and/or soj columns")
            frame.insert(0, "arrival", [
                labels.get((float(h), float(s)), f"h{h}")
                for h, s in zip(frame["h"], frame["soj"])
            ])
            frames.append(frame)
        merged = pd.concat(frames, ignore_index=True)
        merged.to_csv(out_dir / "all.csv", index=False)
        print(f"Merged {len(present)} files, {len(merged)} rows -> {out_dir / 'all.csv'}")
        if not failures and len(present) == len(jobs):
            expected_rows = len(jobs) * reps
            if len(merged) != expected_rows:
                print(f"WARNING: expected {expected_rows} rows but found {len(merged)}")
    else:
        print("No completed configuration files found.")

    print(f"Completed: {counts['completed']}; resumed/skipped: {counts['skipped']}; failed: {counts['failed']}")
    if failures:
        print("Fix the reported failures and rerun. Completed configuration files are retained.")
        sys.exit(1)


if __name__ == "__main__":
    main()
