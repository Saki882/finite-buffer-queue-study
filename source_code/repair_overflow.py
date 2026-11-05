import csv
import json
import math
import shutil
import subprocess
from pathlib import Path

import pandas as pd

BASE = Path("ref_results_v2")
OUT = Path("ref_results_v2_repaired")
BINARY = Path("ref_sim_audit_fixed").resolve()

ARRIVALS = {
    "Poisson": (1.0, 0.05),
    "MMPP-h2": (2.0, 0.02),
    "MMPP-h3": (3.0, 0.02),
    "MMPP-h5": (5.0, 0.02),
}

def tag(x):
    return str(x).replace("-", "m").replace(".", "p")

if not BASE.joinpath("all.csv").is_file():
    raise SystemExit("Original all.csv not found.")

if not BINARY.is_file():
    raise SystemExit("Corrected simulator executable not found.")

if OUT.exists():
    raise SystemExit(
        f"{OUT} already exists. Rename or inspect it before continuing."
    )

manifest = json.loads((BASE / "run_manifest.json").read_text())
simtime = manifest["simTime"]
warm = manifest["warm"]

original = pd.read_csv(BASE / "all.csv")
affected = original.index[
    original["overflowCensored"] == 1
].tolist()

print(f"Original rows: {len(original)}")
print(f"Affected rows: {len(affected)}")

if not affected:
    raise SystemExit("No affected rows found.")

# Create a separate copy. The original dataset is never modified.
OUT.mkdir()
shutil.copy2(BASE / "run_manifest.json", OUT)
shutil.copytree(BASE / "raw", OUT / "raw")
run_dir = OUT / "recovery_runs"
run_dir.mkdir()

raw_cache = {}
log = []

for count, idx in enumerate(affected, 1):
    row = original.loc[idx]

    arrival = row["arrival"]
    h, soj = ARRIVALS[arrival]

    filename = (
        f"{arrival}_rho{tag(float(row['rho']))}"
        f"_K{int(row['K'])}_phi{tag(float(row['phi']))}.csv"
    )

    raw_path = OUT / "raw" / filename

    if filename not in raw_cache:
        raw_cache[filename] = pd.read_csv(raw_path)

    raw = raw_cache[filename]

    matches = raw.index[raw["seed"] == int(row["seed"])].tolist()
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one matching seed in {filename}; found {len(matches)}"
        )

    raw_idx = matches[0]

    if int(raw.loc[raw_idx, "overflowCensored"]) != 1:
        raise RuntimeError(
            f"Raw file does not match affected row: {filename}, "
            f"seed {int(row['seed'])}"
        )

    seed = int(row["seed"])
    output_csv = run_dir / f"run_{idx}_seed_{seed}.csv"

    command = [
        str(BINARY),
        "--mode=corr",
        f"--rho={row['rho']}",
        f"--K={int(row['K'])}",
        f"--h={h}",
        f"--soj={soj}",
        f"--phi={row['phi']}",
        f"--theta={row['theta']}",
        f"--simTime={simtime}",
        f"--warm={warm}",
        "--reps=1",
        f"--seed={seed}",
        f"--csv={output_csv}",
    ]

    result = subprocess.run(
        command, capture_output=True, text=True
    )

    if result.returncode != 0 or not output_csv.is_file():
        raise RuntimeError(
            f"Simulation failed at row {idx}, seed {seed}:\n"
            f"{result.stderr[-2000:]}"
        )

    rerun = pd.read_csv(output_csv)

    if len(rerun) != 1:
        raise RuntimeError(f"Unexpected output rows at seed {seed}")

    new = rerun.iloc[0]

    # Validate every other metric against the original result.
    for col in original.columns:
        if col in ("arrival", "overflowCensoredDur"):
            continue

        if col not in rerun.columns:
            raise RuntimeError(f"Missing column in rerun: {col}")

        old_value = row[col]
        new_value = new[col]

        if pd.isna(old_value) and pd.isna(new_value):
            continue

        try:
            equal = math.isclose(
                float(old_value),
                float(new_value),
                rel_tol=1e-10,
                abs_tol=1e-12,
            )
        except (TypeError, ValueError):
            equal = str(old_value) == str(new_value)

        if not equal:
            raise RuntimeError(
                f"Metric mismatch at row {idx}, seed {seed}, "
                f"column {col}: old={old_value}, new={new_value}"
            )

    duration = float(new["overflowCensoredDur"])

    if not math.isfinite(duration) or duration <= 0:
        raise RuntimeError(
            f"Invalid corrected duration at seed {seed}: {duration}"
        )

    # Update only the censored duration.
    original.at[idx, "overflowCensoredDur"] = duration
    raw.at[raw_idx, "overflowCensoredDur"] = duration

    log.append({
        "original_row": idx,
        "arrival": arrival,
        "rho": row["rho"],
        "K": int(row["K"]),
        "phi": row["phi"],
        "seed": seed,
        "corrected_duration": duration,
    })

    print(
        f"[{count}/{len(affected)}] "
        f"{arrival}, rho={row['rho']}, K={int(row['K'])}, "
        f"phi={row['phi']}, seed={seed}: "
        f"{duration:.9f} s"
    )

# Save corrected raw files and merged dataset.
for filename, frame in raw_cache.items():
    frame.to_csv(OUT / "raw" / filename, index=False)

original.to_csv(OUT / "all.csv", index=False)

pd.DataFrame(log).to_csv(OUT / "repair_log.csv", index=False)

# Final checks.
check = pd.read_csv(OUT / "all.csv")

remaining = int((check["overflowCensored"] == 1).sum())
missing = int(
    check.loc[
        check["overflowCensored"] == 1,
        "overflowCensoredDur"
    ].isna().sum()
)

if remaining != len(affected) or missing != 0:
    raise RuntimeError(
        f"Final audit failed: censored={remaining}, missing={missing}"
    )

print("\nRECOVERY SUCCESSFUL")
print(f"Repaired runs: {len(log)}")
print(f"Remaining missing censored durations: {missing}")
print(f"Repaired dataset: {OUT}/all.csv")
print(f"Original dataset unchanged: {BASE}/all.csv")
