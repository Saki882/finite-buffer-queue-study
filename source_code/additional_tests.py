
#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS = [
    "pctLostInRunsGE2",
    "meanRunLen",
    "recoveryMean",
]

GROUPS = ["arrival", "rho", "K"]


def welch_comparison(x, y):
    """Compare two independent samples using Welch's t-test."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]

    n1, n2 = len(x), len(y)

    result = {
        "n_phi0": n1,
        "n_phi06": n2,
        "mean_phi0": np.mean(x) if n1 else np.nan,
        "mean_phi06": np.mean(y) if n2 else np.nan,
    }

    if n1 < 2 or n2 < 2:
        result.update({
            "difference": np.nan,
            "CI95_low": np.nan,
            "CI95_high": np.nan,
            "t_statistic": np.nan,
            "df_welch": np.nan,
            "p_value": np.nan,
            "hedges_g": np.nan,
            "status": "insufficient valid observations",
        })
        return result

    m1, m2 = np.mean(x), np.mean(y)
    v1, v2 = np.var(x, ddof=1), np.var(y, ddof=1)

    difference = m2 - m1

    se2 = v1 / n1 + v2 / n2
    se = np.sqrt(se2)

    denominator = (
        (v1 / n1) ** 2 / (n1 - 1)
        + (v2 / n2) ** 2 / (n2 - 1)
    )

    df = se2 ** 2 / denominator if denominator > 0 else np.nan

    if se == 0:
        if difference == 0:
            t_stat = 0.0
            p_value = 1.0
        else:
            t_stat = np.sign(difference) * np.inf
            p_value = 0.0

        ci_low = difference
        ci_high = difference

    else:
        t_stat = difference / se
        p_value = 2 * stats.t.sf(abs(t_stat), df)

        critical = stats.t.ppf(0.975, df)
        ci_low = difference - critical * se
        ci_high = difference + critical * se

    # Hedges' g using pooled SD and small-sample correction.
    pooled_df = n1 + n2 - 2

    pooled_var = (
        ((n1 - 1) * v1 + (n2 - 1) * v2)
        / pooled_df
    )

    if pooled_var > 0:
        pooled_sd = np.sqrt(pooled_var)
        correction = 1 - 3 / (4 * pooled_df - 1)
        hedges_g = correction * difference / pooled_sd
    else:
        hedges_g = np.nan

    result.update({
        "difference": difference,
        "CI95_low": ci_low,
        "CI95_high": ci_high,
        "t_statistic": t_stat,
        "df_welch": df,
        "p_value": p_value,
        "hedges_g": hedges_g,
        "status": "tested",
    })

    return result


def holm_adjust(p_values):
    """Holm-Bonferroni adjusted p-values, preserving NaNs."""

    p = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(p), np.nan)

    valid = np.flatnonzero(np.isfinite(p))

    if len(valid) == 0:
        return adjusted

    order = valid[np.argsort(p[valid])]
    m = len(order)

    running_max = 0.0

    for rank, index in enumerate(order):
        candidate = (m - rank) * p[index]
        running_max = max(running_max, candidate)
        adjusted[index] = min(running_max, 1.0)

    return adjusted


def run_tests(d, out_dir):
    rows = []

    for metric in METRICS:
        for (arrival, rho, K), subset in d.groupby(GROUPS):

            baseline = subset[
                np.isclose(subset["phi"], 0.0)
            ].copy()

            treatment = subset[
                np.isclose(subset["phi"], 0.6)
            ].copy()

            # Loss-run metrics are conditional on at least
            # one packet loss in the replication.
            if metric in ["pctLostInRunsGE2", "meanRunLen"]:
                baseline = baseline[baseline["lost"] > 0]
                treatment = treatment[treatment["lost"] > 0]

            # Recovery uses only completed, valid observations.
            baseline_values = baseline[metric].dropna()
            treatment_values = treatment[metric].dropna()

            result = welch_comparison(
                baseline_values,
                treatment_values,
            )

            rows.append({
                "arrival": arrival,
                "rho": rho,
                "K": K,
                "metric": metric,
                **result,
            })

    results = pd.DataFrame(rows)

    # Apply Holm correction separately to each metric family.
    results["p_holm"] = np.nan

    for metric in METRICS:
        mask = (
            (results["metric"] == metric)
            & results["p_value"].notna()
        )

        results.loc[mask, "p_holm"] = holm_adjust(
            results.loc[mask, "p_value"].to_numpy()
        )

    results["significant_holm"] = (
        results["p_holm"] < 0.05
    )

    results.to_csv(
        out_dir / "welch_additional_comparisons.csv",
        index=False,
    )

    print("\nAdditional Welch comparisons")
    print(results.groupby("metric").agg(
        comparisons=("metric", "size"),
        tested=("p_value", "count"),
        significant=("significant_holm", "sum"),
    ).to_string())

    print("\nSaved: welch_additional_comparisons.csv")


def max_run_descriptive(d, out_dir):
    """Descriptive statistics for maximum loss-run length."""

    result = (
        d.groupby(GROUPS + ["phi"])["maxRunLen"]
        .agg(
            n="count",
            mean="mean",
            std="std",
            median="median",
            p95=lambda x: x.quantile(0.95),
            minimum="min",
            maximum="max",
        )
        .reset_index()
    )

    result.to_csv(
        out_dir / "max_run_length_descriptive.csv",
        index=False,
    )

    print("Saved: max_run_length_descriptive.csv")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        default="ref_results_v2_repaired/all.csv",
    )

    parser.add_argument(
        "--out",
        default="ref_results_v2_repaired/analysis",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(csv_path)

    required = set(
        GROUPS + [
            "phi", "lost",
            "pctLostInRunsGE2",
            "meanRunLen",
            "maxRunLen",
            "recoveryMean",
        ]
    )

    missing = required - set(d.columns)

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(sorted(missing))
        )

    print("Input rows:", len(d))
    print("Running additional statistical comparisons...")

    run_tests(d, out_dir)
    max_run_descriptive(d, out_dir)

    print("\nAdditional analysis completed.")


if __name__ == "__main__":
    main()
