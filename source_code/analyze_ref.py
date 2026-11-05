#!/usr/bin/env python3
"""Analyze replication-level CSV output from ref_sim_v3."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = [
    "lossRatio", "burstRatio", "meanRunLen", "maxRunLen", "pctRunsGE2",
    "pctLostInRunsGE2", "meanQ", "fullFrac", "recoveryMean", "recoveryP95",
    "delayMean", "delayP95", "acfInter", "acfService",
]
LOW_COUNT = 1000  # Practical warning threshold, not a universal statistical cutoff.


def ci95(values):
    """95% t-CI half-width across independent simulation replications."""
    x = pd.Series(values).dropna().astype(float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def ratio_ci_log_delta(treatment, baseline):
    """Delta-method CI for ratio of independent sample means, on log scale."""
    x = pd.Series(treatment).dropna().astype(float).to_numpy()
    y = pd.Series(baseline).dropna().astype(float).to_numpy()
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan
    mx, my = x.mean(), y.mean()
    if mx <= 0 or my <= 0:
        return np.nan, np.nan, np.nan
    vx = x.var(ddof=1) / len(x)
    vy = y.var(ddof=1) / len(y)
    a, b = vx / mx**2, vy / my**2
    se2 = a + b
    ratio = mx / my
    if se2 == 0:
        return ratio, ratio, ratio
    df_denom = (a*a / (len(x)-1) if a > 0 else 0) + (b*b / (len(y)-1) if b > 0 else 0)
    df = se2**2 / df_denom if df_denom > 0 else min(len(x), len(y)) - 1
    critical = stats.t.ppf(0.975, df)
    se = np.sqrt(se2)
    return ratio, np.exp(np.log(ratio) - critical*se), np.exp(np.log(ratio) + critical*se)


def load_data(path):
    d = pd.read_csv(path)
    # Backward-compatible column aliases from earlier simulator versions.
    d = d.rename(columns={
        "recMean": "recoveryMean", "recMedian": "recoveryMedian",
        "recP95": "recoveryP95", "acfSvc": "acfService", "little": "littleDiagnostic",
    })
    required = {"rho", "K", "phi", "h", "soj", "lost"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError("CSV is missing required columns: " + ", ".join(sorted(missing)))
    if "arrival" not in d.columns:
        d["arrival"] = np.where(
            d["h"].astype(float) == 1.0, "Poisson", "MMPP-h" + d["h"].astype(str)
        )
    for col in ["rho", "K", "phi", "h", "soj", "lost"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if d[["rho", "K", "phi", "h", "soj", "lost"]].isna().any().any():
        raise ValueError("Required numeric columns contain missing/invalid values")
    return d


def make_summary(d, out_dir):
    metrics = [m for m in METRICS if m in d.columns]
    keys = ["arrival", "rho", "K", "phi"]
    grouped = d.groupby(keys, dropna=False)
    result = grouped[metrics].mean().add_suffix("_mean")
    result = result.join(grouped[metrics].agg(ci95).add_suffix("_ci95"))
    result["meanLost"] = grouped["lost"].mean()
    result["minLost"] = grouped["lost"].min()
    result["maxLost"] = grouped["lost"].max()
    result["lowCount"] = result["minLost"] < LOW_COUNT
    result["n_runs"] = grouped.size()
    result.to_csv(out_dir / "summary.csv")
    print(
        f"{len(result)} configurations; {int(result.lowCount.sum())} flagged: "
        f"at least one replication had fewer than {LOW_COUNT} lost packets."
    )
    return result


def make_phi_effects(d, out_dir):
    rows = []
    for (arrival, rho, K), subset in d.groupby(["arrival", "rho", "K"]):
        baseline = subset.loc[np.isclose(subset.phi, 0.0), "lossRatio"]
        if len(baseline) < 2 or baseline.mean() <= 0:
            continue
        for phi, group in subset.groupby("phi"):
            if np.isclose(float(phi), 0.0):
                continue
            treatment = group["lossRatio"]
            if len(treatment) < 2 or treatment.mean() <= 0:
                continue
            ratio, lower, upper = ratio_ci_log_delta(treatment, baseline)
            rows.append({
                "arrival": arrival, "rho": rho, "K": K, "phi": phi,
                "loss_ratio_of_means": ratio,
                "ratio_ci95_lower": lower, "ratio_ci95_upper": upper,
                "absolute_loss_increase": treatment.mean() - baseline.mean(),
                "baseline_loss_mean": baseline.mean(),
                "correlated_loss_mean": treatment.mean(),
                "baseline_n": len(baseline), "correlated_n": len(treatment),
            })
    pd.DataFrame(rows).to_csv(out_dir / "effect_of_phi.csv", index=False)


def make_figures(d, out_dir):
    plots = [m for m in ["lossRatio", "burstRatio", "pctLostInRunsGE2", "recoveryMean", "delayP95"] if m in d.columns]
    for metric in plots:
        for rho in sorted(d.rho.unique()):
            buffers = sorted(d.K.unique())
            fig, axes = plt.subplots(1, len(buffers), figsize=(4.4*len(buffers), 3.6), squeeze=False)
            for ax, K in zip(axes[0], buffers):
                subset = d[(d.rho == rho) & (d.K == K)]
                for arrival, group in subset.groupby("arrival"):
                    by_phi = group.groupby("phi")[metric]
                    means = by_phi.mean()
                    errors = by_phi.apply(ci95).reindex(means.index).fillna(0)
                    if metric == "lossRatio":
                        positive = means > 0
                        means, errors = means[positive], errors[positive]
                    ax.errorbar(means.index, means.values, yerr=errors.values,
                                marker="o", capsize=3, label=arrival)
                if metric == "lossRatio":
                    ax.set_yscale("log")
                ax.set_title(f"rho={rho}, K={K}")
                ax.set_xlabel("Service correlation (phi)")
                ax.set_ylabel(metric)
                ax.grid(True, alpha=0.25)
            axes[0][0].legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(out_dir / f"fig_{metric}_rho{rho}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)


def make_anova(d, out_dir, selected_rho=None, selected_K=None, selected_arrival=None):
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
    except ImportError:
        print("statsmodels is not installed; ANOVA skipped")
        return

    mmpp_levels = sorted(x for x in d.arrival.unique() if x != "Poisson")
    if not mmpp_levels:
        print("No MMPP rows; ANOVA skipped")
        return
    mmpp = selected_arrival or mmpp_levels[0]
    if mmpp not in mmpp_levels:
        raise ValueError(f"--arrival must be one of: {mmpp_levels}")
    rhos = [selected_rho] if selected_rho is not None else sorted(d.rho.unique())
    buffers = [selected_K] if selected_K is not None else sorted(d.K.unique())
    positive_phi = sorted(float(x) for x in d.phi.unique() if float(x) > 0)
    if not positive_phi:
        print("No positive phi level; ANOVA skipped")
        return
    high_phi = 0.6 if 0.6 in positive_phi else positive_phi[-1]

    for rho in rhos:
        for K in buffers:
            s = d[
                np.isclose(d.rho, rho) & (d.K == K)
                & d.arrival.isin(["Poisson", mmpp])
                & (np.isclose(d.phi, 0.0) | np.isclose(d.phi, high_phi))
            ].copy()
            if s.arrival.nunique() < 2 or s.phi.nunique() < 2:
                continue
            s["arr_factor"] = np.where(s.arrival == "Poisson", "Poisson", "MMPP")
            s["svc_factor"] = np.where(np.isclose(s.phi, 0.0), "independent", "correlated")
            lines = [
                f"rho={rho}, K={K}", f"Arrival: Poisson vs {mmpp}",
                f"Service: phi=0 vs phi={high_phi}", f"Rows: {len(s)}", "",
                "Focused 2x2 factorial analysis only; this is not a test of the entire grid.",
                "Interpret p-values with effect sizes, confidence intervals and diagnostics.", "",
            ]
            for metric in ["lossRatio", "burstRatio", "recoveryMean"]:
                if metric not in s.columns:
                    continue
                sample = s.dropna(subset=[metric]).copy()
                if len(sample) < 8 or sample[metric].nunique() < 2:
                    lines += [f"[{metric}] skipped: insufficient data/variation", ""]
                    continue
                for scale in ["raw", "log10"]:
                    work = sample.copy()
                    if scale == "log10":
                        if (work[metric] <= 0).any():
                            lines += [f"[{metric}, log10] skipped: nonpositive values", ""]
                            continue
                        work[metric] = np.log10(work[metric])
                    model = ols(f"{metric} ~ C(arr_factor) * C(svc_factor)", data=work).fit()
                    table = sm.stats.anova_lm(model, typ=2)
                    table["eta_sq"] = table["sum_sq"] / table["sum_sq"].sum()
                    lines += [f"[{metric}, {scale}; Type-II ANOVA]", table.round(6).to_string(), ""]
                    shapiro_p = stats.shapiro(model.resid).pvalue if len(model.resid) >= 3 else np.nan
                    groups = [g[metric].to_numpy() for _, g in work.groupby(["arr_factor", "svc_factor"])]
                    try:
                        levene_p = stats.levene(*groups, center="median").pvalue
                    except Exception:
                        levene_p = np.nan
                    lines += [
                        f"Residual Shapiro-Wilk p-value: {shapiro_p:.6g}",
                        f"Levene p-value (median-centered): {levene_p:.6g}",
                        f"Residual degrees of freedom: {int(model.df_resid)}", "",
                    ]
            (out_dir / f"anova_rho{rho}_K{K}.txt").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze finite-buffer simulator results")
    parser.add_argument("--csv", default="ref_results/all.csv")
    parser.add_argument("--rho", type=float, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--arrival", default=None, help="MMPP label for focused ANOVA, e.g. MMPP-h3")
    args = parser.parse_args()
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        parser.error(f"Input CSV not found: {csv_path}")
    d = load_data(csv_path)
    out_dir = csv_path.parent / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    make_summary(d, out_dir)
    make_phi_effects(d, out_dir)
    make_figures(d, out_dir)
    make_anova(d, out_dir, args.rho, args.K, args.arrival)
    print(f"Analysis written to {out_dir}")


if __name__ == "__main__":
    main()
