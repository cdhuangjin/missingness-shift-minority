"""Figure generation for Gate A."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_MCAR_RATE = {"iid_mcar": 0.05, "mild_mcar": 0.10, "severe_mcar": 0.30}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (7, 5),
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "font.size": 10,
        }
    )


def fig_f(prefix: str, name: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir / f"{prefix}_{name}.png"


def figure1_minority_recall(raw: pd.DataFrame, outdir: Path) -> Path:
    _style()
    p0 = raw[raw["pipeline"] == "P0"]
    p0 = p0[p0["environment"].isin(_MCAR_RATE)]
    fig, ax = plt.subplots()
    for (ds, model), grp in p0.groupby(["dataset", "model"]):
        rates = grp["environment"].map(_MCAR_RATE)
        ys = grp["minority_recall"]
        order = np.argsort(rates.to_numpy())
        ax.plot(
            rates.to_numpy()[order],
            ys.to_numpy()[order],
            marker="o",
            label=f"{ds} | {model}",
        )
    ax.set_xlabel("MCAR missing rate")
    ax.set_ylabel("Minority recall")
    ax.set_title("Figure 1: Minority recall vs missing rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = fig_f("fig1", "minority_recall_vs_rate", outdir)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure2_majority_recall(raw: pd.DataFrame, outdir: Path) -> Path:
    _style()
    p0 = raw[raw["pipeline"] == "P0"]
    p0 = p0[p0["environment"].isin(_MCAR_RATE)]
    fig, ax = plt.subplots()
    for (ds, model), grp in p0.groupby(["dataset", "model"]):
        rates = grp["environment"].map(_MCAR_RATE)
        ys = grp["majority_recall"]
        order = np.argsort(rates.to_numpy())
        ax.plot(
            rates.to_numpy()[order],
            ys.to_numpy()[order],
            marker="s",
            label=f"{ds} | {model}",
        )
    ax.set_xlabel("MCAR missing rate")
    ax.set_ylabel("Majority recall")
    ax.set_title("Figure 2: Majority recall vs missing rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = fig_f("fig2", "majority_recall_vs_rate", outdir)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure3_slope_comparison(slopes: pd.DataFrame, outdir: Path) -> Path:
    _style()
    fig, ax = plt.subplots()
    x = np.arange(len(slopes))
    width = 0.35
    ax.bar(x - width / 2, slopes["minority_slope"], width, label="minority slope")
    ax.bar(x + width / 2, slopes["majority_slope"], width, label="majority slope")
    labels = [
        f"{d}\n{m}" for d, m in zip(slopes["dataset"], slopes["model"])
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("Recall sensitivity slope (per missing-rate unit)")
    ax.set_title("Figure 3: Minority vs majority degradation slope")
    ax.axhline(0, color="k", lw=0.7)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    path = fig_f("fig3", "degradation_slope", outdir)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure4_cc_vs_comparable(raw: pd.DataFrame, outdir: Path) -> Path:
    _style()
    p0 = raw[raw["pipeline"] == "P0"]
    sub = p0[p0["environment"].isin(["mcar_comparable", "class_conditional"])]
    piv = sub.pivot_table(
        index=["dataset", "model"],
        columns="environment",
        values=["minority_recall", "minority_f1"],
        aggfunc="mean",
    )
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for j, metric in enumerate(["minority_recall", "minority_f1"]):
        a = piv[(metric, "mcar_comparable")]
        b = piv[(metric, "class_conditional")]
        labels = [f"{i[0][:8]}\n{i[1][:8]}" for i in piv.index]
        x = np.arange(len(labels))
        ax[j].bar(x - 0.18, a, 0.34, label="MCAR comparable rate")
        ax[j].bar(x + 0.18, b, 0.34, label="class-conditional")
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax[j].set_title(f"{metric} (mean over seeds)")
        ax[j].grid(alpha=0.3, axis="y")
        ax[j].legend(fontsize=7)
    fig.tight_layout()
    path = fig_f("fig4", "cc_vs_comparable", outdir)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure5_hidden_failure(raw: pd.DataFrame, outdir: Path) -> Path:
    _style()
    p0 = raw[raw["pipeline"] == "P0"]
    recs = []
    for (ds, model, seed), grp in p0.groupby(["dataset", "model", "seed"]):
        iid = grp[grp["environment"] == "iid_mcar"]
        if iid.empty:
            continue
        a0 = float(iid["AUROC"].iloc[0])
        r0 = float(iid["minority_recall"].iloc[0])
        shifted = grp[grp["environment"] != "iid_mcar"]
        for _, row in shifted.iterrows():
            recs.append(
                {
                    "dataset": ds,
                    "model": model,
                    "seed": seed,
                    "environment": row["environment"],
                    "auroc_drop": a0 - float(row["AUROC"]),
                    "recall_drop": r0 - float(row["minority_recall"]),
                }
            )
    recs = pd.DataFrame(recs)
    if recs.empty:
        return None
    fig, ax = plt.subplots()
    models = sorted(recs["model"].unique())
    colors = {m: c for m, c in zip(models, plt.cm.tab10.colors[: len(models)])}
    for m in models:
        sub = recs[recs["model"] == m]
        ax.scatter(
            sub["auroc_drop"],
            sub["recall_drop"],
            color=colors[m],
            s=24,
            alpha=0.8,
            label=m,
        )
    ax.axvspan(-1.0, 0.03, color="green", alpha=0.12)
    ax.axhspan(0.10, 1.0, color="green", alpha=0.12)
    ax.axvline(0.03, color="green", ls="--", lw=1)
    ax.axhline(0.10, color="green", ls="--", lw=1)
    ax.set_xlabel("AUROC drop")
    ax.set_ylabel("Minority recall drop")
    ax.set_title("Figure 5: Hidden minority failure region")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = fig_f("fig5", "hidden_failure", outdir)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure6_mvg(vuln: pd.DataFrame, outdir: Path) -> Path:
    _style()
    p0 = vuln[vuln["pipeline"] == "P0"]
    piv = p0.pivot_table(
        index=["dataset", "model"],
        columns="environment",
        values="MVG_recall",
        aggfunc="mean",
    ).dropna(how="all")
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-0.3, vmax=0.3)
    ax.set_xticks(np.arange(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=20, ha="right", fontsize=7)
    ax.set_yticks(np.arange(piv.shape[0]))
    ax.set_yticklabels(
        [f"{i[0]}\n{i[1]}" for i in piv.index], fontsize=7
    )
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iloc[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="MVG_recall")
    ax.set_title("Figure 6: MVG by dataset / model / environment")
    fig.tight_layout()
    path = fig_f("fig6", "mvg_map", outdir)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_all_figures(
    raw: pd.DataFrame,
    vuln: pd.DataFrame,
    slopes: pd.DataFrame,
    outdir: Path,
) -> list[Path]:
    paths = []
    p1 = figure1_minority_recall(raw, outdir)
    paths.append(p1)
    p2 = figure2_majority_recall(raw, outdir)
    paths.append(p2)
    if not slopes.empty:
        paths.append(figure3_slope_comparison(slopes, outdir))
    paths.append(figure4_cc_vs_comparable(raw, outdir))
    p5 = figure5_hidden_failure(raw, outdir)
    if p5:
        paths.append(p5)
    paths.append(figure6_mvg(vuln, outdir))
    return paths
