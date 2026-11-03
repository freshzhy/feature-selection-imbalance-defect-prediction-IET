"""
论文图表生成脚本 v2
- 使用 v2 实验结果（含 CostSensitive + MCC + AUC-PR）
- Figure 1/2 改为箱线图+散点（PeerJ 推荐）
- 统一输出为 fig1.png–fig7.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

PALETTE_IMB = {
    "None":           "#95a5a6",
    "SMOTE":          "#3498db",
    "ADASYN":         "#e74c3c",
    "CostSensitive":  "#2ecc71",
}
CLF_COLORS = {"LR": "#2ecc71", "SVM": "#3498db", "RF": "#e67e22", "XGB": "#9b59b6"}
CLF_ORDER  = ["LR", "SVM", "RF", "XGB"]
FS_ORDER   = ["None", "IG", "CFS"]
FS_LABELS  = {"None": "No FS", "IG": "Information Gain", "CFS": "ANOVA F-test"}
IMB_ORDER  = ["None", "SMOTE", "ADASYN", "CostSensitive"]
IMB_LABELS = {"None": "No Handling", "SMOTE": "SMOTE", "ADASYN": "ADASYN",
              "CostSensitive": "Cost-Sensitive"}

CSV = os.path.join(os.path.dirname(__file__), "results", "all_results_v2.csv")
df  = pd.read_csv(CSV)
df["FS"]        = df["FS"].fillna("None")
df["Imbalance"] = df["Imbalance"].fillna("None")
DATASETS = sorted(df["Dataset"].unique())

def save(fig, name):
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  -> {name}")


# ════════════════════════════════════════════════════════
# Figure 1 — Feature Selection effect (boxplot + strip)
# ════════════════════════════════════════════════════════
def fig1():
    per_ds = df.groupby(["Dataset", "FS"])["F1"].mean().reset_index()
    per_ds["FS_label"] = per_ds["FS"].map(FS_LABELS)
    order = [FS_LABELS[f] for f in FS_ORDER]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(x="FS_label", y="F1", data=per_ds, order=order,
                palette="Set2", width=0.5, ax=ax,
                boxprops=dict(alpha=0.7), fliersize=0)
    sns.stripplot(x="FS_label", y="F1", data=per_ds, order=order,
                  color="black", size=5, alpha=0.6, jitter=0.1, ax=ax)

    ax.set_xlabel("Feature Selection Method")
    ax.set_ylabel("Per-Dataset Mean F1-score")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.65)

    medians = per_ds.groupby("FS_label")["F1"].median()
    for i, label in enumerate(order):
        ax.text(i, medians[label] + 0.02, f"{medians[label]:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    return fig


# ════════════════════════════════════════════════════════
# Figure 2 — Classifier × Imbalance Handling (boxplot + strip)
# ════════════════════════════════════════════════════════
def fig2():
    per_ds = df.groupby(["Dataset", "Imbalance"])["F1"].mean().reset_index()
    per_ds["Imb_label"] = per_ds["Imbalance"].map(IMB_LABELS)
    order = [IMB_LABELS[m] for m in IMB_ORDER]

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = [PALETTE_IMB[m] for m in IMB_ORDER]
    sns.boxplot(x="Imb_label", y="F1", data=per_ds, order=order,
                palette=palette, width=0.5, ax=ax,
                boxprops=dict(alpha=0.7), fliersize=0)
    sns.stripplot(x="Imb_label", y="F1", data=per_ds, order=order,
                  color="black", size=5, alpha=0.6, jitter=0.1, ax=ax)

    ax.set_xlabel("Imbalance Handling Method")
    ax.set_ylabel("Per-Dataset Mean F1-score")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 0.65)

    medians = per_ds.groupby("Imb_label")["F1"].median()
    for i, label in enumerate(order):
        ax.text(i, medians[label] + 0.02, f"{medians[label]:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    return fig


# ════════════════════════════════════════════════════════
# Figure 3 — Heatmap: Dataset × Classifier (best F1)
# ════════════════════════════════════════════════════════
def fig3():
    pivot = df.groupby(["Dataset", "Classifier"])["F1"].max().reset_index()
    pivot = pivot.pivot(index="Dataset", columns="Classifier", values="F1")
    pivot = pivot[CLF_ORDER]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Best F1-score"},
                ax=ax, vmin=0.1, vmax=0.75)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Dataset")
    return fig


# ════════════════════════════════════════════════════════
# Figure 4 — Heatmap: FS × Imbalance mean F1
# ════════════════════════════════════════════════════════
def fig4():
    pivot = df.groupby(["FS", "Imbalance"])["F1"].mean().reset_index()
    pivot = pivot.pivot(index="FS", columns="Imbalance", values="F1")
    pivot = pivot.loc[FS_ORDER, IMB_ORDER]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="Blues",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Average F1-score"},
                ax=ax, vmin=0.15, vmax=0.50)
    ax.set_xlabel("Imbalance Handling Method")
    ax.set_ylabel("Feature Selection Method")
    ax.set_xticklabels([IMB_LABELS[m] for m in IMB_ORDER], rotation=15, ha="right")
    ax.set_yticklabels([FS_LABELS[f] for f in FS_ORDER], rotation=0)
    return fig


# ════════════════════════════════════════════════════════
# Figure 5 — Per-dataset F1 by imbalance method
# ════════════════════════════════════════════════════════
def fig5():
    per_ds = df.groupby(["Dataset", "Imbalance"])["F1"].mean().reset_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(DATASETS))
    width = 0.2
    offsets = [-1.5, -0.5, 0.5, 1.5]

    for i, imb in enumerate(IMB_ORDER):
        vals = [per_ds.loc[(per_ds.Dataset == d) & (per_ds.Imbalance == imb), "F1"].values[0]
                for d in DATASETS]
        ax.bar(x + offsets[i] * width, vals, width,
               label=IMB_LABELS[imb], color=PALETTE_IMB[imb],
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=30, ha="right")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Average F1-score")
    ax.legend(title="Imbalance Handling")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    return fig


# ════════════════════════════════════════════════════════
# Figure 6 — AUC heatmap: Dataset × Classifier
# ════════════════════════════════════════════════════════
def fig6():
    pivot = df.groupby(["Dataset", "Classifier"])["AUC"].max().reset_index()
    pivot = pivot.pivot(index="Dataset", columns="Classifier", values="AUC")
    pivot = pivot[CLF_ORDER]

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="PuBu",
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "Best AUC-ROC"},
                ax=ax, vmin=0.5, vmax=1.0)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("Dataset")
    return fig


# ════════════════════════════════════════════════════════
# Figure 7 — Radar: representative strategy comparisons
# ════════════════════════════════════════════════════════
def fig7():
    metrics = ["F1", "AUC", "Precision", "Recall", "MCC"]
    combos = {
        "No FS + No Handling":  df[(df.FS == "None") & (df.Imbalance == "None")],
        "No FS + SMOTE":        df[(df.FS == "None") & (df.Imbalance == "SMOTE")],
        "No FS + CostSensitive":df[(df.FS == "None") & (df.Imbalance == "CostSensitive")],
        "IG + SMOTE":           df[(df.FS == "IG")   & (df.Imbalance == "SMOTE")],
        "AF + ADASYN":          df[(df.FS == "CFS")  & (df.Imbalance == "ADASYN")],
    }

    values = {k: [v[m].mean() for m in metrics] for k, v in combos.items()}

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    colors = ["#95a5a6", "#3498db", "#2ecc71", "#e67e22", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    for (label, vals), color in zip(values.items(), colors):
        v = vals + vals[:1]
        ax.plot(angles, v, "o-", linewidth=2, color=color, label=label)
        ax.fill(angles, v, alpha=0.07, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"], size=8)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9)
    return fig


if __name__ == "__main__":
    print("生成 v2 论文图表...")
    save(fig1(), "fig1.png")
    save(fig2(), "fig2.png")
    save(fig3(), "fig3.png")
    save(fig4(), "fig4.png")
    save(fig5(), "fig5.png")
    save(fig6(), "fig6.png")
    save(fig7(), "fig7.png")
    print(f"\n全部图表已保存至: {FIG_DIR}/")
    print("  fig1 — Feature Selection effect (boxplot)")
    print("  fig2 — Imbalance Handling effect (boxplot)")
    print("  fig3 — Heatmap: Dataset × Classifier (F1)")
    print("  fig4 — Heatmap: FS × Imbalance (F1)")
    print("  fig5 — Per-dataset F1 by imbalance method")
    print("  fig6 — Heatmap: Dataset × Classifier (AUC)")
    print("  fig7 — Radar: strategy comparison")
