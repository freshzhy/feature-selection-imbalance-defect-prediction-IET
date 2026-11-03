"""
v3 实验结果统计分析（IET Software 大修用）
==========================================

输入：results_v3/all_folds.csv、results_v3/datasets_summary.csv
输出：results_v3/analysis/ 下的表格（CSV）和图（PNG, 300 dpi）

对应审稿意见
------------
R1-1 不显著≠等价      → 贝叶斯符号秩检验 + ROPE（P(实际等价)）          pairwise_*.csv
R1-2 配对效应量        → matched-pairs rank-biserial r + 均值差 bootstrap CI pairwise_*.csv
R1-4 因子交互          → 区组全因子 ANOVA 方差分解（η²、偏 η²）            anova_*.csv, fig_interaction_*.png
R1-5 效率              → 拟合/预测时间、模型大小、特征数                    efficiency_*.csv
R1-7 AUC-PR 为主       → 所有分析同时给出 MCC / AUC-PR / F1 / AUC
R2-2 机制              → 增益 vs 不平衡比、阈值移动(THR)对照                mechanism_*.csv, fig_mechanism.png
R1-3 最佳配置          → 描述性表（嵌套 CV 版本另见 run_nested_tuning.py）  best_config_descriptive.csv

所有分析分别在 ALL（24 个）、NASA（12 个）、EXTERNAL（AEEEM+JIRA 12 个）三个子集上做，
用于说明结论是否在非 NASA 数据上复现。

运行（项目根目录）：
    pip install statsmodels matplotlib      # 如未安装
    python analyze_v3.py                    # 约 1–2 分钟
"""

import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_v3"
OUT = RES / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

METRICS = ["mcc", "auc_pr", "f1", "auc", "recall", "precision"]
MAIN_METRICS = ["mcc", "auc_pr", "f1", "auc"]
LEVELS = {
    "FS": ["None", "IG", "AF"],
    "IMB": ["None", "SMOTE", "ADASYN", "CS", "THR"],
    "CLF": ["LR", "SVM", "RF", "XGB"],
}
ROPE = 0.01          # 实际等价区间：指标差 ±0.01
ROPE_SENS = 0.02     # 敏感性：±0.02
N_BOOT = 10000
N_BAYES = 50000
RNG = np.random.default_rng(2026)
SUBSETS = {"ALL": None, "NASA": ["NASA"], "EXTERNAL": ["AEEEM", "JIRA"]}
LABEL = {"mcc": "MCC", "auc_pr": "AUC-PR", "f1": "F1-score", "auc": "AUC-ROC",
         "recall": "Recall", "precision": "Precision"}


# ----------------------------------------------------------------------------
# 统计工具
# ----------------------------------------------------------------------------
def rank_biserial_paired(d):
    """matched-pairs rank-biserial correlation（与 Wilcoxon 符号秩检验配套的配对效应量）
    r = (R+ - R-) / (R+ + R-)，范围 [-1, 1]；|r| 约 0.1 小、0.3 中、0.5 大"""
    d = np.asarray(d, float)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    rp, rm = r[d > 0].sum(), r[d < 0].sum()
    return float((rp - rm) / (rp + rm))


def bootstrap_ci(d, n=N_BOOT):
    d = np.asarray(d, float)
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def bayes_signed_rank(d, rope, s=0.5, n_samples=N_BAYES):
    """贝叶斯符号秩检验（Benavoli et al., 2017, JMLR 18:1-36）
    返回 P(A<B), P(实际等价), P(A>B)，其中 d = A - B。"""
    d = np.asarray(d, float)
    z = np.r_[0.0, d]                          # 先验伪观测 z0 = 0
    n = len(z)
    i, j = np.triu_indices(n)                  # 所有 i<=j 的 Walsh 平均
    walsh = (z[i] + z[j]) / 2
    left, rmask, right = walsh < -rope, np.abs(walsh) <= rope, walsh > rope
    alpha = np.r_[s, np.ones(n - 1)]
    w = RNG.dirichlet(alpha, size=n_samples)   # (S, n)
    prod = w[:, i] * w[:, j]                   # (S, pairs)
    pl, pr_, pp = prod[:, left].sum(1), prod[:, rmask].sum(1), prod[:, right].sum(1)
    win = np.argmax(np.c_[pl, pr_, pp], axis=1)
    return [float((win == k).mean()) for k in range(3)]


def pairwise_family(dm, factor, metric):
    """dm: 每个数据集 × 因子水平的均值（已在其它因子上平均）"""
    piv = dm.pivot_table(index="Dataset", columns=factor, values=metric)
    rows = []
    for a, b in itertools.combinations(LEVELS[factor], 2):
        if a not in piv or b not in piv:
            continue
        d = (piv[a] - piv[b]).values
        try:
            W, p = stats.wilcoxon(piv[a], piv[b])
        except ValueError:
            W, p = np.nan, 1.0
        lo, hi = bootstrap_ci(d)
        pl, prope, pr_ = bayes_signed_rank(d, ROPE)
        pl2, prope2, pr2 = bayes_signed_rank(d, ROPE_SENS)
        rows.append({
            "factor": factor, "metric": metric, "A": a, "B": b, "n_datasets": len(d),
            "mean_A": piv[a].mean(), "mean_B": piv[b].mean(),
            "mean_diff(A-B)": d.mean(), "median_diff": np.median(d),
            "CI95_low": lo, "CI95_high": hi,
            "wins_A": int((d > 0).sum()), "ties": int((d == 0).sum()), "wins_B": int((d < 0).sum()),
            "W": W, "p": p, "rank_biserial_r": rank_biserial_paired(d),
            f"P(A<B)_rope{ROPE}": pl, f"P(equiv)_rope{ROPE}": prope, f"P(A>B)_rope{ROPE}": pr_,
            f"P(equiv)_rope{ROPE_SENS}": prope2,
        })
    df = pd.DataFrame(rows)
    if len(df):
        df["p_holm"] = multipletests(df["p"].fillna(1), method="holm")[1]
    return df


def rb_magnitude(r):
    r = abs(r)
    return "negligible" if r < 0.1 else "small" if r < 0.3 else "medium" if r < 0.5 else "large"


# ----------------------------------------------------------------------------
def main():
    folds = pd.read_csv(RES / "all_folds.csv", low_memory=False)
    for c in ["FS", "IMB"]:
        folds[c] = folds[c].fillna("None")
    summ = pd.read_csv(RES / "datasets_summary.csv")

    # 1) 每个 数据集×管道 的均值（10×10 折平均）
    dp = (folds.groupby(["Dataset", "Source", "FS", "IMB", "CLF"])
          [METRICS + ["fit_time", "predict_time", "model_bytes", "n_features_used"]]
          .mean().reset_index())
    dp.to_csv(OUT / "per_dataset_pipeline_means.csv", index=False)

    notes = folds[folds["note"].notna()].groupby(["Dataset", "IMB", "note"]).size()
    if len(notes):
        notes.rename("n_folds").reset_index().to_csv(OUT / "fallback_notes.csv", index=False)

    all_pairs, all_anova, all_marg = [], [], []
    for sub, srcs in SUBSETS.items():
        d = dp if srcs is None else dp[dp["Source"].isin(srcs)]

        # 2) 边际均值（数据集层面：先在其它因子上平均，再跨数据集求均值±SD）
        for f in LEVELS:
            dm = d.groupby(["Dataset", f])[METRICS].mean().reset_index()
            g = dm.groupby(f)[METRICS].agg(["mean", "std"])
            g.columns = [f"{m}_{s}" for m, s in g.columns]
            g = g.reindex(LEVELS[f]).reset_index().rename(columns={f: "level"})
            g.insert(0, "factor", f)
            g.insert(0, "subset", sub)
            all_marg.append(g)

            # 3) 两两比较
            for m in MAIN_METRICS:
                pf = pairwise_family(dm, f, m)
                pf.insert(0, "subset", sub)
                all_pairs.append(pf)

        # 4) 区组全因子 ANOVA：方差分解
        for m in MAIN_METRICS:
            formula = (f"{m} ~ C(Dataset) + C(FS)*C(IMB)*C(CLF)"
                       f" + C(Dataset):C(FS) + C(Dataset):C(IMB) + C(Dataset):C(CLF)")
            fit = smf.ols(formula, data=d).fit()
            a = anova_lm(fit, typ=2)
            ss_tot = a["sum_sq"].sum()
            ss_res = a.loc["Residual", "sum_sq"]
            a["eta_sq"] = a["sum_sq"] / ss_tot
            a["partial_eta_sq"] = a["sum_sq"] / (a["sum_sq"] + ss_res)
            a = a.reset_index().rename(columns={"index": "term"})
            a["term"] = (a["term"].str.replace("C(", "", regex=False)
                         .str.replace(")", "", regex=False))
            a.insert(0, "metric", m)
            a.insert(0, "subset", sub)
            all_anova.append(a)

    marg = pd.concat(all_marg)
    pairs = pd.concat(all_pairs)
    pairs["effect_magnitude"] = pairs["rank_biserial_r"].map(rb_magnitude)
    anova = pd.concat(all_anova)
    marg.round(4).to_csv(OUT / "marginal_means.csv", index=False)
    pairs.to_csv(OUT / "pairwise_tests.csv", index=False)
    anova.to_csv(OUT / "anova_variance_decomposition.csv", index=False)

    # 方差分解精简表：各项 η²（%）
    eta = (anova[anova["term"] != "Residual"]
           .pivot_table(index="term", columns=["subset", "metric"], values="eta_sq") * 100)
    order = ["Dataset", "IMB", "CLF", "FS", "Dataset:IMB", "Dataset:CLF", "Dataset:FS",
             "IMB:CLF", "FS:IMB", "FS:CLF", "FS:IMB:CLF"]
    eta = eta.reindex([t for t in order if t in eta.index])
    eta.round(2).to_csv(OUT / "anova_eta_sq_percent.csv")

    # 5) 机制分析：不平衡处理增益 vs 数据特征
    mech = []
    dm_imb = dp.groupby(["Dataset", "Source", "IMB"])[METRICS].mean().reset_index()
    piv = {m: dm_imb.pivot_table(index=["Dataset", "Source"], columns="IMB", values=m)
           for m in METRICS}
    for (ds, src) in piv["f1"].index:
        row = {"Dataset": ds, "Source": src}
        s = summ.set_index("Dataset").loc[ds]
        row.update({"DefectRate%": s["DefectRate%"], "IR": s["IR"], "Modules": s["Modules"]})
        for m in ["f1", "mcc", "auc_pr", "auc", "recall", "precision"]:
            for k in ["SMOTE", "ADASYN", "CS", "THR"]:
                row[f"d_{m}_{k}"] = piv[m].loc[(ds, src), k] - piv[m].loc[(ds, src), "None"]
        row["THR_minus_bestResample_f1"] = piv["f1"].loc[(ds, src), "THR"] - \
            piv["f1"].loc[(ds, src), ["SMOTE", "ADASYN", "CS"]].max()
        mech.append(row)
    mech = pd.DataFrame(mech).sort_values("IR")
    mech.to_csv(OUT / "mechanism_per_dataset.csv", index=False)

    corr = []
    for col in [c for c in mech.columns if c.startswith("d_")]:
        for x in ["IR", "DefectRate%", "Modules"]:
            rho, p = stats.spearmanr(mech[x], mech[col])
            corr.append({"gain": col, "vs": x, "spearman_rho": rho, "p": p})
    pd.DataFrame(corr).to_csv(OUT / "mechanism_spearman.csv", index=False)

    # 6) 效率（R1-5）
    eff = (folds.groupby(["FS", "CLF"])[["fit_time", "predict_time", "model_bytes", "n_features_used"]]
           .mean().reset_index())
    base = eff[eff["FS"] == "None"].set_index("CLF")
    for c in ["fit_time", "predict_time", "model_bytes"]:
        eff[f"{c}_rel_to_None"] = eff.apply(lambda r: r[c] / base.loc[r["CLF"], c], axis=1)
    eff.to_csv(OUT / "efficiency_by_fs_clf.csv", index=False)
    eff_fs = folds.groupby("FS")[["fit_time", "predict_time", "model_bytes", "n_features_used"]].agg(["mean", "median"])
    eff_fs.to_csv(OUT / "efficiency_by_fs.csv")

    # 7) 描述性"最佳配置"（按 MCC），并给出与第二名的差距
    best = []
    for ds, g in dp.groupby("Dataset"):
        g = g.sort_values("mcc", ascending=False)
        b, s2 = g.iloc[0], g.iloc[1]
        best.append({"Dataset": ds, "Source": b["Source"],
                     "Best (FS+IMB+CLF)": f"{b['FS']}+{b['IMB']}+{b['CLF']}",
                     "MCC": b["mcc"], "AUC-PR": b["auc_pr"], "F1": b["f1"], "AUC": b["auc"],
                     "gap_to_2nd_MCC": b["mcc"] - s2["mcc"]})
    pd.DataFrame(best).to_csv(OUT / "best_config_descriptive.csv", index=False)

    # ------------------------------------------------------------------------
    # 图
    # ------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    src_color = {"NASA": "#3b6ea8", "AEEEM": "#d1803a", "JIRA": "#4c9a6a"}

    # 图 A：不平衡处理对各指标的影响（数据集层面箱线图 + 散点）
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, m in zip(axes, ["f1", "mcc", "auc_pr", "auc"]):
        data = [dm_imb[dm_imb.IMB == k][m].values for k in LEVELS["IMB"]]
        ax.boxplot(data, widths=0.55, showfliers=False,
                   medianprops=dict(color="black"), boxprops=dict(color="#666"))
        for xi, k in enumerate(LEVELS["IMB"], 1):
            sub = dm_imb[dm_imb.IMB == k]
            jitter = RNG.uniform(-0.15, 0.15, len(sub))
            ax.scatter(xi + jitter, sub[m], s=10, alpha=0.8,
                       c=[src_color[s] for s in sub.Source], linewidths=0)
        ax.set_xticks(range(1, 6), LEVELS["IMB"], rotation=30)
        ax.set_title(LABEL[m])
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=s) for s, c in src_color.items()]
    axes[-1].legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "fig_imbalance_effect.png", dpi=300)
    plt.close(fig)

    # 图 B：交互作用图 IMB × CLF、FS × IMB
    for m in ["mcc", "auc_pr", "f1"]:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
        g = dp.groupby(["IMB", "CLF"])[m].mean().unstack()
        for c in LEVELS["CLF"]:
            axes[0].plot(LEVELS["IMB"], g.loc[LEVELS["IMB"], c], marker="o", label=c)
        axes[0].set_title(f"{LABEL[m]}: imbalance × classifier")
        axes[0].legend(frameon=False, fontsize=8)
        g = dp.groupby(["IMB", "FS"])[m].mean().unstack()
        for f in LEVELS["FS"]:
            axes[1].plot(LEVELS["IMB"], g.loc[LEVELS["IMB"], f], marker="o", label=f)
        axes[1].set_title(f"{LABEL[m]}: imbalance × feature selection")
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"fig_interaction_{m}.png", dpi=300)
        plt.close(fig)

    # 图 C：机制——F1 增益 vs 不平衡比；AUC-PR 变化 vs 不平衡比
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for k, mk in [("SMOTE", "o"), ("CS", "s"), ("THR", "^")]:
        axes[0].scatter(mech["IR"], mech[f"d_f1_{k}"], marker=mk, s=22, alpha=0.8, label=k)
        axes[1].scatter(mech["IR"], mech[f"d_auc_pr_{k}"], marker=mk, s=22, alpha=0.8, label=k)
    for ax, t in zip(axes[:2], ["ΔF1 vs. no handling", "ΔAUC-PR vs. no handling"]):
        ax.set_xscale("log")
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("Imbalance ratio (non-defective / defective, log)")
        ax.set_title(t)
        ax.legend(frameon=False, fontsize=8)
    best_res = mech[["d_f1_SMOTE", "d_f1_ADASYN", "d_f1_CS"]].max(axis=1)
    axes[2].scatter(best_res, mech["d_f1_THR"], c=[src_color[s] for s in mech.Source], s=24)
    lim = [min(best_res.min(), mech["d_f1_THR"].min()) - 0.02,
           max(best_res.max(), mech["d_f1_THR"].max()) + 0.02]
    axes[2].plot(lim, lim, color="grey", lw=0.8, ls="--")
    axes[2].set_xlabel("ΔF1 of best resampling/weighting")
    axes[2].set_ylabel("ΔF1 of threshold moving")
    axes[2].set_title("Threshold moving vs. rebalancing")
    axes[2].legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_mechanism.png", dpi=300)
    plt.close(fig)

    # ------------------------------------------------------------------------
    # 控制台摘要
    # ------------------------------------------------------------------------
    pd.set_option("display.width", 200)
    print("\n=== 方差分解 η²（%）===")
    print(eta.round(1).to_string())
    key = pairs[(pairs.factor == "IMB") & (pairs.metric.isin(["mcc", "auc_pr", "f1"]))]
    cols = ["subset", "metric", "A", "B", "mean_diff(A-B)", "wins_A", "wins_B", "p_holm",
            "rank_biserial_r", f"P(equiv)_rope{ROPE}", f"P(A>B)_rope{ROPE}", f"P(A<B)_rope{ROPE}"]
    print("\n=== 不平衡处理两两比较 ===")
    print(key[cols].round(3).to_string(index=False))
    print("\n=== 机制：增益与不平衡比的 Spearman 相关 ===")
    c = pd.DataFrame(corr)
    print(c[c.vs == "IR"].round(3).to_string(index=False))
    print("\n=== 效率（按 FS）===")
    print(eff_fs.round(4).to_string())
    print(f"\n全部输出在 {OUT}")


if __name__ == "__main__":
    main()
