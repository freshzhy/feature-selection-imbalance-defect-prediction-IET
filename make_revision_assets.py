"""
生成 IET 大修稿所需的全部表格与图（从已有结果文件汇总，不重跑任何模型）
=====================================================================
输入：results_v3/、results_cpdp/、results_sensitivity/、results_nested/ 下的结果文件
输出：revision_assets/tables/*.csv 与 revision_assets/figures/fig*.png（300 dpi）

运行（项目根目录）：  python make_revision_assets.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT = ROOT / "revision_assets"
T, F = OUT / "tables", OUT / "figures"
T.mkdir(parents=True, exist_ok=True)
F.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_v3 import bayes_signed_rank, rank_biserial_paired  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
IMB = ["None", "SMOTE", "ADASYN", "CS", "THR"]
IMB_LAB = {"None": "None", "SMOTE": "SMOTE", "ADASYN": "ADASYN", "CS": "Cost-sens.", "THR": "Thr. moving"}
FSL = ["None", "IG", "AF"]
CLF = ["LR", "SVM", "RF", "XGB"]
SRC_C = {"NASA": "#3b6ea8", "AEEEM": "#d1803a", "JIRA": "#4c9a6a"}
LAB = {"mcc": "MCC", "auc_pr": "AUC-PR", "f1": "F1-score", "auc": "AUC-ROC"}


def rd(p, **kw):
    return pd.read_csv(ROOT / p, keep_default_na=False, na_values=[""], **kw)


summ = rd("results_v3/datasets_summary.csv")
dp = rd("results_v3/analysis/per_dataset_pipeline_means.csv")
for c in ["FS", "IMB"]:
    dp[c] = dp[c].replace({"": "None"}).fillna("None")

# ---------------------------------------------------------------- Table 1 数据集
t1 = summ[["Dataset", "Source", "Modules", "Features", "DefectRate%", "IR", "DuplicatesRemoved"]].copy()
t1.to_csv(T / "table1_datasets.csv", index=False)

# ---------------------------------------------------------------- Table 3 边际均值
rows = []
for sub, srcs in [("All (24)", None), ("NASA (12)", ["NASA"]), ("AEEEM+JIRA (12)", ["AEEEM", "JIRA"])]:
    d = dp if srcs is None else dp[dp.Source.isin(srcs)]
    for f, levels in [("FS", FSL), ("IMB", IMB), ("CLF", CLF)]:
        dm = d.groupby(["Dataset", f])[["mcc", "auc_pr", "f1", "auc", "recall", "precision"]].mean().reset_index()
        g = dm.drop(columns="Dataset").groupby(f).agg(["mean", "std"])
        for lv in levels:
            r = {"Subset": sub, "Factor": f, "Level": lv}
            for m in ["mcc", "auc_pr", "f1", "auc", "recall", "precision"]:
                r[m] = f"{g.loc[lv, (m, 'mean')]:.3f} ± {g.loc[lv, (m, 'std')]:.3f}"
                r[m + "_mean"] = g.loc[lv, (m, "mean")]
            rows.append(r)
t3 = pd.DataFrame(rows)
t3.to_csv(T / "table3_marginal_means.csv", index=False)

# ---------------------------------------------------------------- Table 4 不平衡处理关键对比
pw = rd("results_v3/analysis/pairwise_tests.csv")
keep = [("None", "SMOTE"), ("None", "ADASYN"), ("None", "CS"), ("None", "THR"),
        ("SMOTE", "CS"), ("SMOTE", "ADASYN"), ("ADASYN", "CS"), ("CS", "THR"), ("SMOTE", "THR")]
q = pw[(pw.factor == "IMB") & pw.metric.isin(["mcc", "auc_pr", "f1"])]
q = q[[(a, b) in keep for a, b in zip(q.A, q.B)]]
t4 = q[["subset", "metric", "A", "B", "mean_diff(A-B)", "CI95_low", "CI95_high", "wins_A", "wins_B",
        "p_holm", "rank_biserial_r", "P(A<B)_rope0.01", "P(equiv)_rope0.01", "P(A>B)_rope0.01",
        "P(equiv)_rope0.02"]]
t4.to_csv(T / "table4_imbalance_pairwise.csv", index=False)
pw[pw.factor.isin(["FS", "CLF"])].to_csv(T / "tableS_fs_clf_pairwise.csv", index=False)

# ---------------------------------------------------------------- Table 5 方差分解
eta = rd("results_v3/analysis/anova_eta_sq_percent.csv", header=[0, 1], index_col=0)
eta.to_csv(T / "fig2_eta_sq.csv")

# ---------------------------------------------------------------- Table 6 机制
pr = rd("results_v3/mechanism/prior_rule_tests.csv")
pr.to_csv(T / "table7_prior_rule.csv", index=False)
mc = rd("results_v3/mechanism/mechanism_by_classifier.csv", index_col=0)
mf = rd("results_v3/mechanism/mechanism_folds.csv")
cols = ["agree_SMOTE", "kappa_SMOTE", "agree_CS", "kappa_CS", "spearman_p0_SMOTE", "spearman_p0_CS",
        "ece_None", "ece_SMOTE", "ece_CS", "meanprob_None", "meanprob_SMOTE", "meanprob_CS", "actual_rate"]
t6b = mc[cols].copy()
t6b.loc["All"] = mf[cols].mean()
t6b.to_csv(T / "table8_agreement_calibration.csv")

# ---------------------------------------------------------------- Table 7 嵌套选择
ss = rd("results_nested/select_summary.csv")
ss.to_csv(T / "table5_nested_selection.csv", index=False)
d = ss.nested_MCC - ss.default_RF_CS_MCC
nested_stats = {
    "oracle_mean": ss.oracle_MCC.mean(), "nested_mean": ss.nested_MCC.mean(),
    "rfcs_mean": ss.default_RF_CS_MCC.mean(), "optimism_mean": ss.optimism_MCC.mean(),
    "optimism_median": ss.optimism_MCC.median(), "optimism_max": ss.optimism_MCC.max(),
    "optimism_min": ss.optimism_MCC.min(),
    "nested_vs_rfcs_wins": int((d > 0).sum()), "nested_vs_rfcs_p": stats.wilcoxon(d)[1],
    "nested_vs_rfcs_Pequiv": bayes_signed_rank(d.values, 0.01)[1],
    "share_imb": ss.share_selected_with_imbalance.mean(), "share_ens": ss.share_selected_ensemble.mean(),
    "share_fs": ss.share_selected_with_FS.mean(),
}
pd.Series(nested_stats).to_csv(T / "nested_stats.csv")

# ---------------------------------------------------------------- Table 8 调参
tu = rd("results_nested/tune_per_dataset.csv")
tu["IMB"] = tu["IMB"].fillna("None")
tu["tuned"] = tu["tuned"].astype(str).str.lower().map({"true": True, "false": False})
rows = []
for f, levels in [("CLF", CLF), ("IMB", ["None", "SMOTE", "CS"])]:
    g = tu.groupby(["Dataset", "tuned", f])[["mcc", "auc_pr", "f1", "auc"]].mean().reset_index()
    for lv in levels:
        a = g[(g[f] == lv) & (~g.tuned)].set_index("Dataset")
        b = g[(g[f] == lv) & (g.tuned)].set_index("Dataset")
        r = {"Factor": f, "Level": lv}
        for m in ["mcc", "auc_pr"]:
            dd = (b[m] - a.loc[b.index, m]).values
            r[f"{m}_default"] = a[m].mean(); r[f"{m}_tuned"] = b[m].mean()
            r[f"{m}_diff"] = dd.mean(); r[f"{m}_wins_tuned"] = int((dd > 0).sum())
            r[f"{m}_p"] = stats.wilcoxon(dd)[1] if np.any(dd != 0) else 1.0
        rows.append(r)
t8 = pd.DataFrame(rows)
rk = rd("results_nested/tune_classifier_ranks.csv").groupby("CLF")[["rank_default", "rank_tuned"]].mean()
t8 = t8.merge(rk, left_on="Level", right_index=True, how="left")
t8.to_csv(T / "table6_tuning.csv", index=False)
# 调参后 IMB 对比
g = tu[tu.tuned].groupby(["Dataset", "IMB"])[["mcc", "auc_pr", "f1"]].mean().unstack("IMB")
tune_imb = {}
for m in ["mcc", "f1", "auc_pr"]:
    for a in ["SMOTE", "CS"]:
        dd = (g[(m, a)] - g[(m, "None")]).values
        tune_imb[f"{m}_{a}_minus_None"] = dd.mean()
        tune_imb[f"{m}_{a}_wins"] = int((dd > 0).sum())
        tune_imb[f"{m}_{a}_p"] = stats.wilcoxon(dd)[1]
pd.Series(tune_imb).to_csv(T / "tuning_imbalance_stats.csv")

# ---------------------------------------------------------------- Table 9 CPDP
cp = rd("results_cpdp/cpdp_results.csv")
for c in ["FS", "IMB"]:
    cp[c] = cp[c].fillna("None")
rows = []
for f, levels in [("IMB", IMB), ("CLF", CLF), ("FS", FSL)]:
    g = cp.groupby(["Target", f])[["mcc", "auc_pr", "f1", "auc"]].mean().reset_index()
    base = "None" if f != "CLF" else "RF"
    for lv in levels:
        r = {"Factor": f, "Level": lv}
        a = g[g[f] == lv].set_index("Target")
        for m in ["mcc", "auc_pr", "f1", "auc"]:
            r[m] = f"{a[m].mean():.3f} ± {a[m].std():.3f}"
            r[m + "_mean"] = a[m].mean()
        if lv != base:
            b = g[g[f] == base].set_index("Target")
            for m in ["mcc", "auc_pr"]:
                dd = (a[m] - b.loc[a.index, m]).values
                r[f"{m}_vs_{base}_diff"] = dd.mean()
                r[f"{m}_vs_{base}_wins"] = int((dd > 0).sum())
                r[f"{m}_vs_{base}_p"] = stats.wilcoxon(dd)[1]
        rows.append(r)
t9 = pd.DataFrame(rows)
t9.to_csv(T / "table9_cpdp.csv", index=False)
g = cp.groupby(["Target", "IMB"])[["mcc"]].mean().unstack("IMB")["mcc"]
cp_extra = {}
for a, b in [("CS", "SMOTE"), ("THR", "SMOTE"), ("CS", "THR")]:
    dd = (g[a] - g[b]).values
    cp_extra[f"{a}_minus_{b}"] = dd.mean(); cp_extra[f"{a}_vs_{b}_wins"] = int((dd > 0).sum())
    cp_extra[f"{a}_vs_{b}_p"] = stats.wilcoxon(dd)[1]
g2 = cp.groupby(["Target", "CLF"])["mcc"].mean().unstack()
dd = (g2["LR"] - g2["RF"]).values
cp_extra.update({"LR_minus_RF_mcc": dd.mean(), "LR_vs_RF_wins": int((dd > 0).sum()), "LR_vs_RF_p": stats.wilcoxon(dd)[1]})
pd.Series(cp_extra).to_csv(T / "cpdp_extra_stats.csv")

# ---------------------------------------------------------------- 敏感性
sa = rd("results_sensitivity/summary_A.csv")
sb = rd("results_sensitivity/summary_B.csv")
sa.to_csv(T / "tableS_sensitivity_A.csv", index=False)
# B 部分的无 FS 基线：v3 前 5 轮（相同折划分）
af = rd("results_v3/all_folds.csv", low_memory=False)
for c in ["FS", "IMB"]:
    af[c] = af[c].fillna("None")
b0 = (af[(af.FS == "None") & af.IMB.isin(["None", "CS"]) & (af.repeat < 5)]
      .groupby(["Dataset", "IMB"])[["f1", "mcc", "auc_pr", "fit_time"]].mean()
      .groupby("IMB").mean().reset_index())
b0.insert(0, "fs_ratio", 1.0)
b0.insert(0, "FS", "None (all features)")
sbb = pd.concat([b0, sb], ignore_index=True)
sbb.to_csv(T / "tableS_sensitivity_B.csv", index=False)

# 效率
eff = af.groupby(["FS"])[["fit_time", "predict_time", "model_bytes", "n_features_used"]].mean()
effc = af.groupby(["FS", "CLF"])[["fit_time", "predict_time"]].mean().unstack("FS")
eff.to_csv(T / "efficiency_fs.csv")
effc.to_csv(T / "efficiency_fs_clf.csv")

# ================================================================= 图
dm_imb = dp.groupby(["Dataset", "Source", "IMB"])[["mcc", "auc_pr", "f1", "auc"]].mean().reset_index()
dm_fs = dp.groupby(["Dataset", "Source", "FS"])[["mcc", "auc_pr", "f1", "auc"]].mean().reset_index()
dm_clf = dp.groupby(["Dataset", "Source", "CLF"])[["mcc", "auc_pr", "f1", "auc"]].mean().reset_index()
rng = np.random.default_rng(7)
handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=s) for s, c in SRC_C.items()]


def box_panel(ax, dm, f, levels, m, labels=None):
    data = [dm[dm[f] == lv][m].values for lv in levels]
    ax.boxplot(data, widths=0.55, showfliers=False, medianprops=dict(color="black"),
               boxprops=dict(color="#777"), whiskerprops=dict(color="#777"), capprops=dict(color="#777"))
    for xi, lv in enumerate(levels, 1):
        s = dm[dm[f] == lv]
        ax.scatter(xi + rng.uniform(-0.16, 0.16, len(s)), s[m], s=9, alpha=0.85,
                   c=[SRC_C[v] for v in s.Source], linewidths=0, zorder=3)
    ax.set_xticks(range(1, len(levels) + 1), labels or levels, rotation=25 if len(levels) > 4 else 0)


# Fig 1：三个因子 × 三个指标
fig, axes = plt.subplots(3, 3, figsize=(10, 8.4))
for j, m in enumerate(["mcc", "auc_pr", "f1"]):
    box_panel(axes[0, j], dm_fs, "FS", FSL, m)
    box_panel(axes[1, j], dm_imb, "IMB", IMB, m, [IMB_LAB[i] for i in IMB])
    box_panel(axes[2, j], dm_clf, "CLF", CLF, m)
    axes[0, j].set_title(LAB[m])
for i, lab in enumerate(["Feature selection", "Imbalance handling", "Classifier"]):
    axes[i, 0].set_ylabel(lab)
axes[0, 2].legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(F / "fig1_factor_effects.png", dpi=300)
plt.close(fig)

# Fig 2：方差分解
terms = ["IMB", "CLF", "FS", "Dataset:IMB", "Dataset:CLF", "Dataset:FS", "IMB:CLF", "FS:IMB", "FS:CLF", "FS:IMB:CLF"]
tl = {"IMB": "Imbalance (I)", "CLF": "Classifier (C)", "FS": "Feature sel. (F)", "Dataset:IMB": "D×I",
      "Dataset:CLF": "D×C", "Dataset:FS": "D×F", "IMB:CLF": "I×C", "FS:IMB": "F×I", "FS:CLF": "F×C",
      "FS:IMB:CLF": "F×I×C"}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
for ax, sub, tt in zip(axes, ["ALL", "NASA", "EXTERNAL"], ["All 24 datasets", "NASA (12)", "AEEEM + JIRA (12)"]):
    x = np.arange(len(terms)); w = 0.2
    for k, m in enumerate(["mcc", "auc_pr", "f1", "auc"]):
        ax.bar(x + (k - 1.5) * w, [eta.loc[t, (sub, m)] for t in terms], w, label=LAB[m])
    ax.set_xticks(x, [tl[t] for t in terms], rotation=60, fontsize=7.5)
    ax.set_title(tt)
axes[0].set_ylabel("η² (% of total variance)")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(F / "fig2_variance_decomposition.png", dpi=300)
plt.close(fig)

# Fig 3：交互作用 I×C
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, m in zip(axes, ["mcc", "auc_pr", "f1"]):
    g = dp.groupby(["IMB", "CLF"])[m].mean().unstack()
    for c in CLF:
        ax.plot([IMB_LAB[i] for i in IMB], g.loc[IMB, c], marker="o", label=c)
    ax.set_title(LAB[m]); ax.tick_params(axis="x", rotation=25)
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(F / "fig3_interaction.png", dpi=300)
plt.close(fig)

# Fig 4：机制
mech = rd("results_v3/analysis/mechanism_per_dataset.csv")
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
for k, mk in [("SMOTE", "o"), ("CS", "s"), ("THR", "^")]:
    axes[0].scatter(mech.IR, mech[f"d_mcc_{k}"], marker=mk, s=20, alpha=0.8, label=IMB_LAB[k])
    axes[1].scatter(mech.IR, mech[f"d_auc_pr_{k}"], marker=mk, s=20, alpha=0.8, label=IMB_LAB[k])
for ax, t in zip(axes[:2], ["(a) ΔMCC vs. no handling", "(b) ΔAUC-PR vs. no handling"]):
    ax.set_xscale("log"); ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("Imbalance ratio (log scale)"); ax.set_title(t, fontsize=9); ax.legend(frameon=False, fontsize=7)
mb = rd("results_v3/mechanism/mechanism_by_dataset.csv")
for s, c in SRC_C.items():
    z = mb[mb.Source == s]
    axes[2].scatter(z.mcc_CS, z.mcc_PRIOR, color=c, s=22, label=s)
lim = [mb[["mcc_CS", "mcc_PRIOR"]].min().min() - 0.02, mb[["mcc_CS", "mcc_PRIOR"]].max().max() + 0.02]
axes[2].plot(lim, lim, ls="--", color="grey", lw=0.8)
axes[2].set_xlabel("MCC, cost-sensitive learning"); axes[2].set_ylabel("MCC, prior-threshold rule")
axes[2].set_title("(c) Prior-threshold rule vs. cost-sensitive", fontsize=9); axes[2].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(F / "fig5_mechanism.png", dpi=300)
plt.close(fig)

# Fig 5：概率分布（直接复用 analyze_mechanism 输出）
import shutil
src = ROOT / "results_v3/mechanism/fig_score_distributions.png"
if src.exists():
    shutil.copy(src, F / "fig4_score_distributions.png")

# Fig 6：敏感性
sa2 = sa.copy()
fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for ax, m in zip(axes, ["f1", "mcc", "auc_pr"]):
    for imb, k, mk in [("SMOTE", 3.0, "o"), ("ADASYN", 3.0, "s")]:
        z = sa2[(sa2.IMB == imb) & (sa2.k.astype(float) == k)]
        ax.plot(z.ratio.astype(float), z[m], marker=mk, label=f"{imb} (k=3): target ratio")
    z = sa2[sa2.IMB == "CS"]
    ax.plot(z.w.astype(float), z[m], marker="^", label="Cost-sens.: weight multiplier w")
    ax.set_xlabel("Resampling ratio / weight multiplier"); ax.set_title(LAB[m])
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(F / "fig6_sensitivity.png", dpi=300)
plt.close(fig)

# Fig 7：CPDP
g = cp.groupby(["Target", "IMB"])[["mcc", "auc_pr"]].mean().reset_index()
g2 = cp.groupby(["Target", "CLF"])[["mcc", "auc_pr"]].mean().reset_index()
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))
for ax, (gg, f, lv, lab) in zip(axes, [(g, "IMB", IMB, [IMB_LAB[i] for i in IMB]), (g2, "CLF", CLF, CLF)]):
    x = np.arange(len(lv)); w = 0.38
    for k, m in enumerate(["mcc", "auc_pr"]):
        mu = [gg[gg[f] == v][m].mean() for v in lv]
        se = [gg[gg[f] == v][m].std() / np.sqrt(gg[gg[f] == v][m].count()) for v in lv]
        ax.bar(x + (k - 0.5) * w, mu, w, yerr=se, capsize=2, label=LAB[m])
    ax.set_xticks(x, lab, rotation=20)
axes[0].set_title("Cross-project: imbalance handling"); axes[1].set_title("Cross-project: classifier")
axes[0].legend(frameon=False, fontsize=7)
fig.tight_layout()
fig.savefig(F / "fig7_cpdp.png", dpi=300)
plt.close(fig)

print("完成：", OUT)
for p in sorted(T.glob("*.csv")):
    print(" ", p.name)
for p in sorted(F.glob("*.png")):
    print(" ", p.name)
