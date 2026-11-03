"""
机制分析：不平衡处理是否本质上等价于"移动决策阈值"（回应 R2-2 "why"）
=====================================================================

只用 results_v3/oof/*.npz 中已保存的样本级预测概率，不需要重跑任何模型。

检验四件事（均在 FS=None 的管道上做，所有分类器、所有 10 轮）：

1. 理论预测阈值 PRIOR：
   贝叶斯决策理论下，class_weight='balanced'（正类权重 = 负/正）等价于把
   原模型概率的决策阈值从 0.5 移到训练集缺陷率 π（Elkan 2001）。
   → 直接用"无处理"模型的概率 p0，以 p0 ≥ π_train 作判决，得到 PRIOR 规则，
     与 CS / SMOTE / THR 的 F1、MCC 比较。PRIOR 不需要任何额外训练。

2. 分位数匹配一致率：
   在每个测试折里，取 SMOTE（或 CS）模型判为缺陷的个数 k，
   把无处理模型 p0 最高的 k 个样本判为缺陷，看两者判决一致的比例和 Cohen's κ。
   一致率高 → 重采样/加权后的模型 ≈ 原模型 + 换一个阈值。

3. 排序一致性：p0 与 p_SMOTE、p_CS 的 Spearman 相关（逐折计算后平均）。

4. 校准：期望校准误差 ECE（10 等宽箱）与平均预测概率 vs 实际缺陷率，
   说明重采样/加权主要是把概率整体"抬高"（失去校准），而非改善排序。

运行（项目根目录）：
    python analyze_mechanism.py         # 约 2–5 分钟，按数据集并行
输出：results_v3/mechanism/
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results_v3"
OUT = RES / "mechanism"
OUT.mkdir(parents=True, exist_ok=True)

CLFS = ["LR", "SVM", "RF", "XGB"]
FS = "None"
REPRESENTATIVE = [("PC3", "NASA"), ("PDE", "AEEEM"), ("derby-10.5.1.1", "JIRA")]


def ece(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return e


def per_dataset(npz_path, source):
    z = np.load(npz_path)
    ds = npz_path.stem
    y = z["y"].astype(int)
    reps = sorted({int(k.split("|")[1]) for k in z.files if k.startswith("fold|")})
    rows = []
    for clf in CLFS:
        for r in reps:
            key = lambda imb, kind="prob": f"{kind}|{FS}|{imb}|{clf}|{r}"
            p0, pS, pC = z[key("None")], z[key("SMOTE")], z[key("CS")]
            dS, dC, dT = z[key("SMOTE", "pred")], z[key("CS", "pred")], z[key("THR", "pred")]
            d0 = z[key("None", "pred")]
            fid = z[f"fold|{r}"]
            for f in np.unique(fid):
                te = fid == f
                yt = y[te]
                pi = y[~te].mean()                       # 训练折缺陷率
                dPrior = (p0[te] >= pi).astype(int)
                out = {"Dataset": ds, "Source": source, "CLF": clf, "repeat": r, "fold": int(f),
                       "train_prior": pi}
                for name, pred in [("None", d0[te]), ("PRIOR", dPrior), ("CS", dC[te]),
                                   ("SMOTE", dS[te]), ("THR", dT[te])]:
                    out[f"f1_{name}"] = f1_score(yt, pred, zero_division=0)
                    out[f"mcc_{name}"] = matthews_corrcoef(yt, pred)
                    out[f"npos_{name}"] = int(pred.sum())
                # 分位数匹配：同样多的阳性判决，看是否就是 p0 排名最前的那些
                for name, pred in [("SMOTE", dS[te]), ("CS", dC[te])]:
                    k = int(pred.sum())
                    q = np.zeros(te.sum(), int)
                    if k > 0:
                        q[np.argsort(-p0[te], kind="stable")[:k]] = 1
                    out[f"agree_{name}"] = float((q == pred).mean())
                    out[f"kappa_{name}"] = (cohen_kappa_score(q, pred)
                                            if len(np.unique(np.r_[q, pred])) > 1 else 1.0)
                    # 在"阳性判决"中的重合率（Jaccard），比整体一致率更严格
                    inter = int(((q == 1) & (pred == 1)).sum())
                    union = int(((q == 1) | (pred == 1)).sum())
                    out[f"jaccard_pos_{name}"] = inter / union if union else 1.0
                out["spearman_p0_SMOTE"] = stats.spearmanr(p0[te], pS[te])[0]
                out["spearman_p0_CS"] = stats.spearmanr(p0[te], pC[te])[0]
                for name, p in [("None", p0[te]), ("SMOTE", pS[te]), ("CS", pC[te])]:
                    out[f"ece_{name}"] = ece(yt, p)
                    out[f"meanprob_{name}"] = float(p.mean())
                out["actual_rate"] = float(yt.mean())
                rows.append(out)
    return rows


def plot_distributions(fold_df):
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for col, (ds, src) in enumerate(REPRESENTATIVE):
        p = RES / "oof" / f"{ds}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        y = z["y"].astype(int)
        p0, pS = z[f"prob|None|None|RF|0"], z[f"prob|None|SMOTE|RF|0"]
        thr = fold_df[(fold_df.Dataset == ds)]  # 仅用于取 π
        pi = y.mean()
        bins = np.linspace(0, 1, 26)
        for row, (p_, title) in enumerate([(p0, "No handling"), (pS, "SMOTE")]):
            ax = axes[row, col]
            ax.hist(p_[y == 0], bins=bins, alpha=0.6, density=True, label="clean", color="#7a8ca3")
            ax.hist(p_[y == 1], bins=bins, alpha=0.6, density=True, label="defective", color="#c4553d")
            ax.axvline(0.5, color="black", ls="--", lw=1, label="0.5")
            if row == 0:
                ax.axvline(pi, color="#2a7a3b", ls=":", lw=1.5, label=f"prior π={pi:.2f}")
            ax.set_title(f"{ds} ({src}) – RF, {title}", fontsize=9)
            ax.set_xlabel("predicted defect probability")
            if col == 0:
                ax.set_ylabel("density")
            ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_score_distributions.png", dpi=300)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, (ds, src) in zip(axes, REPRESENTATIVE):
        p = RES / "oof" / f"{ds}.npz"
        if not p.exists():
            continue
        z = np.load(p)
        y = z["y"].astype(int)
        p0, pS = z["prob|None|None|RF|0"], z["prob|None|SMOTE|RF|0"]
        r0, rS = stats.rankdata(p0) / len(p0), stats.rankdata(pS) / len(pS)
        ax.scatter(r0[y == 0], rS[y == 0], s=3, alpha=0.3, color="#7a8ca3", label="clean")
        ax.scatter(r0[y == 1], rS[y == 1], s=5, alpha=0.6, color="#c4553d", label="defective")
        rho = stats.spearmanr(p0, pS)[0]
        ax.set_title(f"{ds}: rank(no handling) vs rank(SMOTE), ρ={rho:.2f}", fontsize=9)
        ax.set_xlabel("percentile rank, no handling")
        ax.set_ylabel("percentile rank, SMOTE")
        ax.legend(frameon=False, fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_rank_agreement.png", dpi=300)
    plt.close(fig)


def main():
    summ = pd.read_csv(RES / "datasets_summary.csv")
    src = dict(zip(summ.Dataset, summ.Source))
    files = sorted((RES / "oof").glob("*.npz"))
    jobs = max(1, (os.cpu_count() or 4) - 2)
    res = Parallel(n_jobs=jobs)(delayed(per_dataset)(f, src.get(f.stem, "?")) for f in files)
    folds = pd.DataFrame([r for rs in res for r in rs])
    folds.to_csv(OUT / "mechanism_folds.csv", index=False)

    num = folds.select_dtypes("number").columns.drop(["repeat", "fold"])
    by_ds = folds.groupby(["Dataset", "Source"])[num].mean().reset_index()
    by_ds = by_ds.merge(summ[["Dataset", "IR"]], on="Dataset").sort_values("IR")
    by_ds.to_csv(OUT / "mechanism_by_dataset.csv", index=False)
    by_clf = folds.groupby("CLF")[num].mean()
    by_clf.to_csv(OUT / "mechanism_by_classifier.csv")

    # 1) PRIOR 规则 vs CS / SMOTE / THR（数据集层面，Wilcoxon + 贝叶斯 ROPE）
    from analyze_v3 import bayes_signed_rank, rank_biserial_paired
    rows = []
    for m in ["f1", "mcc"]:
        for a, b in [("PRIOR", "CS"), ("PRIOR", "SMOTE"), ("PRIOR", "THR"), ("PRIOR", "None")]:
            d = (by_ds[f"{m}_{a}"] - by_ds[f"{m}_{b}"]).values
            try:
                p = stats.wilcoxon(d)[1]
            except ValueError:
                p = 1.0
            pl, pe, pr = bayes_signed_rank(d, 0.01)
            rows.append({"metric": m, "A": a, "B": b, "mean_A": by_ds[f"{m}_{a}"].mean(),
                         "mean_B": by_ds[f"{m}_{b}"].mean(), "mean_diff": d.mean(),
                         "wins_A": int((d > 0).sum()), "wins_B": int((d < 0).sum()),
                         "p_wilcoxon": p, "rank_biserial_r": rank_biserial_paired(d),
                         "P(A<B)": pl, "P(equiv,±0.01)": pe, "P(A>B)": pr})
    tests = pd.DataFrame(rows)
    tests.to_csv(OUT / "prior_rule_tests.csv", index=False)

    plot_distributions(folds)

    pd.set_option("display.width", 200)
    print("\n=== 1. 理论阈值 PRIOR（p0 ≥ 训练缺陷率）vs 各方法（24 个数据集）===")
    print(tests.round(3).to_string(index=False))
    print("\n=== 2./3./4. 一致率、排序相关、校准（按分类器平均）===")
    cols = ["agree_SMOTE", "kappa_SMOTE", "jaccard_pos_SMOTE", "agree_CS", "kappa_CS",
            "jaccard_pos_CS", "spearman_p0_SMOTE", "spearman_p0_CS",
            "ece_None", "ece_SMOTE", "ece_CS", "meanprob_None", "meanprob_SMOTE",
            "meanprob_CS", "actual_rate"]
    print(by_clf[cols].round(3).T.to_string())
    print("\n总体平均：")
    print(folds[cols].mean().round(3).to_string())
    print(f"\n输出在 {OUT}")


if __name__ == "__main__":
    main()
