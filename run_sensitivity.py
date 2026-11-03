"""
敏感性实验（回应 R2："SMOTE/ADASYN 参数固定、FS 只有两个过滤法且固定 50%"）
=========================================================================

A 部分：不平衡处理参数（FS=None，4 个分类器）
    SMOTE / ADASYN：k ∈ {1,3,5,7} × 采样后少数/多数比 ∈ {0.25, 0.5, 0.75, 1.0}
    CS（代价敏感）：正类权重 = w × (负/正)，w ∈ {0.25, 0.5, 0.75, 1.0}
    → 共 2×4×4×4 + 4×4 = 144 个配置

B 部分：特征选择方法与保留比例（IMB ∈ {None, CS}，4 个分类器）
    方法：IG、AF（原有过滤法）、mRMR（去冗余过滤法）、
          L1（嵌入式：L1 逻辑回归系数）、RFI（嵌入式：随机森林重要性）
    保留比例 ∈ {0.25, 0.5, 0.75}
    → 共 5×3×2×4 = 120 个配置

交叉验证：5 轮 × 10 折（--repeats 可改），24 个数据集。
依赖 run_experiment_v3.py 中的数据加载与指标函数（放在同一目录即可）。

运行（项目根目录）：
    python run_sensitivity.py --quick          # 冒烟测试（MC2、EQ，1 轮）
    python run_sensitivity.py                  # 全量，预计 2–3 小时
    python run_sensitivity.py --part A         # 只跑 A 部分（或 B）
可中断续跑（每个数据集×部分写一个文件）。

输出：results_sensitivity/folds/<部分>_<数据集>.csv、results_sensitivity/all_folds.csv
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import run_experiment_v3 as base   # 复用加载/分类器/指标

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_sensitivity"
CLFS = ["LR", "SVM", "RF", "XGB"]


# ----------------------------------------------------------------------------
# 额外的特征选择器
# ----------------------------------------------------------------------------
class TopKSelector(BaseEstimator, TransformerMixin):
    """按某种得分取前 k 个特征"""
    def __init__(self, method="mrmr", k=10, seed=42):
        self.method, self.k, self.seed = method, k, seed

    def fit(self, X, y):
        k = min(self.k, X.shape[1])
        if self.method == "mrmr":
            rel = mutual_info_classif(X, y, random_state=self.seed)
            C = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
            chosen = [int(np.argmax(rel))]
            while len(chosen) < k:
                rest = [j for j in range(X.shape[1]) if j not in chosen]
                red = C[np.ix_(rest, chosen)].mean(axis=1)
                score = rel[rest] - red * rel.max()   # MID 形式，冗余按相关系数缩放到 MI 量级
                chosen.append(rest[int(np.argmax(score))])
            self.support_idx_ = np.array(chosen)
        elif self.method == "l1":
            lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.1,
                                    class_weight="balanced", random_state=self.seed, max_iter=2000)
            lr.fit(X, y)
            self.support_idx_ = np.argsort(-np.abs(lr.coef_[0]))[:k]
        elif self.method == "rfi":
            rf = RandomForestClassifier(n_estimators=200, random_state=self.seed,
                                        class_weight="balanced", n_jobs=1).fit(X, y)
            self.support_idx_ = np.argsort(-rf.feature_importances_)[:k]
        else:
            raise ValueError(self.method)
        return self

    def transform(self, X):
        return X[:, self.support_idx_]

    def get_support(self):
        return self.support_idx_


def make_fs(method, k, seed):
    if method == "None":
        return None
    if method == "IG":
        from functools import partial
        return SelectKBest(partial(mutual_info_classif, random_state=seed), k=k)
    if method == "AF":
        return SelectKBest(f_classif, k=k)
    return TopKSelector({"mRMR": "mrmr", "L1": "l1", "RFI": "rfi"}[method], k, seed)


# ----------------------------------------------------------------------------
# 配置清单
# ----------------------------------------------------------------------------
def configs(part):
    cfg = []
    if part == "A":
        for imb in ["SMOTE", "ADASYN"]:
            for k in [1, 3, 5, 7]:
                for ratio in [0.25, 0.5, 0.75, 1.0]:
                    for c in CLFS:
                        cfg.append(dict(FS="None", fs_ratio=1.0, IMB=imb, k=k, ratio=ratio, w=np.nan, CLF=c))
        for w in [0.25, 0.5, 0.75, 1.0]:
            for c in CLFS:
                cfg.append(dict(FS="None", fs_ratio=1.0, IMB="CS", k=np.nan, ratio=np.nan, w=w, CLF=c))
    else:
        for fs in ["IG", "AF", "mRMR", "L1", "RFI"]:
            for r in [0.25, 0.5, 0.75]:
                for imb in ["None", "CS"]:
                    for c in CLFS:
                        cfg.append(dict(FS=fs, fs_ratio=r, IMB=imb, k=np.nan, ratio=np.nan,
                                        w=1.0 if imb == "CS" else np.nan, CLF=c))
    return cfg


def build(cfg, n_feat, pos, neg, seed):
    steps = [("scaler", StandardScaler())]
    if cfg["FS"] != "None":
        steps.append(("fs", make_fs(cfg["FS"], max(1, int(round(n_feat * cfg["fs_ratio"]))), seed)))
    imb = cfg["IMB"]
    if imb in ("SMOTE", "ADASYN"):
        cur = pos / neg
        # 目标少数/多数比若不高于当前比（如近平衡的 EQ），无需过采样，跳过采样器
        if int(min(1.0, cfg["ratio"]) * neg) > pos:
            k = int(max(1, min(cfg["k"], pos - 1)))
            r = min(1.0, cfg["ratio"])
            S = SMOTE(k_neighbors=k, sampling_strategy=r, random_state=seed) if imb == "SMOTE" \
                else ADASYN(n_neighbors=k, sampling_strategy=r, random_state=seed)
            steps.append(("sampler", S))
    clf = base.make_clf(cfg["CLF"], "None", seed)
    if imb == "CS":
        wpos = cfg["w"] * neg / pos
        if cfg["CLF"] == "XGB":
            clf.set_params(scale_pos_weight=wpos)
        else:
            clf.set_params(class_weight={0: 1.0, 1: wpos})
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def run_task(X, y, cfg, rep, n_folds):
    warnings.filterwarnings("ignore")
    seed = 42 + rep
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1000 + rep)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        pos = int(y[tr].sum()); neg = len(tr) - pos
        note = ""
        pipe = build(cfg, X.shape[1], pos, neg, seed)
        if cfg["IMB"] in ("SMOTE", "ADASYN") and "sampler" not in pipe.named_steps:
            note = "no_resampling_needed"
        t0 = time.perf_counter()
        try:
            pipe.fit(X[tr], y[tr])
        except (ValueError, RuntimeError) as e:
            note = f"fallback_SMOTE:{type(e).__name__}"
            c2 = dict(cfg, IMB="SMOTE")
            pipe = build(c2, X.shape[1], pos, neg, seed)
            pipe.fit(X[tr], y[tr])
        fit_t = time.perf_counter() - t0
        prob = pipe.predict_proba(X[te])[:, 1]
        pred = pipe.predict(X[te])
        row = dict(cfg, repeat=rep, fold=fold, fit_time=fit_t, note=note)
        row.update(base.fold_metrics(y[te], pred, prob))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "AB"], default="AB")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    global OUT
    if args.quick:
        OUT = ROOT / "results_sensitivity_quick"
        args.repeats, args.datasets, args.overwrite = 1, ["MC2", "EQ"], True
    (OUT / "folds").mkdir(parents=True, exist_ok=True)

    items = base.discover_datasets(False)
    if args.datasets:
        want = {d.lower() for d in args.datasets}
        items = [it for it in items if it[0].lower() in want]

    t_all = time.time()
    for part in args.part if args.part != "AB" else "AB":
        cfgs = configs(part)
        for name, src, path, dedup in items:
            out = OUT / "folds" / f"{part}_{name}.csv"
            if out.exists() and not args.overwrite:
                print(f"[跳过] {part} {name}")
                continue
            Xdf, ys, _, _ = base.load_arff(path, dedup)
            X, y = Xdf.values, ys.values.astype(int)
            t0 = time.time()
            res = Parallel(n_jobs=args.jobs, batch_size=1)(
                delayed(run_task)(X, y, c, r, args.folds) for r in range(args.repeats) for c in cfgs)
            df = pd.DataFrame([row for rs in res for row in rs])
            df.insert(0, "Source", src)
            df.insert(0, "Dataset", name)
            df.insert(0, "Part", part)
            df.to_csv(out, index=False)
            print(f"[完成] 部分{part} {name}  配置数={len(cfgs)}  用时 {time.time()-t0:.0f}s", flush=True)

    parts = [pd.read_csv(f) for f in sorted((OUT / "folds").glob("*.csv"))]
    allf = pd.concat(parts)
    allf["FS"] = allf["FS"].fillna("None")      # CSV 读回时字符串 "None" 会变成 NaN
    allf["IMB"] = allf["IMB"].fillna("None")
    allf.to_csv(OUT / "all_folds.csv", index=False)

    # 简要汇总（数据集层面平均）
    keyA = ["IMB", "k", "ratio", "w"]
    A = allf[allf.Part == "A"]
    if len(A):
        s = (A.groupby(["Dataset"] + keyA, dropna=False)[["f1", "mcc", "auc_pr"]].mean()
             .groupby(keyA, dropna=False).mean().round(3))
        s.to_csv(OUT / "summary_A.csv")
        print("\n=== A 部分：不平衡参数（24 数据集平均）===")
        print(s.to_string())
    B = allf[allf.Part == "B"]
    if len(B):
        keyB = ["FS", "fs_ratio", "IMB"]
        s = (B.groupby(["Dataset"] + keyB)[["f1", "mcc", "auc_pr", "fit_time"]].mean()
             .groupby(keyB).mean().round(3))
        s.to_csv(OUT / "summary_B.csv")
        print("\n=== B 部分：特征选择方法 × 保留比例（24 数据集平均）===")
        print(s.to_string())
    print(f"\n总用时 {(time.time()-t_all)/60:.1f} 分钟，输出在 {OUT}")


if __name__ == "__main__":
    main()
