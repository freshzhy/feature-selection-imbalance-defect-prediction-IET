"""
嵌套交叉验证（回应 R1-3 "最佳配置乐观偏差" 与 R2 "分类器未调参"）
=================================================================

--part select  嵌套选择最佳配置（R1-3）
    外层 10 折 × 3 轮；每个外层训练集上，用内层 3 折 CV 评估全部 60 个默认管道
    （3 FS × 5 IMB × 4 CLF，与 v3 完全相同），按内层平均 MCC 选出一个管道，
    在外层测试折上评估。同时记录全部 60 个管道的外层结果，便于计算
    "事后挑最好"（oracle，即原 Table 3 的做法）与"嵌套选择"的差距 = 乐观偏差。

--part tune    轻量超参数调优（R2）
    FS=None，IMB ∈ {None, SMOTE, CS}，4 个分类器；外层 10 折 × 1 轮；
    内层 3 折 RandomizedSearchCV（每个分类器 20 组），以 AUC-PR 为调参目标；
    同一外层折上同时跑默认参数版本，便于配对比较"调参前 vs 调参后"，
    并检验调参后"不平衡处理 > 分类器 > 特征选择"的结论是否仍成立。
    SVM 调参时用 decision_function（不开 probability）以节省时间，
    选出参数后再以 probability=True 重新拟合。

运行（项目根目录）：
    python run_nested.py --part select --quick
    python run_nested.py --part tune --quick
    python run_nested.py --part select          # 预计约 1–2 小时
    python run_nested.py --part tune            # 预计约 1–2 小时
可中断续跑（按数据集写文件）。

输出：results_nested/select/*.csv、results_nested/tune/*.csv 及汇总
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import run_experiment_v3 as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_nested"
INNER = 3


# ----------------------------------------------------------------------------
# select：嵌套选择
# ----------------------------------------------------------------------------
def fit_predict(pipe, imb, Xtr, ytr, Xte, seed):
    """拟合并预测；THR 在训练集内部再做 3 折选阈值（与 v3 一致）"""
    thr = 0.5
    if imb == "THR":
        inner = StratifiedKFold(n_splits=base.THR_INNER_FOLDS, shuffle=True, random_state=seed)
        p_in = np.zeros(len(ytr))
        for a, b in inner.split(Xtr, ytr):
            p_in[b] = clone(pipe).fit(Xtr[a], ytr[a]).predict_proba(Xtr[b])[:, 1]
        thr = base.best_f1_threshold(ytr, p_in)
    try:
        pipe.fit(Xtr, ytr)
    except (ValueError, RuntimeError):
        return None, None
    prob = pipe.predict_proba(Xte)[:, 1]
    pred = (prob >= thr).astype(int) if imb == "THR" else pipe.predict(Xte)
    return pred, prob


def select_task(X, y, tr, te, fs, imb, clf, rep, fold):
    warnings.filterwarnings("ignore")
    seed = 42 + rep
    Xtr, ytr = X[tr], y[tr]
    pos = int(ytr.sum())

    def mk(n_pos, n_neg):
        return base.make_pipe(fs, imb, clf, X.shape[1], n_pos, seed, n_neg / max(1, n_pos))

    # 内层 CV：只用外层训练集
    inner = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=2000 + rep * 100 + fold)
    scores = []
    for a, b in inner.split(Xtr, ytr):
        p_ = int(ytr[a].sum())
        pred, _ = fit_predict(mk(p_, len(a) - p_), imb, Xtr[a], ytr[a], Xtr[b], seed)
        if pred is None:
            pred, _ = fit_predict(base.make_pipe(fs, "SMOTE", clf, X.shape[1], p_, seed, 1.0),
                                  "SMOTE", Xtr[a], ytr[a], Xtr[b], seed)
        scores.append(matthews_corrcoef(ytr[b], pred))
    pred, prob = fit_predict(mk(pos, len(tr) - pos), imb, Xtr, ytr, X[te], seed)
    if pred is None:
        pred, prob = fit_predict(base.make_pipe(fs, "SMOTE", clf, X.shape[1], pos, seed, 1.0),
                                 "SMOTE", Xtr, ytr, X[te], seed)
    row = {"FS": fs, "IMB": imb, "CLF": clf, "repeat": rep, "fold": fold,
           "inner_mcc": float(np.mean(scores))}
    row.update({f"outer_{k}": v for k, v in base.fold_metrics(y[te], pred, prob).items()})
    return row


def run_select(name, src, X, y, args):
    out = OUT / "select" / f"{name}.csv"
    if out.exists() and not args.overwrite:
        print(f"[跳过] select {name}")
        return
    tasks = []
    for rep in range(args.repeats):
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=1000 + rep)
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            for fs in base.FS_METHODS:
                for imb in base.IMB_METHODS:
                    for c in base.CLASSIFIERS:
                        tasks.append((tr, te, fs, imb, c, rep, fold))
    t0 = time.time()
    rows = Parallel(n_jobs=args.jobs, batch_size=1)(
        delayed(select_task)(X, y, *t) for t in tasks)
    df = pd.DataFrame(rows)
    df.insert(0, "Source", src)
    df.insert(0, "Dataset", name)
    df.to_csv(out, index=False)
    print(f"[完成] select {name}  用时 {time.time()-t0:.0f}s", flush=True)


def summarize_select():
    df = pd.concat([pd.read_csv(f) for f in sorted((OUT / "select").glob("*.csv"))])
    df["FS"] = df["FS"].fillna("None"); df["IMB"] = df["IMB"].fillna("None")
    df["pipeline"] = df.FS + "+" + df.IMB + "+" + df.CLF
    res = []
    for ds, g in df.groupby("Dataset"):
        # 嵌套：每个外层折选内层 MCC 最高的管道
        sel = g.loc[g.groupby(["repeat", "fold"])["inner_mcc"].idxmax()]
        # oracle：事后按外层平均 MCC 挑最好的固定管道（= 原 Table 3 做法）
        means = g.groupby("pipeline")[["outer_mcc", "outer_f1", "outer_auc_pr"]].mean()
        best = means["outer_mcc"].idxmax()
        res.append({"Dataset": ds, "Source": g.Source.iloc[0],
                    "oracle_pipeline": best,
                    "oracle_MCC": means.loc[best, "outer_mcc"],
                    "nested_MCC": sel.outer_mcc.mean(),
                    "optimism_MCC": means.loc[best, "outer_mcc"] - sel.outer_mcc.mean(),
                    "oracle_F1": means.loc[best, "outer_f1"],
                    "nested_F1": sel.outer_f1.mean(),
                    "oracle_AUCPR": means.loc[best, "outer_auc_pr"],
                    "nested_AUCPR": sel.outer_auc_pr.mean(),
                    "default_RF_CS_MCC": means.loc["None+CS+RF", "outer_mcc"],
                    "most_selected": sel.pipeline.value_counts().index[0],
                    "most_selected_share": sel.pipeline.value_counts(normalize=True).iloc[0],
                    "share_selected_with_imbalance": (sel.IMB != "None").mean(),
                    "share_selected_ensemble": sel.CLF.isin(["RF", "XGB"]).mean(),
                    "share_selected_with_FS": (sel.FS != "None").mean()})
    s = pd.DataFrame(res)
    s.to_csv(OUT / "select_summary.csv", index=False)
    print("\n=== 嵌套选择 vs 事后挑选（oracle）===")
    print(s.round(3).to_string(index=False))
    print("\n平均乐观偏差（MCC）：", round(s.optimism_MCC.mean(), 4),
          "  中位数：", round(s.optimism_MCC.median(), 4))


# ----------------------------------------------------------------------------
# tune：轻量调参
# ----------------------------------------------------------------------------
SPACES = {
    "LR": {"clf__C": loguniform(1e-3, 1e2)},
    "SVM": {"clf__C": loguniform(1e-2, 1e2), "clf__gamma": loguniform(1e-4, 1e0)},
    "RF": {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5, 10, 20],
           "clf__min_samples_leaf": [1, 2, 5, 10], "clf__max_features": ["sqrt", "log2", 0.5]},
    "XGB": {"clf__n_estimators": [100, 300], "clf__max_depth": [3, 4, 6, 8],
            "clf__learning_rate": loguniform(1e-2, 3e-1), "clf__subsample": uniform(0.6, 0.4),
            "clf__colsample_bytree": uniform(0.6, 0.4), "clf__min_child_weight": [1, 3, 5]},
}


def tune_task(X, y, tr, te, imb, clf, rep, fold, n_iter):
    warnings.filterwarnings("ignore")
    seed = 42 + rep
    Xtr, ytr = X[tr], y[tr]
    pos = int(ytr.sum()); neg = len(tr) - pos
    rows = []
    # 默认参数
    pipe = base.make_pipe("None", imb, clf, X.shape[1], pos, seed, neg / pos)
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    r = {"IMB": imb, "CLF": clf, "repeat": rep, "fold": fold, "tuned": False,
         "fit_time": time.perf_counter() - t0, "best_params": ""}
    r.update(base.fold_metrics(y[te], pipe.predict(X[te]), pipe.predict_proba(X[te])[:, 1]))
    rows.append(r)
    # 调参
    search_pipe = base.make_pipe("None", imb, clf, X.shape[1], pos, seed, neg / pos)
    if clf == "SVM":
        search_pipe.set_params(clf__probability=False)
    inner = StratifiedKFold(n_splits=INNER, shuffle=True, random_state=3000 + fold)
    rs = RandomizedSearchCV(search_pipe, SPACES[clf], n_iter=n_iter, scoring="average_precision",
                            cv=inner, random_state=seed, n_jobs=1, refit=False, error_score=np.nan)
    t0 = time.perf_counter()
    rs.fit(Xtr, ytr)
    best = rs.best_params_
    final = base.make_pipe("None", imb, clf, X.shape[1], pos, seed, neg / pos)
    final.set_params(**best)
    final.fit(Xtr, ytr)
    r = {"IMB": imb, "CLF": clf, "repeat": rep, "fold": fold, "tuned": True,
         "fit_time": time.perf_counter() - t0,
         "best_params": str({k.replace("clf__", ""): (round(v, 5) if isinstance(v, float) else v)
                             for k, v in best.items()})}
    r.update(base.fold_metrics(y[te], final.predict(X[te]), final.predict_proba(X[te])[:, 1]))
    rows.append(r)
    return rows


def run_tune(name, src, X, y, args):
    out = OUT / "tune" / f"{name}.csv"
    if out.exists() and not args.overwrite:
        print(f"[跳过] tune {name}")
        return
    tasks = []
    for rep in range(args.repeats):
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=1000 + rep)
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            for imb in ["None", "SMOTE", "CS"]:
                for c in base.CLASSIFIERS:
                    tasks.append((tr, te, imb, c, rep, fold))
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, batch_size=1)(
        delayed(tune_task)(X, y, *t, args.n_iter) for t in tasks)
    df = pd.DataFrame([r for rs in res for r in rs])
    df.insert(0, "Source", src)
    df.insert(0, "Dataset", name)
    df.to_csv(out, index=False)
    print(f"[完成] tune {name}  用时 {time.time()-t0:.0f}s", flush=True)


def summarize_tune():
    df = pd.concat([pd.read_csv(f) for f in sorted((OUT / "tune").glob("*.csv"))])
    df["IMB"] = df["IMB"].fillna("None")
    dm = df.groupby(["Dataset", "tuned", "IMB", "CLF"])[["mcc", "auc_pr", "f1", "auc"]].mean().reset_index()
    dm.to_csv(OUT / "tune_per_dataset.csv", index=False)
    print("\n=== 调参前后：分类器（在 IMB 上平均，24 数据集平均）===")
    print(dm.groupby(["tuned", "CLF"])[["mcc", "auc_pr", "f1", "auc"]].mean().unstack(0).round(3).to_string())
    print("\n=== 调参前后：不平衡处理（在分类器上平均）===")
    print(dm.groupby(["tuned", "IMB"])[["mcc", "auc_pr", "f1", "auc"]].mean().unstack(0).round(3).to_string())
    # 每个数据集上分类器排名（按 AUC-PR）在调参前后是否变化
    rk = (dm.groupby(["Dataset", "tuned", "CLF"])["auc_pr"].mean()
          .groupby(["Dataset", "tuned"]).rank(ascending=False).unstack("tuned"))
    rk.columns = ["rank_default", "rank_tuned"]
    rk.reset_index().to_csv(OUT / "tune_classifier_ranks.csv", index=False)
    print("\n平均排名（AUC-PR，1=最好）：")
    print(rk.groupby("CLF").mean().round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["select", "tune"], required=True)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--n-iter", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.repeats is None:
        args.repeats = 3 if args.part == "select" else 1
    global OUT
    if args.quick:
        OUT = ROOT / "results_nested_quick"
        args.repeats, args.datasets, args.overwrite, args.folds, args.n_iter = 1, ["MC2", "EQ"], True, 3, 4
    (OUT / args.part).mkdir(parents=True, exist_ok=True)

    items = base.discover_datasets(False)
    if args.datasets:
        want = {d.lower() for d in args.datasets}
        items = [it for it in items if it[0].lower() in want]
    t0 = time.time()
    for name, src, path, dedup in items:
        Xdf, ys, _, _ = base.load_arff(path, dedup)
        X, y = Xdf.values, ys.values.astype(int)
        (run_select if args.part == "select" else run_tune)(name, src, X, y, args)
    (summarize_select if args.part == "select" else summarize_tune)()
    print(f"\n总用时 {(time.time()-t0)/60:.1f} 分钟，输出在 {OUT}")


if __name__ == "__main__":
    main()
