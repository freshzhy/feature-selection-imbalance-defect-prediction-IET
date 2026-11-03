"""
软件缺陷预测实验脚本 v3（IET Software 大修用）
=================================================

相对 v2 的变化
--------------
1. 数据：NASA D'' 12 个 + AEEEM 5 个 + JIRA 7 个 = 24 个数据集
   - 外部数据默认去除完全重复行（与 D'' 清洗口径一致），可用 --keep-duplicates 关闭
2. 交叉验证：10×10 重复分层 CV（--repeats 可调），保存【逐折】结果
3. 不平衡处理新增第 5 种：THR（阈值移动）
   - 不做重采样/加权，在训练折内部再做 3 折 CV 得到训练集 OOF 概率，
     选使 F1 最大的阈值，再用于测试折 —— 用于检验"不平衡处理≈移动阈值"
4. 保存样本级 OOF 预测概率（results_v3/oof/<数据集>.npz），供阈值/机制分析
5. 记录 fit/predict 时间、模型序列化大小、实际使用特征数（回应效率质疑）
6. 新增指标：G-mean、PF（误报率）、Balanced Accuracy、Brier 分数
7. 代码中 v2 的 "CFS" 名称实际是 ANOVA F-test，统一改名为 "AF"

运行方式（在项目根目录）
------------------------
    python run_experiment_v3.py --quick        # 冒烟测试：2 个小数据集、1 轮，约 1 分钟
    python run_experiment_v3.py                # 全量：24 数据集 × 60 管道 × 10×10 CV
    python run_experiment_v3.py --jobs 16      # 指定并行进程数（默认 = CPU 核数 - 2）
    python run_experiment_v3.py --datasets CM1 EQ hbase-0.94.0   # 只跑指定数据集

可中断续跑：每个数据集跑完立即写盘，再次运行会跳过已完成的数据集
（加 --overwrite 强制重跑）。

输出
----
    results_v3/folds/<数据集>.csv      逐折结果（每行 = 数据集×管道×轮次×折）
    results_v3/oof/<数据集>.npz        OOF 概率/预测、真实标签、折划分
    results_v3/datasets_summary.csv    数据集特征表（新 Table 1 用）
    results_v3/all_folds.csv           全部逐折结果合并（全部跑完后自动生成）
    results_v3/run_info.json           软件版本、硬件、参数（复现性说明用）
"""

import argparse
import json
import os
import pickle
import platform
import sys
import time
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import arff

from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             matthews_corrcoef, roc_auc_score,
                             average_precision_score, confusion_matrix,
                             brier_score_loss, balanced_accuracy_score)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
# 让并行子进程也屏蔽 sklearn 1.9 对 SVC(probability=True) 的 FutureWarning 刷屏
os.environ.setdefault("PYTHONWARNINGS", "ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_v3"

FS_METHODS = ["None", "IG", "AF"]
IMB_METHODS = ["None", "SMOTE", "ADASYN", "CS", "THR"]
CLASSIFIERS = ["LR", "SVM", "RF", "XGB"]

SMOTE_K = 3            # 与 v2 一致；敏感性分析另见 run_sensitivity.py
FS_RATIO = 0.5         # 保留一半特征，与 v2 一致
THR_INNER_FOLDS = 3    # 阈值移动：训练折内部 CV 折数
POS_LABELS = {"y", "yes", "true", "buggy", "1", "defective", "t"}


# ----------------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------------
def load_arff(path, dedup):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for c in df.select_dtypes([object]).columns:
        df[c] = df[c].str.decode("utf-8")
    target = df.columns[-1]
    if meta.types()[-1] == "nominal":
        y = df[target].astype(str).str.strip().str.lower().isin(POS_LABELS).astype(int)
    else:  # 计数型标签（JIRA: RealBugCount）
        y = (pd.to_numeric(df[target]) > 0).astype(int)
    X = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
    keep = ~X.isna().any(axis=1)
    X, y = X[keep], y[keep]
    n_dup = 0
    if dedup:
        full = pd.concat([X, y.rename("__y__")], axis=1)
        mask = ~full.duplicated()
        n_dup = int((~mask).sum())
        X, y = X[mask], y[mask]
    # 去掉常数列（在全部样本上都不变的列不含信息，也会让 F 检验报警）
    const = X.columns[X.nunique() <= 1]
    X = X.drop(columns=const)
    return X.reset_index(drop=True).astype(float), y.reset_index(drop=True), n_dup, len(const)


def discover_datasets(keep_duplicates):
    """返回 [(名称, 来源, 路径, 是否去重)]"""
    items = []
    for p in sorted((ROOT / "data").glob("*.arff")):
        if p.stem.upper() == "KC4" or p.stat().st_size < 200:
            continue
        items.append((p.stem, "NASA", p, False))      # D'' 版已清洗
    for src in ["AEEEM", "JIRA"]:
        for p in sorted((ROOT / "data_ext" / src).glob("*.arff")):
            items.append((p.stem, src, p, not keep_duplicates))
    return items


# ----------------------------------------------------------------------------
# 管道构建
# ----------------------------------------------------------------------------
def make_clf(name, imb, seed, spw=1.0):
    cw = "balanced" if imb == "CS" else None
    if name == "LR":
        return LogisticRegression(max_iter=1000, random_state=seed, class_weight=cw)
    if name == "SVM":
        return SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                   random_state=seed, class_weight=cw)
    if name == "RF":
        return RandomForestClassifier(n_estimators=100, random_state=seed,
                                      class_weight=cw, n_jobs=1)
    if name == "XGB":
        return XGBClassifier(n_estimators=100, eval_metric="logloss",
                             random_state=seed, verbosity=0, n_jobs=1,
                             scale_pos_weight=(spw if imb == "CS" else 1.0))
    raise ValueError(name)


def make_pipe(fs, imb, clf_name, n_feat, n_min_train, seed, spw):
    steps = [("scaler", StandardScaler())]
    k = max(1, int(n_feat * FS_RATIO))
    if fs == "IG":
        steps.append(("fs", SelectKBest(partial(mutual_info_classif, random_state=seed), k=k)))
    elif fs == "AF":
        steps.append(("fs", SelectKBest(f_classif, k=k)))
    kn = max(1, min(SMOTE_K, n_min_train - 1))
    if imb == "SMOTE":
        steps.append(("sampler", SMOTE(k_neighbors=kn, random_state=seed)))
    elif imb == "ADASYN":
        steps.append(("sampler", ADASYN(n_neighbors=kn, random_state=seed)))
    steps.append(("clf", make_clf(clf_name, imb, seed, spw)))
    return ImbPipeline(steps)


def best_f1_threshold(y, p):
    """在训练集 OOF 概率上找使 F1 最大的阈值"""
    order = np.argsort(-p)
    ys, ps = y[order], p[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    P = ys.sum()
    if P == 0:
        return 0.5
    f1 = 2 * tp / (tp + fp + P)
    # 只在概率值变化处取阈值
    distinct = np.r_[np.diff(ps) != 0, True]
    i = np.argmax(np.where(distinct, f1, -1))
    nxt = ps[i + 1] if i + 1 < len(ps) else ps[i] - 1e-6
    return float((ps[i] + nxt) / 2)


def fold_metrics(y, pred, prob):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    two = len(np.unique(y)) == 2
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": rec,
        "f1": f1_score(y, pred, zero_division=0),
        "mcc": matthews_corrcoef(y, pred),
        "auc": roc_auc_score(y, prob) if two else np.nan,
        "auc_pr": average_precision_score(y, prob) if two else np.nan,
        "gmean": float(np.sqrt(rec * spec)),
        "pf": fp / (fp + tn) if fp + tn else 0.0,
        "bal_acc": balanced_accuracy_score(y, pred),
        "brier": brier_score_loss(y, prob),
        "n_pred_pos": int(tp + fp),
    }


# ----------------------------------------------------------------------------
# 单个任务：一个数据集上的一个管道 × 一轮 10 折
# ----------------------------------------------------------------------------
def run_task(X, y, fs, imb, clf_name, rep, n_folds):
    warnings.filterwarnings("ignore")
    seed = 42 + rep
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=1000 + rep)
    n = len(y)
    oof_prob = np.full(n, np.nan, dtype=np.float32)
    oof_pred = np.full(n, -1, dtype=np.int8)
    rows = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
        pos, neg = int(ytr.sum()), int(len(ytr) - ytr.sum())
        spw = neg / pos if pos else 1.0
        note = ""
        pipe = make_pipe(fs, imb, clf_name, X.shape[1], pos, seed, spw)

        t0 = time.perf_counter()
        thr = 0.5
        if imb == "THR":
            inner = StratifiedKFold(n_splits=THR_INNER_FOLDS, shuffle=True, random_state=seed)
            p_in = np.zeros(len(ytr))
            for itr, ite in inner.split(Xtr, ytr):
                m = clone(pipe).fit(Xtr[itr], ytr[itr])
                p_in[ite] = m.predict_proba(Xtr[ite])[:, 1]
            thr = best_f1_threshold(ytr, p_in)
        try:
            pipe.fit(Xtr, ytr)
        except (ValueError, RuntimeError) as e:   # ADASYN 偶发"无样本可生成"
            note = f"fallback_SMOTE:{type(e).__name__}"
            pipe = make_pipe(fs, "SMOTE", clf_name, X.shape[1], pos, seed, spw)
            pipe.fit(Xtr, ytr)
        fit_t = time.perf_counter() - t0

        t0 = time.perf_counter()
        prob = pipe.predict_proba(Xte)[:, 1]
        pred = (prob >= thr).astype(int) if imb == "THR" else pipe.predict(Xte)
        pred_t = time.perf_counter() - t0

        oof_prob[te] = prob
        oof_pred[te] = pred
        n_used = pipe.named_steps["fs"].get_support().sum() if "fs" in pipe.named_steps else X.shape[1]
        row = {"FS": fs, "IMB": imb, "CLF": clf_name, "repeat": rep, "fold": fold,
               "n_train": len(ytr), "n_test": len(yte), "train_pos_rate": pos / len(ytr),
               "threshold": thr, "n_features_used": int(n_used),
               "fit_time": fit_t, "predict_time": pred_t,
               "model_bytes": len(pickle.dumps(pipe)), "note": note}
        row.update(fold_metrics(yte, pred, prob))
        rows.append(row)
    return rows, (fs, imb, clf_name, rep, oof_prob, oof_pred)


def run_dataset(name, src, path, dedup, args):
    fold_csv = OUT / "folds" / f"{name}.csv"
    oof_npz = OUT / "oof" / f"{name}.npz"
    Xdf, yser, n_dup, n_const = load_arff(path, dedup)
    X, y = Xdf.values, yser.values.astype(int)
    info = {"Dataset": name, "Source": src, "Modules": len(y), "Features": X.shape[1],
            "Defective": int(y.sum()), "DefectRate%": round(100 * y.mean(), 2),
            "IR": round((len(y) - y.sum()) / max(1, y.sum()), 2),
            "DuplicatesRemoved": n_dup, "ConstantColsRemoved": n_const}
    if fold_csv.exists() and oof_npz.exists() and not args.overwrite:
        print(f"[跳过] {name} 已完成")
        return info

    tasks = [(fs, imb, c, r) for r in range(args.repeats)
             for fs in FS_METHODS for imb in IMB_METHODS for c in CLASSIFIERS]
    print(f"[开始] {name} ({src})  n={len(y)}  p={X.shape[1]}  缺陷率={100*y.mean():.1f}%  "
          f"任务数={len(tasks)}", flush=True)
    t0 = time.time()
    res = Parallel(n_jobs=args.jobs, verbose=0, batch_size=1)(
        delayed(run_task)(X, y, fs, imb, c, r, args.folds) for fs, imb, c, r in tasks)

    rows, oof = [], {}
    for r_rows, (fs, imb, c, rep, p, pr) in res:
        for row in r_rows:
            row["Dataset"] = name
            row["Source"] = src
        rows.extend(r_rows)
        oof[f"prob|{fs}|{imb}|{c}|{rep}"] = p
        oof[f"pred|{fs}|{imb}|{c}|{rep}"] = pr
    # 保存折划分（便于事后复现每个样本属于哪一折）
    for rep in range(args.repeats):
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=1000 + rep)
        fid = np.zeros(len(y), dtype=np.int8)
        for k, (_, te) in enumerate(cv.split(X, y)):
            fid[te] = k
        oof[f"fold|{rep}"] = fid
    oof["y"] = y.astype(np.int8)

    df = pd.DataFrame(rows)
    front = ["Dataset", "Source", "FS", "IMB", "CLF", "repeat", "fold"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(fold_csv, index=False)
    np.savez_compressed(oof_npz, **oof)
    m = df.groupby("IMB")[["f1", "mcc", "auc_pr"]].mean().round(3)
    print(f"[完成] {name}  用时 {time.time()-t0:.0f}s\n{m.to_string()}\n", flush=True)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quick", action="store_true", help="冒烟测试：MC2 + EQ，1 轮")
    args = ap.parse_args()

    if args.quick:
        global OUT
        OUT = ROOT / "results_v3_quick"
        args.repeats, args.datasets, args.overwrite = 1, ["MC2", "EQ"], True
    (OUT / "folds").mkdir(parents=True, exist_ok=True)
    (OUT / "oof").mkdir(parents=True, exist_ok=True)

    items = discover_datasets(args.keep_duplicates)
    if args.datasets:
        want = {d.lower() for d in args.datasets}
        items = [it for it in items if it[0].lower() in want]
    if not items:
        sys.exit("未找到数据集：请确认 data/ 与 data_ext/AEEEM、data_ext/JIRA 存在")

    import sklearn, imblearn, xgboost, scipy
    run_info = {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0], "platform": platform.platform(),
        "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "jobs": args.jobs,
        "versions": {"scikit-learn": sklearn.__version__, "imbalanced-learn": imblearn.__version__,
                     "xgboost": xgboost.__version__, "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "repeats": args.repeats, "folds": args.folds, "smote_k": SMOTE_K,
        "fs_ratio": FS_RATIO, "thr_inner_folds": THR_INNER_FOLDS,
        "dedup_external": not args.keep_duplicates,
        "datasets": [it[0] for it in items],
    }
    if platform.system() == "Darwin":
        run_info["macos"] = platform.mac_ver()[0]
    print(json.dumps(run_info, ensure_ascii=False, indent=2))

    t0 = time.time()
    infos = [run_dataset(n, s, p, d, args) for n, s, p, d in items]
    pd.DataFrame(infos).to_csv(OUT / "datasets_summary.csv", index=False)

    parts = [pd.read_csv(f) for f in sorted((OUT / "folds").glob("*.csv"))]
    if parts:
        pd.concat(parts).to_csv(OUT / "all_folds.csv", index=False)
    run_info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_info["wall_clock_minutes"] = round((time.time() - t0) / 60, 1)
    (OUT / "run_info.json").write_text(json.dumps(run_info, ensure_ascii=False, indent=2))
    print(f"全部完成，总用时 {run_info['wall_clock_minutes']} 分钟。结果在 {OUT}")


if __name__ == "__main__":
    main()
