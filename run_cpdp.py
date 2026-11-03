"""
跨项目缺陷预测（CPDP）小实验（回应 R2 "缺少跨项目验证"）
========================================================

设置：在 AEEEM（5 个项目，共同度量）和 JIRA（7 个项目，共同度量）内部分别做
"留一项目"（leave-one-project-out）：用其余项目合并作训练（源），目标项目作测试。
- 每个项目先在自身内部做 z-score 标准化（CPDP 常用做法，消除项目间尺度差异）
- 管道与 v3 相同：3 FS × 5 IMB（含 THR）× 4 CLF = 60；THR 的阈值只在源数据上用内部 3 折选
- 不同随机种子重复 5 次（--seeds）
- 外部数据与 v3 一致：去除完全重复行；只保留各项目共有且非常数的列

运行（项目根目录）：
    python run_cpdp.py --quick       # 冒烟测试（AEEEM，1 个种子）
    python run_cpdp.py               # 全量，预计 20–40 分钟
输出：results_cpdp/cpdp_results.csv、results_cpdp/summary_*.csv
"""

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")

import run_experiment_v3 as base

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results_cpdp"


def load_group(src):
    projs = {}
    for p in sorted((ROOT / "data_ext" / src).glob("*.arff")):
        X, y, _, _ = base.load_arff(p, dedup=True)
        projs[p.stem] = (X, y)
    common = sorted(set.intersection(*[set(X.columns) for X, _ in projs.values()]))
    out = {}
    for k, (X, y) in projs.items():
        Z = X[common].astype(float)
        sd = Z.std(ddof=0).replace(0, 1.0)
        out[k] = (((Z - Z.mean()) / sd).values, y.values.astype(int))
    return out, common


def task(Xs, ys, Xt, yt, fs, imb, clf, seed):
    warnings.filterwarnings("ignore")
    pos = int(ys.sum()); neg = len(ys) - pos
    pipe = base.make_pipe(fs, imb, clf, Xs.shape[1], pos, seed, neg / pos)
    thr = 0.5
    t0 = time.perf_counter()
    if imb == "THR":
        inner = StratifiedKFold(n_splits=base.THR_INNER_FOLDS, shuffle=True, random_state=seed)
        p_in = np.zeros(len(ys))
        for a, b in inner.split(Xs, ys):
            p_in[b] = clone(pipe).fit(Xs[a], ys[a]).predict_proba(Xs[b])[:, 1]
        thr = base.best_f1_threshold(ys, p_in)
    note = ""
    try:
        pipe.fit(Xs, ys)
    except (ValueError, RuntimeError) as e:
        note = f"fallback_SMOTE:{type(e).__name__}"
        pipe = base.make_pipe(fs, "SMOTE", clf, Xs.shape[1], pos, seed, 1.0).fit(Xs, ys)
    fit_t = time.perf_counter() - t0
    prob = pipe.predict_proba(Xt)[:, 1]
    pred = (prob >= thr).astype(int) if imb == "THR" else pipe.predict(Xt)
    row = {"FS": fs, "IMB": imb, "CLF": clf, "seed": seed, "threshold": thr,
           "source_pos_rate": pos / len(ys), "fit_time": fit_t, "note": note}
    row.update(base.fold_metrics(yt, pred, prob))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    groups = ["AEEEM", "JIRA"]
    global OUT
    if args.quick:
        OUT = ROOT / "results_cpdp_quick"
        args.seeds, groups = 1, ["AEEEM"]
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    t_all = time.time()
    for g in groups:
        projs, cols = load_group(g)
        print(f"[{g}] 项目数={len(projs)}  共同特征数={len(cols)}")
        for target in projs:
            Xt, yt = projs[target]
            Xs = np.vstack([projs[k][0] for k in projs if k != target])
            ys = np.concatenate([projs[k][1] for k in projs if k != target])
            t0 = time.time()
            res = Parallel(n_jobs=args.jobs, batch_size=1)(
                delayed(task)(Xs, ys, Xt, yt, fs, imb, c, 42 + s)
                for s in range(args.seeds)
                for fs in base.FS_METHODS for imb in base.IMB_METHODS for c in base.CLASSIFIERS)
            for r in res:
                r.update({"Group": g, "Target": target, "n_source": len(ys), "n_target": len(yt),
                          "target_pos_rate": yt.mean()})
            rows.extend(res)
            m = pd.DataFrame(res).groupby("IMB")[["f1", "mcc", "auc_pr"]].mean().round(3)
            print(f"  目标={target:22s} 源样本={len(ys):5d}  用时 {time.time()-t0:.0f}s\n{m.to_string()}", flush=True)

    df = pd.DataFrame(rows)
    front = ["Group", "Target", "FS", "IMB", "CLF", "seed"]
    df = df[front + [c for c in df.columns if c not in front]]
    df.to_csv(OUT / "cpdp_results.csv", index=False)
    per_t = df.groupby(["Group", "Target", "IMB"])[["f1", "mcc", "auc_pr", "auc"]].mean().reset_index()
    for f in ["IMB", "CLF", "FS"]:
        s = (df.groupby(["Target", f])[["f1", "mcc", "auc_pr", "auc"]].mean()
             .groupby(f).agg(["mean", "std"]).round(3))
        s.to_csv(OUT / f"summary_{f}.csv")
        print(f"\n=== CPDP：按 {f}（目标项目层面平均）===")
        print(s.to_string())
    per_t.to_csv(OUT / "per_target_imb.csv", index=False)
    print(f"\n总用时 {(time.time()-t_all)/60:.1f} 分钟，输出在 {OUT}")


if __name__ == "__main__":
    main()
