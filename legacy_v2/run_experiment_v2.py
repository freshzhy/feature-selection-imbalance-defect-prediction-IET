"""
软件缺陷预测实验脚本 v2
特征选择 × 类别不平衡处理 × 分类器 全组合对比实验

v2 变更:
- 新增第四种不平衡处理: class_weight (代价敏感学习)
- 新增 MCC 和 AUC-PR 指标
- 实验从 36 条管道扩展为 48 条 (3 FS × 4 imbalance × 4 classifier)
"""

import sys

import os, warnings
import numpy as np
import pandas as pd
from scipy.io import arff

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (make_scorer, precision_score, recall_score,
                             f1_score, roc_auc_score, matthews_corrcoef,
                             average_precision_score)
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)

SKIP = {"KC4"}

def make_classifiers(imb_name):
    """Return classifier dict; for 'CostSensitive', use class_weight/scale_pos_weight."""
    if imb_name == "CostSensitive":
        return {
            "LR":  LogisticRegression(max_iter=1000, random_state=42,
                                      class_weight="balanced"),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42,
                       class_weight="balanced"),
            "RF":  RandomForestClassifier(n_estimators=100, random_state=42,
                                          class_weight="balanced"),
            "XGB": None,  # handled specially per fold
        }
    else:
        return {
            "LR":  LogisticRegression(max_iter=1000, random_state=42),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
            "RF":  RandomForestClassifier(n_estimators=100, random_state=42),
            "XGB": XGBClassifier(n_estimators=100, use_label_encoder=False,
                                 eval_metric="logloss", random_state=42, verbosity=0),
        }

FS_METHODS = ["None", "IG", "CFS"]

IMBALANCE_METHODS = ["None", "SMOTE", "ADASYN", "CostSensitive"]

def make_fs(name, k):
    if name == "None":
        return None
    if name == "IG":
        return SelectKBest(mutual_info_classif, k=k)
    if name == "CFS":
        return SelectKBest(f_classif, k=k)

def mcc_scorer(y_true, y_pred):
    return matthews_corrcoef(y_true, y_pred)

SCORING = {
    "precision": make_scorer(precision_score, zero_division=0),
    "recall":    make_scorer(recall_score,    zero_division=0),
    "f1":        make_scorer(f1_score,        zero_division=0),
    "auc":       make_scorer(roc_auc_score,   response_method="predict_proba"),
    "mcc":       make_scorer(mcc_scorer),
    "auc_pr":    make_scorer(average_precision_score, response_method="predict_proba"),
}

CV = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)


def load_arff(path):
    data, meta = arff.loadarff(path)
    df = pd.DataFrame(data)
    for col in df.select_dtypes([object]):
        df[col] = df[col].str.decode("utf-8")
    target_col = df.columns[-1]
    df[target_col] = (df[target_col].str.upper() == "Y").astype(int)
    df = df.dropna()
    X = df.drop(columns=[target_col]).astype(float)
    y = df[target_col]
    return X, y


def get_sampler(imb_name):
    if imb_name == "SMOTE":
        return SMOTE(random_state=42, k_neighbors=3)
    elif imb_name == "ADASYN":
        return ADASYN(random_state=42, n_neighbors=3)
    return None


def run_one(X, y, fs_name, imb_name, clf_name, clf):
    k = max(1, X.shape[1] // 2)
    fs = make_fs(fs_name, k)
    sampler = get_sampler(imb_name)

    steps = [("scaler", StandardScaler())]
    if fs:
        steps.append(("fs", fs))
    if sampler:
        steps.append(("sampler", sampler))
    steps.append(("clf", clf))

    pipe = ImbPipeline(steps)
    scores = cross_validate(pipe, X, y, cv=CV, scoring=SCORING,
                            error_score="raise", n_jobs=-1)
    return {
        "precision": np.nanmean(scores["test_precision"]),
        "recall":    np.nanmean(scores["test_recall"]),
        "f1":        np.nanmean(scores["test_f1"]),
        "auc":       np.nanmean(scores["test_auc"]),
        "mcc":       np.nanmean(scores["test_mcc"]),
        "auc_pr":    np.nanmean(scores["test_auc_pr"]),
    }


def run_one_xgb_costsensitive(X, y, fs_name):
    """XGBoost with scale_pos_weight computed per-fold."""
    k = max(1, X.shape[1] // 2)
    fs = make_fs(fs_name, k)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    fold_metrics = {m: [] for m in ["precision", "recall", "f1", "auc", "mcc", "auc_pr"]}

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        if fs:
            from sklearn.base import clone
            fs_clone = clone(fs)
            X_train_s = fs_clone.fit_transform(X_train_s, y_train)
            X_test_s  = fs_clone.transform(X_test_s)

        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        spw = neg / pos if pos > 0 else 1.0

        clf = XGBClassifier(n_estimators=100, use_label_encoder=False,
                            eval_metric="logloss", random_state=42,
                            verbosity=0, scale_pos_weight=spw)
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        y_proba = clf.predict_proba(X_test_s)[:, 1]

        fold_metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
        fold_metrics["auc"].append(roc_auc_score(y_test, y_proba))
        fold_metrics["mcc"].append(matthews_corrcoef(y_test, y_pred))
        fold_metrics["auc_pr"].append(average_precision_score(y_test, y_proba))

    return {k: np.nanmean(v) for k, v in fold_metrics.items()}


def main():
    records = []
    arff_files = [f for f in os.listdir(DATA_DIR)
                  if f.endswith(".arff") and os.path.splitext(f)[0] not in SKIP
                  and os.path.getsize(os.path.join(DATA_DIR, f)) > 200]

    total = len(arff_files) * len(FS_METHODS) * len(IMBALANCE_METHODS) * 4
    done = 0

    for fname in sorted(arff_files):
        ds_name = os.path.splitext(fname)[0]
        path = os.path.join(DATA_DIR, fname)
        print(f"\n{'='*55}")
        print(f"数据集: {ds_name}")

        try:
            X, y = load_arff(path)
        except Exception as e:
            print(f"  加载失败: {e}")
            continue

        defect_rate = y.mean() * 100
        print(f"样本: {len(y)}  特征: {X.shape[1]}  缺陷率: {defect_rate:.1f}%")

        min_class = y.value_counts().min()

        for fs_name in FS_METHODS:
            for imb_name in IMBALANCE_METHODS:
                if imb_name == "ADASYN" and min_class < 10:
                    done += 4
                    continue

                classifiers = make_classifiers(imb_name)

                for clf_name in ["LR", "SVM", "RF", "XGB"]:
                    clf = classifiers[clf_name]
                    tag = f"  [{fs_name}+{imb_name}+{clf_name}]"

                    try:
                        if imb_name == "CostSensitive" and clf_name == "XGB":
                            m = run_one_xgb_costsensitive(X, y, fs_name)
                        else:
                            m = run_one(X, y, fs_name, imb_name, clf_name, clf)

                        records.append({
                            "Dataset": ds_name,
                            "FS": fs_name if fs_name != "None" else None,
                            "Imbalance": imb_name if imb_name != "None" else None,
                            "Classifier": clf_name,
                            "Precision": round(float(m["precision"]), 4),
                            "Recall":    round(float(m["recall"]),    4),
                            "F1":        round(float(m["f1"]),        4),
                            "AUC":       round(float(m["auc"]),       4) if not np.isnan(m["auc"]) else None,
                            "MCC":       round(float(m["mcc"]),       4),
                            "AUC_PR":    round(float(m["auc_pr"]),    4) if not np.isnan(m["auc_pr"]) else None,
                        })
                        done += 1
                        print(f"{tag}  F1={m['f1']:.3f}  AUC={m['auc']:.3f}"
                              f"  MCC={m['mcc']:.3f}  AUC-PR={m['auc_pr']:.3f}"
                              f"  [{done}/{total}]")
                    except Exception as e:
                        done += 1
                        print(f"{tag}  SKIP ({e})")

    df_all = pd.DataFrame(records)
    csv_path  = os.path.join(OUT_DIR, "all_results_v2.csv")
    xlsx_path = os.path.join(OUT_DIR, "all_results_v2.xlsx")
    df_all.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_all.to_excel(writer, sheet_name="All", index=False)

        for ds in df_all["Dataset"].unique():
            df_all[df_all["Dataset"] == ds].to_excel(
                writer, sheet_name=ds[:31], index=False)

        best = (df_all.sort_values("F1", ascending=False)
                      .groupby("Dataset").first().reset_index())
        best.to_excel(writer, sheet_name="BestPerDataset", index=False)

    print(f"\n{'='*55}")
    print(f"实验完成！结果已保存：")
    print(f"  CSV  → {csv_path}")
    print(f"  Excel→ {xlsx_path}")
    print(f"总记录数：{len(df_all)}")


if __name__ == "__main__":
    main()
