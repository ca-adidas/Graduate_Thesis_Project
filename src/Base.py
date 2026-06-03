#!/usr/bin/env python
# coding: utf-8

# Converted from Base.ipynb after analysis-oriented cell regrouping

# %% [markdown]
# # SECOM 데이터 분석 파이프라인
#
# 실험 환경 설정부터 데이터 로드, 전처리, 모델링, 평가, 시각화까지 분석 흐름 순서로 셀을 재구성했습니다.

# %%
import os
import tempfile
import ast


def get_env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


CPU_THREADS = get_env_int("SECOM_CPU_THREADS", os.cpu_count() or 1)
OUTER_N_JOBS = get_env_int("SECOM_OUTER_N_JOBS", min(9, CPU_THREADS))
MODEL_N_JOBS = get_env_int("SECOM_MODEL_N_JOBS", max(1, CPU_THREADS // max(1, OUTER_N_JOBS)))
BLAS_THREADS = get_env_int("SECOM_BLAS_THREADS", MODEL_N_JOBS)

for thread_env in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
]:
    os.environ.setdefault(thread_env, str(BLAS_THREADS))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    SelectKBest, chi2, f_classif, mutual_info_classif,
    SelectFpr, SelectFdr, SelectFwe
)
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import (
    f1_score, fbeta_score, roc_auc_score, roc_curve,
    confusion_matrix, precision_score, recall_score
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import warnings
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("xgboost 설치하세요.") from e

def resolve_base_dir():
    search_roots = [Path.cwd(), *Path.cwd().parents, Path.home() / "secom_project" / "data2"]

    for root in search_roots:
        if (root / "kfold" / "fold1_train.xlsx").exists():
            return root

        for candidate in root.glob("**/kfold/fold1_train.xlsx"):
            return candidate.parent.parent

    raise FileNotFoundError(
        "Could not find kfold/fold1_train.xlsx. "
        "Run the notebook from the project directory or add the data path to resolve_base_dir()."
    )


base_dir = resolve_base_dir()

# Use system temp directory with ASCII-only path to avoid joblib UnicodeEncodeError
# joblib's resource tracker cannot handle paths with non-ASCII characters
joblib_temp_dir = Path(tempfile.gettempdir()) / "secom_joblib_tmp"
joblib_temp_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(joblib_temp_dir))

raw_dir = base_dir / "kfold"
inpaintknn_dir = base_dir / "kfold_inpaintknn"
mice_dir = base_dir / "kfold_mice"
missforest_dir = base_dir / "kfold_missforest"
checkpoint_dir = base_dir / "checkpoint"
checkpoint_dir.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FEATURES_SELECT = 50
POSITIVE_CLASS = 1
LABEL_COL = "Column1"

N_JOBS = OUTER_N_JOBS
CHECKPOINT_EVERY = 50
INNER_THRESHOLD_VALID_SIZE = 0.25
THRESHOLD_OPTIMIZE_METRIC = "f2"
TUNE_HYPERPARAMETERS = True
MAX_PARAM_CANDIDATES = {
    "knn": 10,
    "svm": 12,
    "rf": 18,
    "lr": 8,
    "hgb": 18,
    "xgboost": 24,
}

# %% [markdown]
# ## 1. 실험 설정 및 임계값 로드

# %%
threshold_path = base_dir / "threshold_results_valid.csv"

IMPUTATION_NAME_MAP = {
    "raw": "raw",
    "raw+M": "raw_M",
    "knn": "inpaintknn",
    "knn+M": "inpaintknn_M",
    "mice": "mice",
    "mice+M": "mice_M",
    "mean": "mean",
    "mean+M": "mean_M",
    "missforest": "missforest",
    "missforest+M": "missforest_M",
    "M_only": "M_only",
}

BALANCING_NAME_MAP = {
    "raw": "raw",
    "SMOTE": "smote",
    "smote": "smote",
}

FEATURE_SELECTION_NAME_MAP = {
    "anova": "f_classif",
    "ANOVA": "f_classif",
    "f_classif": "f_classif",
    "chi2": "chi2",
    "fdr": "fdr",
    "fpr": "fpr",
    "fwe": "fwe",
    "mutual_info": "mutual_info",
    "raw": "raw",
}

MODEL_NAME_MAP = {
    "KNN": "knn",
    "SVM": "svm",
    "RF": "rf",
    "LR": "lr",
    "XGBoost": "xgboost",
    "HGB": "hgb",
    "knn": "knn",
    "svm": "svm",
    "rf": "rf",
    "lr": "lr",
    "xgboost": "xgboost",
    "hgb": "hgb",
}

def load_threshold_table(threshold_csv_path):
    """
    threshold_results_valid.csv를 읽고 코드 내부 이름으로 매핑한다.
    필요한 컬럼:
    - imputation
    - balancing
    - feature_selection
    - model
    - best_threshold
    """
    threshold_csv_path = Path(threshold_csv_path)

    if not threshold_csv_path.exists():
        raise FileNotFoundError(
            f"threshold 파일 없음: {threshold_csv_path}\n"
            "base_dir 아래에 threshold_results_valid.csv 파일을 넣어주세요."
        )

    df = pd.read_csv(threshold_csv_path)

    required_cols = ["imputation", "balancing", "feature_selection", "model", "best_threshold"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"threshold 파일에 필수 컬럼이 없습니다: {missing_cols}\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    df = df.copy()

    # 문자열 컬럼 공백 제거
    for col in ["imputation", "balancing", "feature_selection", "model"]:
        df[col] = df[col].astype(str).str.strip()

    df["pipeline"] = df["imputation"].map(IMPUTATION_NAME_MAP)
    df["balancing_code"] = df["balancing"].map(BALANCING_NAME_MAP)
    df["feature_selection_code"] = df["feature_selection"].map(FEATURE_SELECTION_NAME_MAP)
    df["model_code"] = df["model"].map(MODEL_NAME_MAP)

    mapping_cols = ["pipeline", "balancing_code", "feature_selection_code", "model_code"]
    failed_map = df[df[mapping_cols].isna().any(axis=1)]

    if len(failed_map) > 0:
        print("\n오류: threshold 이름 매핑 실패 행:")
        print(
            failed_map[
                ["imputation", "balancing", "feature_selection", "model"]
            ].drop_duplicates().to_string(index=False)
        )
        raise ValueError("threshold_results_valid.csv의 이름이 코드 이름과 매핑되지 않습니다.")

    df["best_threshold"] = pd.to_numeric(df["best_threshold"], errors="coerce")
    invalid_threshold = df[df["best_threshold"].isna()]

    if len(invalid_threshold) > 0:
        print("\n오류: best_threshold 숫자 변환 실패 행:")
        print(
            invalid_threshold[
                ["imputation", "balancing", "feature_selection", "model", "best_threshold"]
            ].head(20).to_string(index=False)
        )
        raise ValueError("best_threshold에 숫자가 아닌 값이 있습니다.")

    # 0~1 밖의 threshold는 보통 비정상. 일단 즉시 중단.
    out_of_range = df[(df["best_threshold"] < 0) | (df["best_threshold"] > 1)]
    if len(out_of_range) > 0:
        print("\n오류: best_threshold 범위 오류 행:")
        print(
            out_of_range[
                ["imputation", "balancing", "feature_selection", "model", "best_threshold"]
            ].head(20).to_string(index=False)
        )
        raise ValueError("best_threshold는 0~1 사이여야 합니다.")

    # 동일 조합 중복 체크
    key_cols = ["pipeline", "balancing_code", "feature_selection_code", "model_code"]
    duplicated = df[df.duplicated(key_cols, keep=False)].sort_values(key_cols)

    if len(duplicated) > 0:
        print("\n오류: threshold 조합 중복 행:")
        print(
            duplicated[
                ["imputation", "balancing", "feature_selection", "model", "best_threshold"]
            ].head(30).to_string(index=False)
        )
        raise ValueError("threshold 파일에 동일 조합이 중복되어 있습니다.")

    print(f"✓ threshold 파일 로드 완료: {threshold_csv_path}")
    print(f"  - threshold 조합 수: {len(df)}")

    return df


threshold_df = None

def get_best_threshold(pipeline, balancing, feature_selection, model, threshold_df):
    """
    현재 실험 조합에 대응되는 best_threshold를 threshold_df에서 가져온다.
    없으면 실험 조건 불일치이므로 중단하지 않고 해당 조합만 skip되도록 ValueError 발생.
    """
    row = threshold_df[
        (threshold_df["pipeline"] == pipeline) &
        (threshold_df["balancing_code"] == balancing) &
        (threshold_df["feature_selection_code"] == feature_selection) &
        (threshold_df["model_code"] == model)
    ]

    if len(row) == 0:
        raise ValueError(
            "threshold 없음: "
            f"pipeline={pipeline}, balancing={balancing}, "
            f"feature_selection={feature_selection}, model={model}"
        )

    return float(row.iloc[0]["best_threshold"])



# =========================================================
# 실험 조건
# =========================================================
selected_pipelines = [
    "raw",
    "raw_M",
    "M_only",
    "inpaintknn",
    "mice",
    "missforest",
    "mean",
    "mean_M",
    "inpaintknn_M",
    "mice_M",
    "missforest_M"
]

selected_balancers = ["raw", "smote"]

selected_feature_selectors = [
    "chi2",
    "f_classif",
    "mutual_info",
    "fpr",
    "fdr",
    "fwe",
    "raw"
]

selected_models = ["knn", "svm", "rf", "lr", "hgb", "xgboost"]
selected_folds = list(range(1, 10))

# %% [markdown]
# ## 2. 데이터 로드 및 컬럼 정합성 유틸

# %%
def read_fold_file(data_dir, fold, split):
    """
    fold 파일을 읽는다.
    raw kfold는 fold1_train.xlsx 형태이고,
    imputation 결과 파일은 fold1_train_imputed.xlsx,
    fold1_train_mice.xlsx, fold1_train_missforest.xlsx 형태일 수 있으므로
    후보 파일명을 순서대로 탐색한다.
    """
    data_dir = Path(data_dir)

    candidate_names = [
        f"fold{fold}_{split}.xlsx",
        f"fold{fold}_{split}.csv",

        f"fold{fold}_{split}_imputed.xlsx",
        f"fold{fold}_{split}_imputed.csv",

        f"fold{fold}_{split}_mice.xlsx",
        f"fold{fold}_{split}_mice.csv",

        f"fold{fold}_{split}_missforest.xlsx",
        f"fold{fold}_{split}_missforest.csv",
    ]

    for fname in candidate_names:
        path = data_dir / fname

        if path.exists():
            if path.suffix.lower() == ".xlsx":
                return pd.read_excel(path)
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)

    raise FileNotFoundError(
        f"{data_dir} 에서 fold{fold}_{split} 파일을 찾지 못했습니다. "
        f"확인한 후보: {candidate_names}"
    )

def check_label_column(df, where):
    if LABEL_COL not in df.columns:
        raise ValueError(f"{where}: label column '{LABEL_COL}' 없음. 현재 컬럼 예시: {list(df.columns)[:10]}")

def align_train_valid_columns(train_df, valid_df):
    train_df = train_df.copy()
    valid_df = valid_df.copy()

    train_cols = list(train_df.columns)
    valid_cols = list(valid_df.columns)

    for c in train_cols:
        if c not in valid_cols:
            valid_df[c] = 0

    for c in valid_cols:
        if c not in train_cols:
            train_df[c] = 0

    valid_df = valid_df[train_cols]
    return train_df, valid_df

# %% [markdown]
# ## 3. 결측치 파생변수 생성 및 평균 대체

# %%
def create_missing_indicator_features(df, label_col=LABEL_COL):
    df = df.copy()
    check_label_column(df, "create_missing_indicator_features")

    feature_cols = [c for c in df.columns if c != label_col]
    m_df = pd.DataFrame(index=df.index)

    for col in feature_cols:
        if not str(col).endswith("_M"):
            m_df[f"{col}_M"] = df[col].isna().astype(int)

    return m_df

def load_missing_indicator_from_original_kfold(fold):
    raw_train = read_fold_file(raw_dir, fold, "train")
    raw_valid = read_fold_file(raw_dir, fold, "valid")

    check_label_column(raw_train, f"raw fold{fold} train")
    check_label_column(raw_valid, f"raw fold{fold} valid")

    train_m = create_missing_indicator_features(raw_train, label_col=LABEL_COL)
    valid_m = create_missing_indicator_features(raw_valid, label_col=LABEL_COL)

    train_m, valid_m = align_train_valid_columns(train_m, valid_m)
    return train_m, valid_m

def mean_impute_dataframe(train_df, valid_df):
    train = train_df.copy()
    valid = valid_df.copy()

    check_label_column(train, "mean train")
    check_label_column(valid, "mean valid")

    feature_cols = [c for c in train.columns if c != LABEL_COL and not str(c).endswith("_M")]

    means = train[feature_cols].mean(axis=0, skipna=True).fillna(0)

    train[feature_cols] = train[feature_cols].fillna(means)
    valid[feature_cols] = valid[feature_cols].fillna(means)

    return train, valid

# %% [markdown]
# ## 4. 파이프라인별 학습/검증 데이터 구성

# %%
def get_base_data_by_pipeline(pipeline, fold):
    if pipeline in ["raw", "raw_M", "M_only", "mean", "mean_M"]:
        data_dir = raw_dir
    elif pipeline in ["inpaintknn", "inpaintknn_M"]:
        data_dir = inpaintknn_dir
    elif pipeline in ["mice", "mice_M"]:
        data_dir = mice_dir
    elif pipeline in ["missforest", "missforest_M"]:
        data_dir = missforest_dir
    else:
        raise ValueError(f"알 수 없는 pipeline: {pipeline}")

    train_df = read_fold_file(data_dir, fold, "train")
    valid_df = read_fold_file(data_dir, fold, "valid")

    check_label_column(train_df, f"{pipeline} fold{fold} train")
    check_label_column(valid_df, f"{pipeline} fold{fold} valid")

    train_df, valid_df = align_train_valid_columns(train_df, valid_df)

    if pipeline in ["mean", "mean_M"]:
        train_df, valid_df = mean_impute_dataframe(train_df, valid_df)

    return train_df, valid_df

def load_experiment_data(pipeline, fold):
    train_base, valid_base = get_base_data_by_pipeline(pipeline, fold)

    y_train = train_base[LABEL_COL]
    y_valid = valid_base[LABEL_COL]

    X_train_base = train_base.drop(columns=[LABEL_COL])
    X_valid_base = valid_base.drop(columns=[LABEL_COL])

    X_train_base = X_train_base[[c for c in X_train_base.columns if not str(c).endswith("_M")]]
    X_valid_base = X_valid_base[X_train_base.columns]

    train_m, valid_m = load_missing_indicator_from_original_kfold(fold)

    if pipeline == "M_only":
        X_train = train_m.copy()
        X_valid = valid_m.copy()

    elif pipeline.endswith("_M"):
        X_train = pd.concat([X_train_base, train_m], axis=1)
        X_valid = pd.concat([X_valid_base, valid_m], axis=1)
        X_train, X_valid = align_train_valid_columns(X_train, X_valid)

    else:
        X_train = X_train_base.copy()
        X_valid = X_valid_base.copy()

    return X_train, y_train, X_valid, y_valid

# %% [markdown]
# ## 5. Feature Selection 설정 및 적용

# %%
FEATURE_SELECTORS = {
    "raw": lambda X, y: None,
    "chi2": lambda X, y: SelectKBest(chi2, k=min(N_FEATURES_SELECT, X.shape[1])),
    "f_classif": lambda X, y: SelectKBest(f_classif, k=min(N_FEATURES_SELECT, X.shape[1])),
    "mutual_info": lambda X, y: SelectKBest(mutual_info_classif, k=min(N_FEATURES_SELECT, X.shape[1])),
    "fpr": lambda X, y: SelectFpr(alpha=0.05),
    "fdr": lambda X, y: SelectFdr(alpha=0.05),
    "fwe": lambda X, y: SelectFwe(alpha=0.05),
}

def _check_selected_cols(selected_cols, fs_key, pipeline):
    if len(selected_cols) == 0:
        raise ValueError(f"{pipeline}-{fs_key}: feature selection selected 0 features")

def apply_feature_selection(fs_key, X_train, y_train, X_valid, pipeline):
    X_train = pd.DataFrame(X_train).copy()
    X_valid = pd.DataFrame(X_valid).copy()

    if pipeline.endswith("_M"):
        m_cols = [c for c in X_train.columns if str(c).endswith("_M")]
        base_cols = [c for c in X_train.columns if not str(c).endswith("_M")]

        X_train_base = X_train[base_cols].copy()

        if fs_key == "raw":
            selected_base_cols = base_cols
            selector = None
        else:
            X_train_for_fs = X_train_base.copy()

            if fs_key == "chi2":
                X_train_for_fs = np.clip(X_train_for_fs, 0, None)

            selector = FEATURE_SELECTORS[fs_key](X_train_for_fs, y_train)
            selector.fit(X_train_for_fs, y_train)

            selected_base_cols = list(X_train_base.columns[selector.get_support()])

        _check_selected_cols(selected_base_cols, fs_key, pipeline)

        selected_m_cols = [f"{c}_M" for c in selected_base_cols if f"{c}_M" in m_cols]
        final_cols = selected_base_cols + selected_m_cols

        return X_train[final_cols], X_valid[final_cols], selector, final_cols

    if pipeline == "M_only":
        if fs_key == "raw":
            selected_cols = list(X_train.columns)
            _check_selected_cols(selected_cols, fs_key, pipeline)
            return X_train, X_valid, None, selected_cols

        X_train_for_fs = X_train.copy()

        if fs_key == "chi2":
            X_train_for_fs = np.clip(X_train_for_fs, 0, None)

        selector = FEATURE_SELECTORS[fs_key](X_train_for_fs, y_train)
        selector.fit(X_train_for_fs, y_train)

        selected_cols = list(X_train.columns[selector.get_support()])
        _check_selected_cols(selected_cols, fs_key, pipeline)

        return X_train[selected_cols], X_valid[selected_cols], selector, selected_cols

    if fs_key == "raw":
        selected_cols = list(X_train.columns)
        _check_selected_cols(selected_cols, fs_key, pipeline)
        return X_train, X_valid, None, selected_cols

    X_train_for_fs = X_train.copy()

    if fs_key == "chi2":
        X_train_for_fs = np.clip(X_train_for_fs, 0, None)

    selector = FEATURE_SELECTORS[fs_key](X_train_for_fs, y_train)
    selector.fit(X_train_for_fs, y_train)

    selected_cols = list(X_train.columns[selector.get_support()])
    _check_selected_cols(selected_cols, fs_key, pipeline)

    return X_train[selected_cols], X_valid[selected_cols], selector, selected_cols

# %% [markdown]
# ## 6. 모델, 불균형 처리, 스케일링 설정

# %%
MODELS = {
    "knn": KNeighborsClassifier(n_neighbors=5),
    "svm": SVC(probability=True, random_state=RANDOM_STATE),
    "rf": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=MODEL_N_JOBS
    ),
    "lr": LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=10000
    ),
    "hgb": HistGradientBoostingClassifier(
        random_state=RANDOM_STATE
    ),
    "xgboost": XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=MODEL_N_JOBS
    )
}


PARAM_GRIDS = {
    "knn": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
    "svm": {
        "C": [0.1, 1, 10, 50],
        "gamma": ["scale", 0.01, 0.001],
        "class_weight": [None, "balanced"],
    },
    "rf": {
        "n_estimators": [300, 500],
        "max_depth": [None, 5, 10, 20],
        "min_samples_leaf": [1, 3, 5],
        "class_weight": [None, "balanced", "balanced_subsample"],
    },
    "lr": {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"],
        "class_weight": [None, "balanced"],
    },
    "hgb": {
        "learning_rate": [0.03, 0.05, 0.1],
        "max_iter": [200, 500],
        "max_leaf_nodes": [15, 31, 63],
        "l2_regularization": [0, 0.1, 1],
    },
    "xgboost": {
        "n_estimators": [200, 500],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "scale_pos_weight": ["auto"],
    },
}


BALANCERS = {
    "raw": lambda X, y: (X, y),
    "smote": lambda X, y: SMOTE(random_state=RANDOM_STATE).fit_resample(X, y)
}

def scale_data(model_key, X_train, X_valid):
    if model_key in ["lr", "svm", "knn"]:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        return X_train_scaled, X_valid_scaled

    return X_train, X_valid

# %% [markdown]
# ## 7. 평가 지표 및 학습/예측 함수

# %%
def get_positive_proba(model, X, positive_class=POSITIVE_CLASS):
    if not hasattr(model, "predict_proba"):
        return None

    y_proba = model.predict_proba(X)

    if y_proba.ndim == 1:
        return y_proba

    classes = list(model.classes_)

    if positive_class in classes:
        pos_idx = classes.index(positive_class)
    elif 1 in classes:
        pos_idx = classes.index(1)
    else:
        pos_idx = -1

    return y_proba[:, pos_idx]

def get_auc(y_true, y_score, positive_class=POSITIVE_CLASS):
    if y_score is None:
        return None

    y_true_array = np.asarray(y_true)

    if len(np.unique(y_true_array)) < 2:
        return None

    try:
        y_binary = (y_true_array == positive_class).astype(int)
        return roc_auc_score(y_binary, y_score)
    except Exception:
        return None

def evaluate_model(y_true, y_pred, y_score=None, positive_class=POSITIVE_CLASS):
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)

    labels_for_cm = [-1, positive_class] if set(np.unique(y_true_array)).issubset({-1, positive_class}) else np.unique(y_true_array)

    if len(labels_for_cm) == 2:
        negative_class = labels_for_cm[0] if labels_for_cm[0] != positive_class else labels_for_cm[1]

        cm = confusion_matrix(
            y_true_array,
            y_pred_array,
            labels=[negative_class, positive_class]
        )

        tn, fp, fn, tp = cm.ravel()

        precision = precision_score(
            y_true_array,
            y_pred_array,
            pos_label=positive_class,
            zero_division=0
        )

        recall = recall_score(
            y_true_array,
            y_pred_array,
            pos_label=positive_class,
            zero_division=0
        )

        f1_binary = f1_score(
            y_true_array,
            y_pred_array,
            pos_label=positive_class,
            zero_division=0
        )

        f2_binary = fbeta_score(
            y_true_array,
            y_pred_array,
            beta=2,
            pos_label=positive_class,
            zero_division=0
        )

        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    else:
        cm = confusion_matrix(y_true_array, y_pred_array)
        tn = fp = fn = tp = np.nan

        precision = precision_score(
            y_true_array,
            y_pred_array,
            average="weighted",
            zero_division=0
        )

        recall = recall_score(
            y_true_array,
            y_pred_array,
            average="weighted",
            zero_division=0
        )

        f1_binary = np.nan
        f2_binary = np.nan
        tnr = np.nan

    auc_value = get_auc(
        y_true_array,
        y_score,
        positive_class=positive_class
    )

    return {
        "f1": f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true_array, y_pred_array, average="macro", zero_division=0),
        "f1_binary": f1_binary,
        "f2": f2_binary,
        "f2_binary": f2_binary,
        "auc": auc_value,
        "tp": int(tp) if not pd.isna(tp) else np.nan,
        "tn": int(tn) if not pd.isna(tn) else np.nan,
        "fp": int(fp) if not pd.isna(fp) else np.nan,
        "fn": int(fn) if not pd.isna(fn) else np.nan,
        "precision": precision,
        "recall": recall,
        "tnr": tnr,
        "confusion_matrix": cm.tolist()
    }

def fit_predict_model(model_key, model, X_train, y_train, X_valid):
    if model_key == "xgboost":
        y_train_fit = (np.asarray(y_train) == POSITIVE_CLASS).astype(int)

        model.fit(X_train, y_train_fit)

        y_pred_binary = model.predict(X_valid)
        y_pred = np.where(y_pred_binary == 1, POSITIVE_CLASS, -1)

        y_score = get_positive_proba(model, X_valid, positive_class=1)

        return y_pred, y_score

    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_score = get_positive_proba(model, X_valid, positive_class=POSITIVE_CLASS)

    return y_pred, y_score

def get_class_ratio(y, positive_class=POSITIVE_CLASS):
    y_array = np.asarray(y)
    n_pos = np.sum(y_array == positive_class)
    n_neg = len(y_array) - n_pos

    if n_pos == 0:
        return 1.0

    return float(n_neg / n_pos)


def resolve_model_params(model_key, params, y_reference, balancing_key):
    resolved = dict(params)

    if model_key == "xgboost" and resolved.get("scale_pos_weight") == "auto":
        resolved["scale_pos_weight"] = (
            1.0 if balancing_key == "smote"
            else get_class_ratio(y_reference, positive_class=POSITIVE_CLASS)
        )

    return resolved


def get_param_candidates(model_key):
    if not TUNE_HYPERPARAMETERS:
        return [{}]

    all_candidates = list(ParameterGrid(PARAM_GRIDS[model_key]))
    max_candidates = MAX_PARAM_CANDIDATES.get(model_key, len(all_candidates))

    if len(all_candidates) > max_candidates:
        rng = np.random.default_rng(RANDOM_STATE)
        selected_idx = sorted(rng.choice(len(all_candidates), size=max_candidates, replace=False))
        all_candidates = [all_candidates[i] for i in selected_idx]

    return [{}] + [p for p in all_candidates if p != {}]


def find_best_threshold_by_f2(y_true, y_score, positive_class=POSITIVE_CLASS):
    if y_score is None:
        return 0.5

    y_score = np.asarray(y_score)
    if y_score.size == 0:
        return 0.5

    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], y_score)))
    best_threshold = 0.5
    best_score = -1.0

    for threshold in thresholds:
        y_pred = np.where(y_score >= threshold, positive_class, -1)
        score = fbeta_score(
            y_true,
            y_pred,
            beta=2,
            pos_label=positive_class,
            zero_division=0
        )

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold


def tune_model_and_threshold_from_inner_train(model_key, X_train, y_train, balancing_key):
    y_array = np.asarray(y_train)
    classes, counts = np.unique(y_array, return_counts=True)

    if len(classes) < 2 or counts.min() < 2:
        return {}, 0.5, np.nan

    try:
        X_inner_train, X_inner_valid, y_inner_train, y_inner_valid = train_test_split(
            X_train,
            y_train,
            test_size=INNER_THRESHOLD_VALID_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_train
        )
    except ValueError:
        return {}, 0.5, np.nan

    best_params = {}
    best_threshold = 0.5
    best_score = -1.0

    for candidate_params in get_param_candidates(model_key):
        try:
            resolved_params = resolve_model_params(
                model_key=model_key,
                params=candidate_params,
                y_reference=y_inner_train,
                balancing_key=balancing_key
            )

            X_inner_bal, y_inner_bal = BALANCERS[balancing_key](X_inner_train, y_inner_train)
            X_inner_bal_scaled, X_inner_valid_scaled = scale_data(
                model_key,
                X_inner_bal,
                X_inner_valid
            )

            model = clone(MODELS[model_key])
            model.set_params(**resolved_params)

            _, y_inner_score = fit_predict_model(
                model_key=model_key,
                model=model,
                X_train=X_inner_bal_scaled,
                y_train=y_inner_bal,
                X_valid=X_inner_valid_scaled
            )

            threshold = find_best_threshold_by_f2(
                y_true=y_inner_valid,
                y_score=y_inner_score,
                positive_class=POSITIVE_CLASS
            )

            y_inner_pred = np.where(y_inner_score >= threshold, POSITIVE_CLASS, -1)
            score = fbeta_score(
                y_inner_valid,
                y_inner_pred,
                beta=2,
                pos_label=POSITIVE_CLASS,
                zero_division=0
            )

            if score > best_score:
                best_score = score
                best_threshold = threshold
                best_params = resolved_params

        except Exception:
            continue

    if best_score < 0:
        return {}, 0.5, np.nan

    return best_params, best_threshold, best_score


# =========================================================
# 단일 실험
# =========================================================

# %% [markdown]
# ## 8. 단일 실험 실행 함수

# %%
def run_experiment(pipeline, balancing_key, fs_key, model_key, fold):
    X_train, y_train, X_valid, y_valid = load_experiment_data(pipeline, fold)

    X_train_fs, X_valid_fs, selector, selected_cols = apply_feature_selection(
        fs_key, X_train, y_train, X_valid, pipeline
    )

    best_params, best_threshold, inner_f2 = tune_model_and_threshold_from_inner_train(
        model_key=model_key,
        X_train=X_train_fs,
        y_train=y_train,
        balancing_key=balancing_key
    )

    X_train_bal, y_train_bal = BALANCERS[balancing_key](X_train_fs, y_train)

    X_train_scaled, X_valid_scaled = scale_data(model_key, X_train_bal, X_valid_fs)

    model = clone(MODELS[model_key])
    model.set_params(**best_params)

    y_pred_default, y_score = fit_predict_model(
        model_key=model_key,
        model=model,
        X_train=X_train_scaled,
        y_train=y_train_bal,
        X_valid=X_valid_scaled
    )

    if y_score is not None:
        y_pred = np.where(y_score >= best_threshold, POSITIVE_CLASS, -1)
    else:
        # 확률 점수가 없는 모델이면 기본 predict 결과 사용
        # 현재 모델 구성에서는 대부분 predict_proba를 제공하지만 안전장치로 둔다.
        y_pred = y_pred_default

    metrics = evaluate_model(
        y_true=y_valid,
        y_pred=y_pred,
        y_score=y_score,
        positive_class=POSITIVE_CLASS
    )

    if y_score is not None and len(np.unique(y_valid)) == 2:
        y_binary = (np.asarray(y_valid) == POSITIVE_CLASS).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_score)

        metrics["roc_fpr"] = fpr.tolist()
        metrics["roc_tpr"] = tpr.tolist()

    n_base_features = len([c for c in selected_cols if not str(c).endswith("_M")])
    n_m_features = len([c for c in selected_cols if str(c).endswith("_M")])

    metrics.update({
        "pipeline": pipeline,
        "balancing": balancing_key,
        "feature_selection": fs_key,
        "model": model_key,
        "fold": fold,
        "best_threshold": best_threshold,
        "threshold_source": "inner_train_holdout_f2",
        "inner_f2": inner_f2,
        "model_params": str(best_params),
        "n_param_candidates": len(get_param_candidates(model_key)),
        "n_features": X_train_fs.shape[1],
        "n_base_features": n_base_features,
        "n_m_features": n_m_features,
        "selected_features": "|".join(map(str, selected_cols))
    })

    return metrics


# =========================================================
# 데이터 검증
# =========================================================

# %% [markdown]
# ## 9. 데이터 사전 검증 함수

# %%
def validate_data_availability(pipelines, folds):
    print("\n=== 데이터 검증 ===\n")

    dir_mapping = {
        "raw": raw_dir,
        "raw_M": raw_dir,
        "M_only": raw_dir,
        "mean": raw_dir,
        "mean_M": raw_dir,
        "inpaintknn": inpaintknn_dir,
        "inpaintknn_M": inpaintknn_dir,
        "mice": mice_dir,
        "mice_M": mice_dir,
        "missforest": missforest_dir,
        "missforest_M": missforest_dir,
    }

    missing_count = 0

    for pipeline in pipelines:
        data_dir = dir_mapping[pipeline]

        for fold in folds:
            for split in ["train", "valid"]:
                try:
                    _ = read_fold_file(data_dir, fold, split)
                except FileNotFoundError:
                    print(f"⚠️ 누락: {pipeline} fold{fold}_{split}")
                    missing_count += 1

    if missing_count == 0:
        print("✓ 모든 데이터 파일 확인 완료")
    else:
        print(f"⚠️ {missing_count}개 파일 누락")

    return missing_count == 0

def smoke_test_file_loading(pipelines, fold=1):
    """
    전체 실험 전에 각 pipeline이 실제로 train/valid 파일을 읽을 수 있는지 확인한다.
    여기서 inpaintknn/mice/missforest가 실패하면 파일명 또는 경로 문제다.
    """
    print("\n[파일 로드 smoke test]")
    failed = []

    for pipeline in pipelines:
        try:
            train_df, valid_df = get_base_data_by_pipeline(pipeline, fold)
            print(f"✓ {pipeline}: train={train_df.shape}, valid={valid_df.shape}")
        except Exception as e:
            failed.append((pipeline, str(e)))
            print(f"✗ {pipeline}: {str(e)[:200]}")

    if failed:
        print("\n⚠️ smoke test 실패 pipeline:")
        for pipeline, err in failed:
            print(f"  - {pipeline}: {err[:200]}")
        raise RuntimeError("파일 로드 smoke test 실패. 위 pipeline의 파일명/경로를 확인하세요.")

    print("✓ 모든 pipeline 파일 로드 smoke test 통과")

# %% [markdown]
# ## 10. M 피처 효과 비교 함수

# %%
def get_pair_pipeline(pipeline):
    pair_map = {
        "raw_M": "raw",
        "mean_M": "mean",
        "inpaintknn_M": "inpaintknn",
        "mice_M": "mice",
        "missforest_M": "missforest",
    }
    return pair_map.get(pipeline, None)

def build_m_comparison_table(results_df):
    key_cols = ["balancing", "feature_selection", "model", "fold"]

    metrics = [
        "f1",
        "f1_binary",
        "f1_macro",
        "f2",
        "f2_binary",
        "auc",
        "precision",
        "recall",
        "tnr"
    ]

    required_cols = set(["pipeline"] + key_cols)
    if results_df is None or len(results_df) == 0 or not required_cols.issubset(results_df.columns):
        return pd.DataFrame()

    comparison_rows = []
    m_pipelines = ["raw_M", "mean_M", "inpaintknn_M", "mice_M", "missforest_M"]

    for m_pipeline in m_pipelines:
        base_pipeline = get_pair_pipeline(m_pipeline)

        m_df = results_df[results_df["pipeline"] == m_pipeline]
        base_df = results_df[results_df["pipeline"] == base_pipeline]

        merged = pd.merge(
            base_df,
            m_df,
            on=key_cols,
            suffixes=("_base", "_M")
        )

        for _, row in merged.iterrows():
            out = {
                "base_pipeline": base_pipeline,
                "M_pipeline": m_pipeline
            }

            for c in key_cols:
                out[c] = row[c]

            for metric in metrics:
                out[f"{metric}_base"] = row.get(f"{metric}_base", np.nan)
                out[f"{metric}_M"] = row.get(f"{metric}_M", np.nan)
                out[f"{metric}_gain"] = out[f"{metric}_M"] - out[f"{metric}_base"]

            comparison_rows.append(out)

    return pd.DataFrame(comparison_rows)

def aggregate_m_comparison(comparison_df):
    if len(comparison_df) == 0:
        return pd.DataFrame()

    group_cols = ["base_pipeline", "M_pipeline", "balancing", "feature_selection", "model"]
    numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns.tolist()

    agg_df = comparison_df.groupby(group_cols)[numeric_cols].agg(["mean", "std"]).reset_index()
    agg_df.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg_df.columns
    ]

    return agg_df

# %% [markdown]
# ## 11. 결과 시각화 함수

# %%
def plot_performance_boxplots(results_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["f1", "f2", "auc", "precision", "recall", "tnr"]

    for metric in metrics:
        if metric not in results_df.columns:
            continue

        pipelines = results_df["pipeline"].dropna().unique().tolist()

        data = [
            results_df[results_df["pipeline"] == p][metric].dropna().values
            for p in pipelines
        ]

        if all(len(x) == 0 for x in data):
            continue

        plt.figure(figsize=(12, 5))
        plt.boxplot(data, labels=pipelines)
        plt.ylabel(metric)
        plt.title(f"{metric} distribution by pipeline")
        plt.xticks(rotation=70, ha="right")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / f"boxplot_{metric}_by_pipeline.png", dpi=300, bbox_inches="tight")
        plt.close()

def plot_gain_barplots(agg_comparison_df, output_dir, top_n=20):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gain_cols = [
        "f1_gain_mean",
        "f2_gain_mean",
        "auc_gain_mean",
        "precision_gain_mean",
        "recall_gain_mean",
        "tnr_gain_mean"
    ]

    for gain_col in gain_cols:
        if gain_col not in agg_comparison_df.columns:
            continue

        top_df = agg_comparison_df.sort_values(gain_col, ascending=False).head(top_n).copy()

        labels = (
            top_df["M_pipeline"].astype(str) + "-" +
            top_df["balancing"].astype(str) + "-" +
            top_df["feature_selection"].astype(str) + "-" +
            top_df["model"].astype(str)
        )

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(top_df)), top_df[gain_col].values)
        plt.xticks(range(len(top_df)), labels, rotation=80, ha="right")
        plt.ylabel(gain_col)
        plt.title(f"Top {top_n} {gain_col}")
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(output_dir / f"barplot_top_{gain_col}.png", dpi=300, bbox_inches="tight")
        plt.close()

def plot_roc_curves_top_models(results_df, output_dir, top_n=5):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "roc_fpr" not in results_df.columns or "roc_tpr" not in results_df.columns:
        return

    top_results = results_df.sort_values("f1", ascending=False).head(top_n)

    plt.figure(figsize=(7, 6))
    plotted = 0

    for _, row in top_results.iterrows():
        if isinstance(row.get("roc_fpr"), list) and isinstance(row.get("roc_tpr"), list):
            auc_text = row["auc"] if pd.notna(row["auc"]) else np.nan
            label = f"{row['pipeline']}-{row['model']}-{row['feature_selection']} AUC={auc_text:.3f}"
            plt.plot(row["roc_fpr"], row["roc_tpr"], label=label)
            plotted += 1

    if plotted == 0:
        plt.close()
        return

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: Top Models by F1")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves_top_models_by_f1.png", dpi=300, bbox_inches="tight")
    plt.close()

def normalize_confusion_matrix(cm):
    if isinstance(cm, str):
        text = cm.strip()
        try:
            cm = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            values = np.fromstring(
                text.replace("[", " ").replace("]", " ").replace(",", " "),
                sep=" "
            )
            if values.size == 4:
                cm = values.reshape(2, 2)
            else:
                raise ValueError(f"Invalid confusion_matrix value: {text}")

    cm = np.asarray(cm, dtype=float)
    if cm.ndim != 2:
        raise ValueError(f"confusion_matrix must be 2D, got shape {cm.shape}")
    return cm


def plot_confusion_matrix_heatmap(cm, title, save_path):
    cm = normalize_confusion_matrix(cm)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred -1", "Pred 1"])
    plt.yticks(tick_marks, ["True -1", "True 1"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(int(cm[i, j]), "d"), ha="center", va="center")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

def plot_top_confusion_matrices(results_df, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results_df) == 0:
        return

    top_f1 = results_df.sort_values("f1", ascending=False).head(1)
    top_f2 = results_df.sort_values("f2", ascending=False).head(1)

    if len(top_f1) > 0:
        row = top_f1.iloc[0]
        title = (
            f"Top F1 CM\n"
            f"{row['pipeline']}-{row['balancing']}-{row['feature_selection']}-{row['model']}-fold{row['fold']}"
        )
        plot_confusion_matrix_heatmap(
            row["confusion_matrix"],
            title,
            output_dir / "confusion_matrix_top_f1.png"
        )

    if len(top_f2) > 0:
        row = top_f2.iloc[0]
        title = (
            f"Top F2 CM\n"
            f"{row['pipeline']}-{row['balancing']}-{row['feature_selection']}-{row['model']}-fold{row['fold']}"
        )
        plot_confusion_matrix_heatmap(
            row["confusion_matrix"],
            title,
            output_dir / "confusion_matrix_top_f2.png"
        )

def create_all_visualizations(results_df, agg_comparison_df, output_dir):
    plot_performance_boxplots(results_df, output_dir)

    if len(agg_comparison_df) > 0:
        plot_gain_barplots(agg_comparison_df, output_dir)

    plot_roc_curves_top_models(results_df, output_dir)
    plot_top_confusion_matrices(results_df, output_dir)

    print(f"✓ 시각화 저장 완료: {output_dir}")

# %% [markdown]
# ## 12. Simple Baseline 추출 함수

# %%
def extract_simple_baseline(results_df):
    baseline = results_df[
        (results_df["pipeline"] == "mean") &
        (results_df["balancing"] == "raw") &
        (results_df["feature_selection"] == "raw") &
        (results_df["model"] == "lr")
    ].copy()

    return baseline

# %% [markdown]
# ## 13. 분석 실행 설정 및 사전 점검

# %%
print("=" * 70)
print("파이프라인 분석 시작")
print("=" * 70)

print("\n[전체 파이프라인]")
print("1. Data pipeline 11가지:")
for p in selected_pipelines:
    print(f"  - {p}")

print("\n2. Balancing:")
print("  - raw")
print("  - smote")
print("  - M_only includes smote")

print("\n3. Feature selection 7가지:")
for fs in selected_feature_selectors:
    print(f"  - {fs}")

print("\n4. ML model 6가지:")
for m in selected_models:
    print(f"  - {m}")

print("\n5. Evaluation:")
print("  - F1-score")
print("  - F2-score")
print("  - hyperparameters and best_threshold selected on inner-train holdout by F2")
print("  - ROC Curve, Top F1 기준")
print("  - Confusion Matrix, Top F1 1장 + Top F2 1장")
print("  - Precision / Recall / TNR / TP / TN / FP / FN")

print(f"\n[병렬 설정 확인] N_JOBS = {N_JOBS}")
print(f"  - CPU_THREADS = {CPU_THREADS}")
print(f"  - OUTER_N_JOBS = {OUTER_N_JOBS}")
print(f"  - MODEL_N_JOBS = {MODEL_N_JOBS}")
print(f"  - BLAS_THREADS = {BLAS_THREADS}")
print(f"  - JOBLIB_TEMP_FOLDER = {os.environ.get('JOBLIB_TEMP_FOLDER')}")
print("  - Override example: SECOM_OUTER_N_JOBS=9 SECOM_MODEL_N_JOBS=4 SECOM_BLAS_THREADS=4")
print("  - 코어 이중 점유 방지 구조")

data_valid = validate_data_availability(selected_pipelines, selected_folds)

# 전체 병렬 실험 전에 fold1 기준으로 각 pipeline 파일 로드 여부를 확인
smoke_test_file_loading(selected_pipelines, fold=1)

results = []
error_details = []

print("\n단계 3: 실험 실행 중...\n")

# %% [markdown]
# ## 14. Fold별 실험 결과 로드 및 저장

# %%
def run_fold_experiments(fold):
    fold_results = []
    fold_errors = []

    print(f"\n>>> Fold {fold} 시작")

    for pipeline in selected_pipelines:
        for balancing in selected_balancers:

            for fs in selected_feature_selectors:
                for model_key in selected_models:

                    try:
                        result = run_experiment(
                            pipeline=pipeline,
                            balancing_key=balancing,
                            fs_key=fs,
                            model_key=model_key,
                            fold=fold
                        )

                        fold_results.append(result)

                        print(
                            f"  {pipeline}-{balancing[:1]}-{fs[:4]}-{model_key}: "
                            f"F1={result['f1']:.4f}, "
                            f"F2={result['f2']:.4f}, "
                            f"TH={result['best_threshold']:.3f}, "
                            f"Recall={result['recall']:.4f}, "
                            f"TNR={result['tnr']:.4f}, "
                            f"AUC={str(result['auc'])[:6]}"
                        )

                        if len(fold_results) % CHECKPOINT_EVERY == 0:
                            checkpoint_df = pd.DataFrame(fold_results)
                            checkpoint_path = checkpoint_dir / f"checkpoint_fold{fold}_{len(fold_results)}.xlsx"
                            checkpoint_df.to_excel(checkpoint_path, index=False)
                            print(f"✓ Checkpoint 저장 완료: {checkpoint_path.name}")

                    except Exception as e:
                        fold_errors.append((
                            pipeline, balancing, fs, model_key, fold, str(e)
                        ))

                        if len(fold_errors) <= 10:
                            print(
                                f"  SKIP {pipeline}-{balancing}-{fs}-{model_key}-fold{fold}: "
                                f"{str(e)[:120]}"
                            )

    if len(fold_results) > 0:
        checkpoint_df = pd.DataFrame(fold_results)
        checkpoint_path = checkpoint_dir / f"checkpoint_fold{fold}_final.xlsx"
        checkpoint_df.to_excel(checkpoint_path, index=False)
        print(f"✓ Fold {fold} final checkpoint 저장 완료: {checkpoint_path.name}")

    return fold_results, fold_errors

results = []
error_details = []
error_count = 0
parallel_results = []

for fold in selected_folds:
    final_path = checkpoint_dir / f"checkpoint_fold{fold}_final.xlsx"

    if final_path.exists():
        fold_df = pd.read_excel(final_path)
        fold_results = fold_df.to_dict("records")
        fold_errors = []

        parallel_results.append((fold_results, fold_errors))
        print(f"✓ Fold {fold}: final checkpoint 로드 완료 ({len(fold_results)}개)")
    else:
        latest_files = sorted(
            checkpoint_dir.glob(f"checkpoint_fold{fold}_*.xlsx"),
            key=lambda path: path.stat().st_mtime
        )

        if len(latest_files) > 0:
            latest_path = latest_files[-1]
            fold_df = pd.read_excel(latest_path)
            fold_results = fold_df.to_dict("records")
            fold_errors = []

            parallel_results.append((fold_results, fold_errors))
            print(f"⚠ Fold {fold}: final 없음, {latest_path.name} 로드 ({len(fold_results)}개)")
        else:
            print(f"✗ Fold {fold}: checkpoint 없음, 실험을 새로 실행합니다.")
            fold_results, fold_errors = run_fold_experiments(fold)
            parallel_results.append((fold_results, fold_errors))

for fold_results, fold_errors in parallel_results:
    results.extend(fold_results)
    error_details.extend(fold_errors)

error_count = len(error_details)

print("\n✓ 단계 3 완료")
print(f"  - 성공한 실험: {len(results)}개")
print(f"  - 스킵/에러 발생: {error_count}개")

if error_count > 0:
    error_df = pd.DataFrame(
        error_details,
        columns=["pipeline", "balancing", "feature_selection", "model", "fold", "error"]
    )
    error_df.to_excel(base_dir / "error_details.xlsx", index=False)
    print("  - 에러 상세 저장: error_details.xlsx")

results_df = pd.DataFrame(results)
results_df.to_excel(base_dir / "analysis_results.xlsx", index=False)
print("\n✓ 결과 저장: analysis_results.xlsx")

# %% [markdown]
# ## 15. 전체 실험 결과 요약

# %%
summary_cols = [
    "f1",
    "f1_binary",
    "f1_macro",
    "f2",
    "f2_binary",
    "auc",
    "precision",
    "recall",
    "tnr"
]

print("\n" + "=" * 70)
print("단계 4: 분석 결과 요약")
print("=" * 70)

if len(results_df) > 0:
    summary = results_df.groupby(
        ["pipeline", "balancing", "feature_selection", "model"]
    )[summary_cols].agg(["mean", "std"]).reset_index()

    summary.columns = [
        "_".join([str(x) for x in col if str(x) != ""])
        if isinstance(col, tuple) else str(col)
        for col in summary.columns
    ]

    summary.to_excel(base_dir / "summary_by_pipeline.xlsx", index=False)
    print("\n✓ 요약 결과 저장: summary_by_pipeline.xlsx")

    print("\n[상위 15개: F2 기준]")
    print(
        results_df.groupby(["pipeline", "balancing", "feature_selection", "model"])
        .agg(
            f1_mean=("f1", "mean"),
            f2_mean=("f2", "mean"),
            recall_mean=("recall", "mean"),
            tnr_mean=("tnr", "mean"),
            auc_mean=("auc", "mean"),
            count=("fold", "count")
        )
        .reset_index()
        .sort_values("f2_mean", ascending=False)
        .head(15)
        .to_string(index=False)
    )

# %% [markdown]
# ## 16. M 피처 비교 결과

# %%
if len(results_df) > 0 and "pipeline" in results_df.columns:
    comparison_df = build_m_comparison_table(results_df)
    comparison_df.to_excel(base_dir / "m_feature_comparison_by_fold.xlsx", index=False)
    print("\n✓ Fold별 M 피처 비교 결과 저장: m_feature_comparison_by_fold.xlsx")

    agg_comparison_df = aggregate_m_comparison(comparison_df)

    if len(agg_comparison_df) > 0:
        agg_comparison_df.to_excel(base_dir / "m_feature_comparison_summary.xlsx", index=False)
        print("✓ 평균 M 피처 비교 결과 저장: m_feature_comparison_summary.xlsx")

        print("\n[M 효과 상위 20개: F2 gain 기준]")
        show_cols = [
            "base_pipeline", "M_pipeline", "balancing", "feature_selection", "model",
            "f1_base_mean", "f1_M_mean", "f1_gain_mean",
            "f2_base_mean", "f2_M_mean", "f2_gain_mean",
            "recall_base_mean", "recall_M_mean", "recall_gain_mean",
            "tnr_base_mean", "tnr_M_mean", "tnr_gain_mean"
        ]

        available_show_cols = [c for c in show_cols if c in agg_comparison_df.columns]

        print(
            agg_comparison_df.sort_values("f2_gain_mean", ascending=False)
            .head(20)[available_show_cols]
            .to_string(index=False)
        )
    else:
        print("M 비교 가능한 결과 없음")

else:
    comparison_df = pd.DataFrame()
    agg_comparison_df = pd.DataFrame()
    print("실험 결과가 비어 있어 M 피처 비교를 건너뜁니다.")

# %% [markdown]
# ## 17. Simple Baseline 결과 요약

# %%
baseline_df = extract_simple_baseline(results_df)
baseline_df.to_excel(base_dir / "simple_baseline_mean_lr_raw.xlsx", index=False)

print("\n[Simple baseline: mean + LR + raw balancing + raw FS]")

if len(baseline_df) > 0:
    print(
        baseline_df[[
            "fold",
            "f1",
            "f1_binary",
            "f2",
            "best_threshold",
            "auc",
            "precision",
            "recall",
            "tnr",
            "tp",
            "tn",
            "fp",
            "fn"
        ]]
        .to_string(index=False)
    )

    print("\nSimple baseline 평균:")
    print(
        baseline_df[[
            "f1",
            "f1_binary",
            "f2",
            "auc",
            "precision",
            "recall",
            "tnr"
        ]]
        .mean()
        .to_string()
    )
else:
    print("Simple baseline 결과 없음")

print("✓ Simple baseline 저장: simple_baseline_mean_lr_raw.xlsx")

# %% [markdown]
# ## 18. 전체 통계 요약

# %%
print("\n[전체 통계 요약]")
print(f"총 성공 실험 수: {len(results_df)}개")

if len(results_df) > 0:
    print(
        f"AUC 계산 가능: "
        f"{(~results_df['auc'].isna()).sum()}개 "
        f"({100 * (~results_df['auc'].isna()).sum() / len(results_df):.1f}%)"
    )

    metric_cols = ["f1", "f1_binary", "f2", "auc", "precision", "recall", "tnr"]

    pipeline_summary = results_df.groupby("pipeline")[metric_cols].agg(["mean", "std"])
    print("\nPipeline별 성능:")
    print(pipeline_summary.to_string())

    balancing_summary = results_df.groupby("balancing")[metric_cols].agg(["mean", "std"])
    print("\nBalancing별 성능:")
    print(balancing_summary.to_string())

    model_summary = results_df.groupby("model")[metric_cols].agg(["mean", "std"])
    print("\nModel별 성능:")
    print(model_summary.to_string())

# %% [markdown]
# ## 19. 결과 시각화 생성

# %%
print("\n단계 7: 결과 시각화 생성 중...")

viz_dir = base_dir / "visualizations"

if len(results_df) > 0:
    create_all_visualizations(results_df, agg_comparison_df, viz_dir)
else:
    print("시각화할 결과 없음")

# %% [markdown]
# ## 20. 분석 완료 및 산출물 안내

# %%
print("\n" + "=" * 70)
print("분석 파이프라인 완료")
print("=" * 70)

print("\n결과 파일:")
print("  - analysis_results.xlsx: 모든 실험 결과, best_threshold 포함")
print("  - summary_by_pipeline.xlsx: pipeline별 요약")
print("  - m_feature_comparison_by_fold.xlsx: fold별 base vs M 비교")
print("  - m_feature_comparison_summary.xlsx: 평균 base vs M 비교")
print("  - simple_baseline_mean_lr_raw.xlsx: simple baseline")
print("  - error_details.xlsx: 스킵/에러 상세")
print("  - visualizations/: 시각화 결과 폴더")
print("    - confusion_matrix_top_f1.png")
print("    - confusion_matrix_top_f2.png")
print("    - roc_curves_top_models_by_f1.png")
