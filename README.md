# B.S. Graduation Thesis in Industrial Engineering

## Overview

본 프로젝트는 결측치와 클래스 불균형이 동시에 존재하는 반도체 제조 공정 데이터에서 머신러닝 기반 불량 검출 성능을 분석한 학사 졸업논문 프로젝트다. 반도체 제조 공정에서는 다수의 센서 변수가 수집되지만, 실제 데이터에는 센서 측정 실패, 설비 이상, 공정 불안정 등으로 인한 결측치가 빈번하게 발생했다. 또한 불량 샘플의 비율이 정상 샘플에 비해 매우 낮기 때문에, 일반적인 머신러닝 모델은 정상 클래스로 편향된 예측을 수행하기 쉬운 문제가 있었다.

따라서 본 연구에서는 UCI SECOM 데이터셋을 기반으로 결측치 대치 방법, 클래스 불균형 처리 기법, 특성 선택 방법, 분류 모델을 조합한 다양한 머신러닝 파이프라인의 성능을 비교하였다. 특히 결측값을 단순히 제거하거나 복원해야 하는 대상으로만 보지 않고, 결측 발생 여부 자체가 공정 이상과 관련된 정보를 포함할 수 있다는 관점에서 Missing Indicator, 즉 M feature의 활용 가능성을 함께 분석하였다.

총 924개의 파이프라인 조합을 구성하였으며, 데이터 누출을 방지하기 위해 모든 전처리 과정을 9-Fold 교차검증 루프 내부에서 수행하였다. 모델 성능은 불량 샘플 탐지 능력을 중점적으로 평가하기 위해 F2-score, Recall, TNR을 기준으로 비교하였다.

## Research Objective

본 연구의 주요 목적은 결측치와 클래스 불균형이 동시에 존재하는 반도체 공정 데이터 환경에서 어떤 머신러닝 파이프라인 구성이 불량 검출에 효과적인지 분석하는 것이다.
구체적인 연구 목표는 다음과 같다.

* 반도체 공정 데이터의 결측치와 클래스 불균형 문제가 머신러닝 기반 불량 검출 성능에 미치는 영향을 분석하였다.
* Mean Imputation, In-painting KNN, MICE, MissForest 등 다양한 결측치 대치 방법의 성능 차이를 비교하였다.
* SMOTE 적용 여부에 따라 불량 샘플 탐지 성능이 어떻게 달라지는지 평가하였다.
* f_classif, mutual information, FPR, FDR, FWE 등 다양한 특성 선택 기법이 모델 성능에 미치는 영향을 분석하였다.
* KNN, SVM, Logistic Regression, Random Forest, XGBoost, HGB 등 여러 분류 모델을 비교하여 안정적인 파이프라인 조합을 도출하였다.
* 결측 여부를 이진 변수로 표현한 결측지시변수(Missing Indicator), 즉 M feature가 불량 검출 성능 향상에 보조적인 정보를 제공할 수 있는지 검증하였다.

## Dataset

본 연구에서는 UCI Machine Learning Repository에서 제공하는 SECOM 데이터셋을 사용하였다. SECOM 데이터셋은 반도체 제조 공정에서 수집된 센서 기반 공정 데이터로, 고차원 변수와 결측치, 클래스 불균형 문제를 동시에 포함하고 있어 반도체 불량 검출 연구에서 자주 활용되는 벤치마크 데이터셋이다.

데이터셋의 주요 특징은 다음과 같다.

* Dataset: UCI SECOM Dataset(2008)
* Domain: Semiconductor Manufacturing Process
* Task: Binary Classification
* Target Variable: 정상 샘플과 불량 샘플 구분
* Data Characteristics:

  * 다수의 센서 변수로 구성된 고차원 공정 데이터
  * 광범위한 결측치 포함
  * 정상 샘플에 비해 불량 샘플 수가 매우 적은 클래스 불균형 구조
  * 결측 발생 패턴이 공정 이상과 관련된 보조 정보일 가능성 존재

본 연구에서는 결측률이 50% 이상인 변수와 분산이 매우 낮은 변수를 제거한 뒤, 정제된 데이터를 기준으로 다양한 결측치 대치 및 분류 파이프라인을 구성하였다. 또한 기존 공정 변수뿐 아니라 결측 발생 여부를 나타내는 Missing Indicator를 추가하여 결측 패턴 자체의 정보적 활용 가능성을 분석하였다.

## Project Structure

```text
Graduate_Thesis_Project/
├── README.md
├── data/
│   ├── raw/
│   │   └── raw.xlsx
│   └── processed/
│       ├── secom_full_with_labels.csv
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── Y_train.csv
│       └── Y_test.csv
├── notebooks/
│   ├── Base.ipynb
│   └── Base_before.ipynb
├── src/
│   └── Base.py
├── checkpoint/
├── kfold/
├── kfold_M/
├── kfold_inpaintknn/
├── kfold_mice/
├── kfold_missforest/
├── kfold_M_inpaintknn/
├── kfold_M_mice/
├── kfold_M_missforest/
├── visualizations/
├── results/
│   ├── analysis_results.xlsx
│   ├── m_feature_comparison_by_fold.xlsx
│   ├── m_feature_comparison_summary.xlsx
│   ├── paired_ttest_results.xlsx
│   ├── simple_baseline_mean_lr_raw.xlsx
│   └── summary_by_pipeline.xlsx
└── .gitignore
```
