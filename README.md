# B.S. Graduation Thesis in Industrial Engineering
### 설계에 사용된 노트는 notebooks 폴더의 01_baseline을 참조바랍니다.

## Overview

본 프로젝트는 결측치와 클래스 불균형이 동시에 존재하는 반도체 제조 공정 데이터에서 머신러닝 기반 불량 검출 성능을 분석한 학사 졸업논문 프로젝트입니다.

결측치(Missing Value)와 클래스 불균형(Class Imbalance)은 머신러닝 실무에서 매우 흔히 나타나는 두 가지 문제이며, 이 두 문제가 모델 성능에 미치는 영향을 체계적으로 분석하기 위해 본 연구를 수행하였습니다.

따라서 본 연구에서는 UCI SECOM 데이터셋을 기반으로 결측치 대치 방법, 클래스 불균형 처리 기법, 특성 선택 방법, 분류 모델을 조합한 다양한 머신러닝 파이프라인을 구성하고 각 구성 요소가 불량 검출 성능에 미치는 영향을 정량적으로 분석하였습니다.

총 924개의 파이프라인 조합을 구성하였으며, 데이터 누출을 방지하기 위해 모든 전처리 과정을 9-Fold 교차검증 루프 내부에서 수행하였다. 모델 성능은 F1-Score, Precision, Recall, AUC-ROC 등 다양한 평가 지표로 측정하였습니다.

## Research Objective

본 연구의 주요 목적은 결측치와 클래스 불균형이 동시에 존재하는 반도체 공정 데이터 환경에서 어떤 머신러닝 파이프라인 구성이 불량 검출에 효과적인지를 체계적으로 규명하고, 이를 통해 실무적 가이드라인을 제시하는 것입니다.

구체적인 연구 목표는 다음과 같다.

* 반도체 공정 데이터의 결측치와 클래스 불균형 문제가 머신러닝 기반 불량 검출 성능에 미치는 영향을 분석하였다.
* Mean Imputation, In-painting KNN, MICE, MissForest 등 다양한 결측치 대치 방법의 성능 차이를 비교하였다.
* SMOTE 적용 여부에 따라 불량 샘플 탐지 성능이 어떻게 달라지는지 평가하였다.
* f_classif, mutual information, FPR, FDR, FWE 등 다양한 특성 선택 기법이 모델 성능에 미치는 영향을 분석하였다.
* KNN, SVM, Logistic Regression, Random Forest, XGBoost, HGB 등 여러 분류 모델을 비교하여 안정적인 파이프라인 조합을 도출하였다.
* 결측 여부를 이진 변수로 표현한 결측지시변수(Missing Indicator), 즉 M feature가 불량 검출 성능 향상에 보조적인 정보를 제공할 수 있는지 검증하였다.

## Dataset

본 연구에서는 UCI Machine Learning Repository에서 제공하는 SECOM 데이터셋을 사용하였다. SECOM 데이터셋은 반도체 제조 공정에서 수집된 센서 기반 공정 데이터로 광범위한 결측치와 뚜렷한 클래스 불균형을 특징으로 합니다.

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

본 연구에서는 결측률이 50% 이상인 변수와 분산이 매우 낮은 변수를 제거한 뒤, 정제된 데이터를 기준으로 다양한 결측치 대치 및 분류 파이프라인을 구성하였습니다.

## Project Structure

```text
Graduate_Thesis_Project/
├── .github/
├── .gitignore
├── README.md
├── environment.yml
├── requirements.txt
├── Old/
├── checkpoint/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
│   └── thesis_summary.md
├── experiments/
│   ├── mean, mean_M, M_only.txt
│   ├── kfold/
│   ├── kfold_M/
│   ├── kfold_inpaintknn/
│   ├── kfold_mice/
│   ├── kfold_missforest/
│   ├── kfold_M_inpaintknn/
│   ├── kfold_M_mice/
│   └── kfold_M_missforest/
├── notebooks/
│   ├── 00_archive/
│   └── 01_Baseline.ipynb
├── results/
│   ├── analysis_results.xlsx
│   ├── m_feature_comparison_by_fold.xlsx
│   ├── m_feature_comparison_summary.xlsx
│   ├── simple_baseline_mean_lr_raw.xlsx
│   ├── summary_by_pipeline.xlsx
│   └── visualizations/
└── src/
    └── Base.py
```
