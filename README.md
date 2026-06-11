<div align="center">



<h1>Machine Learning Pipeline Analysis for Improving Semiconductor Defect Detection Performance in Missing Value and Class Imbalanced Environments</h1>



<h2>결측치 및 클래스 불균형 환경에서의 반도체 불량 검출 성능 향상을 위한 머신러닝 파이프라인 분석</h2>



<p>

  <h3>설계에 사용된 노트는 <code>notebooks/02_baseline.ipynb</code>을 참조 바랍니다.

</p>



<p>

  <h3>논문 원본 및 설계에 인용된 결과 Table, 시각화 자료, figures는 <code>docs</code> 폴더를 참조 바랍니다.<br>

  <h3>(논문 및 figures는 <code>docs/paper</code>, 이외 Table 및 시각화 자료는 <code>docs/results</code>)

</p>



</div>

## Overview

본 프로젝트는 결측치와 클래스 불균형이 동시에 존재하는 반도체 제조 공정 데이터에서 머신러닝 기반 불량 검출 성능을 분석한 학사 졸업논문 프로젝트다. 반도체 제조 공정에서는 다수의 센서 변수가 수집되지만, 실제 데이터에는 센서 측정 실패, 설비 이상, 공정 불안정 등으로 인한 결측치가 빈번하게 발생했다. 또한 불량 샘플의 비율이 정상 샘플에 비해 매우 낮기 때문에, 일반적인 머신러닝 모델은 정상 클래스로 편향된 예측을 수행하기 쉬운 문제가 있었다.

따라서 본 연구에서는 UCI SECOM 데이터셋을 기반으로 결측치 대치 방법, 클래스 불균형 처리 기법, 특성 선택 방법, 분류 모델을 조합한 다양한 머신러닝 파이프라인의 성능을 비교하였다. 특히 결측값을 단순히 제거하거나 복원해야 하는 대상으로만 보지 않고, 결측 발생 여부 자체가 공정 이상과 관련된 정보를 포함할 수 있다는 관점에서 Missing Indicator, 즉 M feature의 활용 가능성을 함께 분석하였다.

총 924개의 파이프라인 조합을 구성하였으며, 데이터 누출을 방지하기 위해 모든 전처리 과정을 9-Fold 교차검증 루프 내부에서 수행하였다. 모델 성능은 불량 샘플 탐지 능력을 중점적으로 평가하기 위해 F2-score, Recall, TNR을 기준으로 비교하였다.

## Research Objective

본 연구의 주요 목적은 결측치와 클래스 불균형이 동시에 존재하는 반도체 공정 데이터 환경에서 어떤 머신러닝 파이프라인 구성이 불량 검출에 효과적인지 분석하는 것이다.
구체적인 연구 목표는 다음과 같다.

* 반도체 공정 데이터의 결측치와 클래스 불균형 문제가 머신러닝 기반 불량 검출 성능에 미치는 영향을 분석하였다.
* Mean Imputation, In-painting KNN, MICE, MissForest 등 다양한 결측치 대치 방법의 성능 차이를 비교하였다.
* 오버샘플링 적용 여부에 따라 불량 샘플 탐지 성능이 어떻게 달라지는지 평가하였다.
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

### 실험 파이프라인 구조

전체 데이터 분석 및 모델링 과정은 다음과 같은 순서로 진행하였다.

```text
0. 데이터 정제
   └── 1. 결측치 대치
       └── 2. 클래스 불균형 처리
           └── 3. 특성 선택
               └── 4. 머신러닝 분류 모델
                   └── 5. 성능 평가
```

#### 0. 데이터 정제

원본 SECOM 데이터셋에 대해 두 단계의 정제 과정을 수행하였다.

```text
원본 SECOM Dataset
├── 1. 결측률이 50% 이상인 컬럼 제거
├── 2. 분산이 0.05 이하인 컬럼 제거
└── 이후 정제된 Data를 Raw 로 정의
```

본 연구에서는 결측률이 50% 이상인 컬럼을 제거한 뒤, 추가적으로 분산이 0.05 이하인 컬럼을 제거하였다. 이후 정제된 데이터를 **Raw**라고 지칭하였다.

#### 1. 결측치 대치

결측치 대치 과정은 데이터 누출을 방지하기 위해 K-Fold 교차검증 내부에서 수행하였다. 본 연구에서는 K=9를 적용한 9-Fold 교차검증 구조를 사용하였다.

결측치 대치 단계에서는 총 11가지 조건을 비교하였다.

```text
1. Raw
2. Raw + M
3. Mean
4. Mean + M
5. Inpainting KNN
6. Inpainting KNN + M
7. MICE
8. MICE + M
9. MissForest
10. MissForest + M
11. M_only
```

여기서 **M**은 결측 지시변수, 즉 Missing Indicator를 의미한다. M feature는 각 센서 변수의 값이 결측인지 아닌지를 이진 변수 형태로 나타낸 것이다.

특히 **M_only** 조건은 원래의 공정 변수는 사용하지 않고 결측 발생 여부만을 입력 변수로 사용한 조건이다. 이를 통해 결측 패턴 자체가 불량 검출에 유의미한 정보를 포함하는지 분석하고자 하였다.

#### 2. 클래스 불균형 처리

SECOM 데이터셋은 정상 샘플에 비해 불량 샘플의 수가 매우 적은 클래스 불균형 구조를 가진다. 이를 고려하여 다음 두 가지 조건을 비교하였다.

```text
1. Raw
2. SMOTE
```

Raw 조건은 별도의 클래스 불균형 처리를 적용하지 않은 경우이다. SMOTE 조건은 소수 클래스인 불량 샘플을 오버샘플링하여 클래스 불균형을 완화한 경우이다.

#### 3. 특성 선택

고차원 센서 데이터에서 불필요한 변수를 제거하고 모델의 학습 효율을 높이기 위해 총 7가지 특성 선택 조건을 비교하였다.

```text
단일변수 선택법
├── chi_square
├── f_classif
└── mutual_info

통계적 유의성 기반 선택법
├── FPR
├── FDR
└── FWE

특성 선택 미적용
└── Raw
```

단일변수 선택법에는 Chi_Square 검정, ANOVA F-test, 상호정보량 기반 방법을 사용하였다. 통계적 유의성 기반 선택법에는 FPR, FDR, FWE를 사용하였다. 또한 특성 선택을 수행하지 않은 Raw 조건도 함께 비교하였다.

#### 4. 머신러닝 분류 모델

최종 분류 모델로는 총 6가지 머신러닝 모델을 사용하였다.

```text
1. KNN
2. SVM
3. Logistic Regression
4. Random Forest
5. XGBoost
6. HGB
```

이를 통해 거리 기반 / 비선형, 선형 / 트리기반 앙상블 / 부스팅 기반 비선 모델이 결측치 및 클래스 불균형 환경에서 어떤 성능 차이를 보이는지 비교하였다.

#### 5. 성능 평가 지표

모델 성능은 세 가지 지표를 기준으로 평가하였다.

```text
1. F2-score
2. Recall
3. TNR
```

Recall은 실제 불량 샘플을 모델이 얼마나 잘 탐지했는지를 나타내는 지표로, 공정Data에서는 불량을 정상으로 잘못 판단하는 것이 큰 비용을 초래할 수 있는 도메인 특성에 따라 Recall을 중요한 지표로 사용하였다.

F2-score는 Precision보다 Recall에 더 큰 가중치를 부여하는 지표로, 불량 탐지가 중요 목적인 이 연구에 적합한 평가 지표이다.

TNR은 실제 정상 샘플을 정상으로 올바르게 분류한 비율이다. Recall만 높이는 모델은 정상 샘플까지 과도하게 불량으로 예측할 수 있으므로, 정상 샘플 구분 성능을 함께 확인하기 위해 TNR을 사용하였다.

#### 전체 파이프라인 조합 수

```text
11개 결측치 대치 조건
× 2개 클래스 불균형 처리 조건
× 7개 특성 선택 조건
× 6개 분류 모델
= 총 924개 파이프라인 조합
```

총 **924개의 머신러닝 파이프라인 조합**을 체계적으로 비교·분석하였다.

## Repository Structure

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
│   ├── thesis_summary.md             
│   ├── Paper/                        
│   └── results/                      
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
│   └── 02_Baseline.ipynb             
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

### Directory

| 디렉터리 / 파일                                   | 설명                                                                                              |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `.github/`                                  | GitHub 설정 파일 및 Actions workflows 저장 폴더               |
| `.gitignore`                                | Git 추적에서 제외할 파일 및 폴더를 지정         |
| `README.md`                                 | 프로젝트 개요, 연구 목적, 데이터셋, 실험 파이프라인, 저장소 구조를 설명                                           |
| `environment.yml`                           | Conda 환경 재현 설정 / 프로젝트 실행에 필요한 Python 버전 + 주요 패키지 의존성                              |
| `requirements.txt`                          | pip 기반 패키지 설치를 위한 의존성 목록 파일   |
| `Old/`                                      | 이전 버전의 코드, 노트북, 결과물 또는 더 이상 사용하지 않는 파일 보관 / 과거 파일 분리 |
| `checkpoint/`                               | 실험 중간 저장 파일, 모델 체크포인트                                                   |
| `data/`                                     | 프로젝트 데이터 파일 저장소 (원본 데이터 & 전처리 데이터 구분)                                      |
| `data/raw/`                                 | 원본 SECOM 데이터셋                                                          |
| `data/processed/`                           | 정제된 데이터 (결측률 기준 제거, 저분산 변수 제거, train/test 분할 등)                                |
| `docs/`                                     | 논문 요약, 방법론 설명, 추가 분석 문서                             |
| `docs/thesis_summary.md`                    | 졸업논문의 연구 배경, 방법론, 실험 결과, 결론 요약                                             |
| `docs/Paper/`                               | 논문 원문 및 관련 자료 저장                                           |
| `docs/results/`                             | 추가 분석 결과 및 중간 결과물 저장                                             |
| `experiments/`                              | 결측치 대치 방법 + M feature 적용 여부 별 K-Fold 실험 결과 저장                                         |
| `experiments/mean, mean_M, M_only.txt`      | Mean, Mean + M, M_only 조건 실험 참고사항                                             |
| `experiments/kfold/`                        | 기본 Raw 조건 기반 K-Fold 실험 결과 저장                                                           |
| `experiments/kfold_M/`                      | Missing Indicator(M feature) 추가한 K-Fold 실험 결과 저장                                    |
| `experiments/kfold_inpaintknn/`             | Inpainting KNN 결측치 대치 K-Fold 실험 결과 저장                                          |
| `experiments/kfold_mice/`                   | MICE 결측치 대치 K-Fold 실험 결과 저장                                                 |
| `experiments/kfold_missforest/`             | MissForest 결측치 대치 K-Fold 실험 결과 저장                                              |
| `experiments/kfold_M_inpaintknn/`           | Inpainting KNN 결측치 대치 + M feature K-Fold 실험 결과 저장                               |
| `experiments/kfold_M_mice/`                 | MICE 결측치 대치 + M feature K-Fold 실험 결과 저장                                         |
| `experiments/kfold_M_missforest/`           | MissForest 결측치 대치 + M feature K-Fold 실험 결과 저장                                   |
| `notebooks/`                                | Jupyter Notebook(데이터 분석, 전처리, 모델 실험, 결과 확인)                                  |
| `notebooks/00_archive/`                     | 구버전 노트북                                              |
| `notebooks/02_Baseline.ipynb`               | baseline 실험 핵심파일 = 주요 분석 과정 Jupyter Notebook                                         |
| `results/`                                  | 최종 실험 결과 + 성능 비교표 + 요약 분석 결과 저장 (시각화)                                                          |
| `results/analysis_results.xlsx`             | 전체 실험 결과 + 주요 성능 분석 결과 정리                                                           |
| `results/m_feature_comparison_by_fold.xlsx` | Fold별 M feature 추가 효과 비교                                                             |
| `results/m_feature_comparison_summary.xlsx` | M feature 추가 여부에 따른 성능 차이 비교, 요약                                                         |
| `results/simple_baseline_mean_lr_raw.xlsx`  | Mean imputation, Logistic Regression, Raw 조건 등 simple baseline 조합 결과 정리                    |
| `results/summary_by_pipeline.xlsx`          | 파이프라인 조합별 성능 요약 (최종 결과 확인)                                                                   |
| `results/visualizations/`                   | 실험 결과 시각화(그래프, 그림, 차트)                                                          |
| `src/`                                      | Python 소스코드                                                       |
| `src/Base.py`                               | 데이터 전처리, 결측치 대치, feature selection, 모델 학습 및 평가, 주요실험코드 py                     |
