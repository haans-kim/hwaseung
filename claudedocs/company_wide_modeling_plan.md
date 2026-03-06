# R*A와 통합기술본부 적정인력 산정 시스템 구현 계획

## 📋 프로젝트 개요

### 목표
- R*A와 통합기술본부(tonggibon) 각각에 대한 적정인력 산정 모델 구축
- 2025년 데이터를 기반으로 2026년 적정인력 예측
- 기존 전사 적정인력 Dashboard와 동일한 기능 제공

### 핵심 요구사항
1. **데이터 증강**: 원본 데이터를 200개로 증강 (기존 augmentation_service 활용)
2. **모델링**: PyCaret을 사용한 회귀 모델 (각 organization별 독립 모델)
3. **예측**: 최신 연도 데이터로 다음 해 예측
4. **Dashboard**: Feature 중요도, 영향요인 분석, 변수 조정 시뮬레이션, 트렌드 차트

---

## 🏗️ 시스템 아키텍처

### 데이터 구조

#### 입력 데이터 (company_wide_features 테이블)
```sql
- organization: 'R*A' or 'tonggibon'
- year: 연도 (모델링 시 제외)
- headcount: 정원 (TARGET 변수)

-- 공통 Feature (10개)
- ev_growth_gl: 글로벌 EV시장성장률
- v_growth_gl: 글로벌 자동차 시장성장률
- v_export_kr: 국내 자동차 수출액 증가율
- vp_export_kr: 국내 자동차부품 수출액 증가율
- gdp_growth_kr: GDP성장률
- cpi_kr: 소비자물가상승률
- exchange_rate_change_krw: 환율변화율
- scm_index_gl: 글로벌물류비지수
- oil_gl: 국제유가
- labor_cost: 인건비 증감률

-- R*A 전용 Feature (4개)
- revenue: 매출액 증감률
- profit: 영업이익 증감률
- operating_rate: 가동률 증감률
- operating_date: 가동일수 증감률

-- tonggibon 전용 Feature (4개)
- revenue: 매출액 증가율
- profit: 영업이익 증가율
- operating_rate: 연구개발비용 증감률 (컬럼 재사용)
- operating_date: 연구개발정부보조금 증감률 (컬럼 재사용)
```

### 모델 관리 전략

#### 1. Organization별 독립 모델
- **모델 파일**:
  - `company_wide_model_R*A_latest.pkl`
  - `company_wide_model_tonggibon_latest.pkl`
- **PyCaret 세션**: 각 organization별로 독립적인 세션 관리
- **모델 메타데이터**: DB에 저장 (organization, model_type, metrics, created_at)

#### 2. 데이터 증강
- **방법**: `augmentation_service.smart_augment()` 사용
- **원본 데이터 < 20개** → **200개로 증강**
- **증강 알고리즘**:
  - Gaussian Noise (작은 데이터셋)
  - Mixup (충분한 데이터셋)
- **보호 컬럼**: headcount (target), year

#### 3. Feature 처리
- **연도 제외**: 시계열이 아닌 회귀 문제로 접근
- **Target**: headcount (정원)
- **Feature 수**: 공통 10개 + 조직별 4개 = 14개

---

## 🔧 구현 단계

### Phase 1: 백엔드 - 모델링 서비스

#### 1.1. company_wide_modeling_service.py
```python
class CompanyWideModelingService:
    def __init__(self):
        # Organization별 모델 관리
        self.models = {
            'R*A': None,
            'tonggibon': None
        }
        self.experiments = {
            'R*A': None,
            'tonggibon': None
        }

    def prepare_data(self, organization: str):
        """
        1. company_wide_features 테이블에서 데이터 로드
        2. organization 필터링
        3. year 컬럼 제거
        4. headcount를 target으로 설정
        """

    def augment_data(self, organization: str, target_size: int = 200):
        """
        augmentation_service.smart_augment() 활용
        - target_column: 'headcount'
        - year_column: 'year' (보호)
        - target_size: 200
        """

    def setup_pycaret(self, organization: str):
        """
        PyCaret 환경 설정
        - 증강된 데이터 사용
        - target: headcount
        - session_id: organization별 고정값
        """

    def compare_models(self, organization: str):
        """
        작은 데이터셋용 모델 비교
        - 모델: lr, ridge, lasso, en, dt, rf, gbr
        - 정렬: R2 score
        """

    def train_model(self, organization: str, model_name: str):
        """
        선택된 모델 학습
        - tune_model로 하이퍼파라미터 튜닝
        - finalize_model로 전체 데이터 학습
        - organization별 파일로 저장
        """

    def predict_2026(self, organization: str):
        """
        2026년 예측
        - 2025년 데이터 (최신 연도) 사용
        - headcount 제외한 모든 feature 입력
        """
```

#### 1.2. API 엔드포인트 (/api/company-wide/modeling)
```python
POST /setup
- Body: { "organization": "R*A" or "tonggibon" }
- 동작: 데이터 준비 + 증강 + PyCaret setup
- 응답: { "message", "data_info", "augmented_size" }

POST /compare
- Body: { "organization": "R*A" or "tonggibon" }
- 동작: 모델 비교
- 응답: { "models": [...], "best_model", "comparison_df" }

POST /train
- Body: { "organization": "R*A" or "tonggibon", "model_name": "rf" }
- 동작: 특정 모델 학습
- 응답: { "message", "model_type", "metrics" }

GET /status
- Query: organization
- 동작: 현재 모델링 상태 확인
- 응답: { "has_data", "has_model", "model_type", "trained_at" }
```

---

### Phase 2: 백엔드 - Dashboard 서비스

#### 2.1. company_wide_dashboard_service.py
```python
class CompanyWideDashboardService:
    def get_2026_prediction(self, organization: str):
        """
        2026년 적정인력 예측
        - 학습된 모델 로드
        - 2025년 데이터로 예측
        - 전년(2025년) 대비 증감 계산
        """

    def get_feature_importance(self, organization: str):
        """
        Permutation Importance 계산
        - sklearn.inspection.permutation_importance
        - 상위 10개 feature 반환
        """

    def simulate_scenario(self, organization: str, variables: dict):
        """
        변수 조정 시뮬레이션
        - 사용자가 조정한 변수값으로 예측
        - 기본값 대비 변화량 계산
        """

    def get_trend_data(self, organization: str):
        """
        트렌드 데이터 생성
        - 과거 실적: company_wide_features 테이블
        - 2026년 예측: 모델 예측값
        - 시각화용 JSON 반환
        """
```

#### 2.2. API 엔드포인트 (/api/company-wide/dashboard)
```python
GET /prediction
- Query: organization
- 응답: {
    "year": 2026,
    "predicted_headcount": 527,
    "previous_headcount": 545,
    "change": -18,
    "change_percent": -3.3,
    "model_r2": 0.85
  }

GET /importance
- Query: organization
- 응답: {
    "features": [
      { "name": "국내 자동차부품 수출액 증가율", "importance": 0.18 },
      ...
    ]
  }

POST /simulate
- Body: {
    "organization": "R*A",
    "variables": {
      "oil_gl": -13.7,
      "exchange_rate_change_krw": 4.2,
      ...
    }
  }
- 응답: {
    "predicted_headcount": 530,
    "baseline_headcount": 527,
    "change": 3
  }

GET /trend
- Query: organization
- 응답: {
    "years": [2021, 2022, 2023, 2024, 2025, 2026],
    "actual": [500, 520, 540, 550, 545, null],
    "predicted": [null, null, null, null, null, 527]
  }
```

---

### Phase 3: 프론트엔드

#### 3.1. 모델링 페이지 (CompanyWideModeling.tsx)
```typescript
// 레이아웃
- Organization 선택 탭 (R*A / tonggibon)
- 데이터 현황 카드
  - 원본 데이터 수
  - 증강 데이터 수
  - Feature 수
- 모델링 워크플로우
  1. Setup 버튼 → 데이터 증강 + 환경 설정
  2. Compare 버튼 → 모델 비교 결과 테이블
  3. 모델 선택 → 드롭다운
  4. Train 버튼 → 선택된 모델 학습
- 학습 결과
  - 모델 타입
  - R2 score
  - MAE, RMSE
```

#### 3.2. Dashboard 페이지 (CompanyWideDashboard.tsx)
```typescript
// 기존 Dashboard와 동일한 구조

// 상단 4개 카드
1. 2026년 적정인력 (예측값)
2. 증감 (전년 대비)
3. 모델 정확도 (R2 score)
4. 2026년 예상 직원수 (시나리오 기반)

// 변수 조정 섹션
- 슬라이더로 주요 변수 조정
- 실시간 예측 업데이트
- 변수별 min/max 범위 설정

// 트렌드 분석 차트
- Line chart
- 과거 실적 (2021-2025)
- 2026년 예측값 (점선)

// 영향 요인 분석
- Horizontal bar chart
- Permutation Importance
- 상위 10개 feature
```

---

## 🔄 워크플로우

### 사용자 시나리오

#### 1. 모델 학습 단계
```
1. 데이터 업로드 페이지 → R*A 또는 tonggibon 데이터 업로드
2. 모델링 페이지 이동 → organization 선택
3. Setup 클릭 → 데이터 증강 (200개)
4. Compare 클릭 → 모델 비교 (R2 기준 정렬)
5. 최적 모델 선택 → Train 클릭
6. 학습 완료 → Dashboard 이동
```

#### 2. Dashboard 활용 단계
```
1. Dashboard 접속 → organization 선택
2. 2026년 예측 확인
3. Feature importance 확인
4. 변수 조정 시뮬레이션
5. 트렌드 분석
```

---

## 📊 데이터 플로우

```
[Excel Upload]
    ↓
[company_wide_features 테이블]
    ↓
[Organization 필터링]
    ↓
[데이터 증강 (200개)]
    ↓
[PyCaret Setup (year 제외)]
    ↓
[모델 비교 & 선택]
    ↓
[모델 학습 & 저장]
    ↓
[2026년 예측]
    ↓
[Dashboard 시각화]
```

---

## 🎯 핵심 기술 포인트

### 1. 데이터 증강
- **문제**: 원본 데이터가 4-5개로 매우 적음
- **해결**: augmentation_service로 200개 증강
- **방법**: Gaussian Noise 또는 Mixup
- **주의**: target(headcount)과 year는 보호

### 2. 모델 선택
- **작은 데이터셋**: lr, ridge, lasso, en, dt
- **중간 데이터셋**: + rf, gbr
- **큰 데이터셋**: + xgboost, lightgbm
- **기준**: R2 score

### 3. Feature Engineering
- **연도 제외**: 회귀 문제로 접근
- **조직별 Feature**: 같은 컬럼 재사용, 의미만 다르게 해석
- **정규화**: PyCaret에서 자동 처리

### 4. 예측 방식
- **입력**: 2025년 데이터 (최신 연도)
- **출력**: 2026년 headcount 예측
- **시뮬레이션**: 변수값 조정하여 재예측

---

## ✅ 검증 항목

### 모델링
- [ ] R*A와 tonggibon 각각 독립적으로 모델 학습 가능
- [ ] 데이터 증강이 200개로 정확히 동작
- [ ] 연도 컬럼이 Feature에서 제외됨
- [ ] 모델이 organization별로 저장됨

### Dashboard
- [ ] 2026년 예측값이 정확히 표시
- [ ] Feature importance가 계산됨
- [ ] 변수 조정 시 실시간 예측 업데이트
- [ ] 트렌드 차트에 과거 + 예측 표시

### 통합
- [ ] 모델링 → Dashboard 전환이 원활함
- [ ] organization 변경 시 데이터 자동 갱신
- [ ] 에러 처리 및 사용자 피드백

---

## 📅 구현 순서

1. ✅ **계획 수립 및 아키텍처 설계**
2. **백엔드 - 모델링 서비스** (company_wide_modeling_service.py)
3. **백엔드 - 모델링 API** (routes/company_wide_modeling.py)
4. **백엔드 - Dashboard 서비스** (company_wide_dashboard_service.py)
5. **백엔드 - Dashboard API** (routes/company_wide_dashboard.py)
6. **프론트엔드 - 모델링 페이지** (CompanyWideModeling.tsx)
7. **프론트엔드 - Dashboard 페이지** (CompanyWideDashboard.tsx)
8. **통합 테스트 및 검증**

---

## 🔍 참고 자료

### 기존 코드 재사용
- `modeling_service.py`: 모델링 로직 참고
- `dashboard_service.py`: Dashboard 기능 참고
- `augmentation_service.py`: 데이터 증강
- 기존 Dashboard UI: 동일한 구조 재사용

### 주요 라이브러리
- **PyCaret**: 모델링 및 AutoML
- **pandas/numpy**: 데이터 처리
- **sklearn**: Permutation importance
- **React**: 프론트엔드
- **Recharts**: 차트 시각화
