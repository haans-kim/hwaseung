# SambioWage Backend API

FastAPI 기반 머신러닝 임금인상률 예측 백엔드 서버

## 설치 및 실행

### 1. 가상환경 활성화
```bash
source ../.venv/bin/activate
```

### 2. 서버 실행
```bash
# 개발 모드 (자동 리로드)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 또는 run.py 사용
python run.py
```

### 3. API 문서 확인
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### 기본
- `GET /` - API 정보
- `GET /health` - 헬스체크

### 데이터 관리 (`/api/data`)
- `POST /api/data/upload` - Excel/CSV 파일 업로드
- `GET /api/data/sample` - 샘플 데이터 조회
- `GET /api/data/info` - 데이터 정보

### 모델링 (`/api/modeling`)
- `POST /api/modeling/setup` - PyCaret 환경 설정
- `POST /api/modeling/compare` - 모델 비교
- `POST /api/modeling/train/{model_name}` - 모델 학습
- `GET /api/modeling/status` - 모델링 상태

### 분석 (`/api/analysis`)
- `GET /api/analysis/explainer` - ExplainerDashboard 데이터
- `GET /api/analysis/shap` - SHAP 분석
- `GET /api/analysis/lime/{instance_id}` - LIME 설명
- `GET /api/analysis/feature-importance` - Feature 중요도

### 대시보드 (`/api/dashboard`)
- `POST /api/dashboard/predict` - 임금인상률 예측
- `POST /api/dashboard/scenario-analysis` - 시나리오 분석
- `GET /api/dashboard/variables` - 사용 가능한 변수
- `GET /api/dashboard/charts` - 차트 데이터

## 프로젝트 구조

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 애플리케이션
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── data.py      # 데이터 업로드/관리
│   │       ├── modeling.py  # ML 모델링
│   │       ├── analysis.py  # 모델 분석
│   │       └── dashboard.py # 대시보드 API
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py        # 설정
│   ├── models/              # 데이터 모델
│   ├── services/            # 비즈니스 로직
│   └── utils/               # 유틸리티
├── run.py                   # 서버 실행 스크립트
└── README.md
```

## 설정

환경변수는 `.env` 파일에서 관리됩니다:

```env
# CORS 설정
ALLOWED_HOSTS=["http://localhost:3000"]

# 파일 업로드
MAX_FILE_SIZE=10485760  # 10MB
UPLOAD_DIR=uploads

# 모델 저장
MODEL_DIR=models
```