# 중요 파일 목록 (Git에서 제외됨)

## 꼭 필요한 파일들
이 파일들은 `.gitignore`에 포함되어 있어 Git으로 전송되지 않습니다.
맥미니로 수동 전송이 필요합니다.

### 1. 데이터 파일 (`backend/data/`)
- `current_data.pkl` - 현재 작업 데이터
- `master_data.pkl` - 마스터 데이터
- `working_data.pkl` - 작업 중인 데이터

### 2. 모델 파일 (`backend/models/`)
- `latest.pkl` - 최신 학습 모델 ⭐️
- `wage_model_*.pkl` - 저장된 모델들

### 3. 업로드 파일 (`backend/uploads/`)
- 사용자가 업로드한 Excel/CSV 파일들

### 4. 저장된 모델 (`backend/saved_models/`)
- `baseup_model.pkl` - Base-up 모델
- `performance_model.pkl` - 성과급 모델
- `current_model.pkl` - 현재 모델

## 전송 방법

### 방법 1: 전체 자동 전송 스크립트
```bash
# transfer-to-mac-mini.sh 수정 (호스트 정보 입력)
vi transfer-to-mac-mini.sh

# 실행
./transfer-to-mac-mini.sh
```

### 방법 2: 수동 전송
```bash
# 데이터 파일
scp backend/data/*.pkl user@mac-mini:~/sambiowage/backend/data/

# 모델 파일
scp backend/models/*.pkl user@mac-mini:~/sambiowage/backend/models/

# 업로드 폴더
scp -r backend/uploads/ user@mac-mini:~/sambiowage/backend/
```

### 방법 3: USB/외장 드라이브
1. 필요한 파일들을 압축
```bash
tar -czf sambiowage-data.tar.gz \
  backend/data/*.pkl \
  backend/models/*.pkl \
  backend/uploads/
```

2. USB로 복사 후 맥미니에서 압축 해제
```bash
tar -xzf sambiowage-data.tar.gz
```

## 주의사항
- 이 파일들 없이는 앱이 제대로 동작하지 않습니다
- 특히 `latest.pkl` 모델 파일이 중요합니다
- 데이터 파일이 없으면 대시보드가 빈 화면으로 나타납니다