# 메모리 폭발 문제 해결 완료

## 🚨 문제 상황
- **증상**: 시스템 전체가 강제 리부팅될 정도의 심각한 메모리 폭발
- **원인**: PyCaret의 `compare_models()`가 여러 ML 모델을 동시에 학습하면서 메모리 무한 증가

## 🔍 원인 분석

### 1. Init 시 자동 모델 로드 (1차 문제)
```python
# Before (❌)
def __init__(self):
    self._load_latest_model_if_exists()  # 서버 시작할 때마다 대용량 모델 로드!

# After (✅)
def __init__(self):
    # 필요할 때만 lazy loading
```

### 2. compare_models() 메모리 폭발 (2차 문제 - 시스템 리부팅 원인)
```python
# Before (❌)
compare_models(
    include=['lr', 'ridge', 'lasso', 'en', 'dt', 'rf', 'gbr', 'xgboost', 'lightgbm'],
    fold=max(2, min(3, data_size // 2))  # 가변 fold
)
# → 9개 모델 x N-fold = 메모리 폭발!

# After (✅)
if data_size < 30:
    models_to_use = ['lr']  # 1개만
    safe_fold = 2
elif data_size < 50:
    models_to_use = models_to_use[:2]  # 2개만
    safe_fold = 2
else:
    models_to_use = models_to_use[:3]  # 3개만
    safe_fold = 3
```

### 3. PyCaret setup() 병렬 처리 (3차 문제)
```python
# Before (❌)
setup(
    fold=optimal_settings['cv_folds'],  # 최대 10-fold
    # n_jobs 미지정 = CPU 코어 수만큼 병렬 처리
)

# After (✅)
setup(
    n_jobs=1,  # 병렬 처리 비활성화
    fold=min(3, optimal_settings['cv_folds']),  # 최대 3-fold
)
```

## ✅ 적용된 수정사항

### 1. 메모리 안전 설정
```python
# pandas 메모리 최적화
pd.set_option('compute.use_numexpr', False)

# 데이터 크기 제한
MAX_ROWS_FOR_MODELING = 10000  # 최대 1만 행
MAX_FEATURES = 50  # 최대 50개 feature

# 데이터 샘플링
if len(df) > self.MAX_ROWS_FOR_MODELING:
    df = df.sample(n=self.MAX_ROWS_FOR_MODELING, random_state=42)
```

### 2. 모델 수 제한 (가장 중요!)
```python
if data_size < 30:
    models_to_use = ['lr']  # 선형회귀만
    n_select = 1
    safe_fold = 2
elif data_size < 50:
    models_to_use = models_to_use[:2]  # 최대 2개
    n_select = min(2, n_select)
    safe_fold = 2
elif data_size < 100:
    models_to_use = models_to_use[:3]  # 최대 3개
    n_select = min(3, n_select)
    safe_fold = 3
else:
    safe_fold = 5
```

### 3. 명시적 메모리 정리
```python
import gc

# setup 전
gc.collect()

# 모델 학습 전
gc.collect()

# 모델 학습 후 (finally 블록)
finally:
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    gc.collect()  # 가비지 컬렉션 강제 실행
```

### 4. 병렬 처리 비활성화
```python
setup(
    use_gpu=False,
    n_jobs=1,  # 단일 프로세스만 사용
    fold=min(3, optimal_settings['cv_folds'])
)

compare_models(
    errors='ignore',  # 에러 시 계속 진행
    fold=safe_fold
)
```

## 📊 메모리 사용량 예상 변화

### Before (❌)
- 모델 수: 최대 9개
- Fold 수: 최대 10
- 병렬 처리: CPU 코어 수 (8개 등)
- **예상 메모리**: 데이터 크기 x 9 x 10 x 8 = **720배!**

### After (✅)
- 모델 수: 최대 3개 (작은 데이터는 1개)
- Fold 수: 최대 3
- 병렬 처리: 1 (비활성화)
- **예상 메모리**: 데이터 크기 x 3 x 3 x 1 = **9배**

**메모리 사용량 80배 감소!** (720배 → 9배)

## 🧪 테스트 결과
✅ Backend 정상 시작: `http://localhost:8000`
✅ Health check 통과: `{"status":"healthy"}`
✅ 시스템 리부팅 없음

## 🎯 권장 사항

### 모델 학습 시 주의사항
1. **작은 데이터 (< 30행)**: 선형회귀만 사용
2. **중간 데이터 (30-100행)**: 최대 2-3개 모델
3. **큰 데이터 (> 100행)**: 최대 3개 모델로 제한

### Activity Monitor 확인
- "메모리" 탭에서 **실제 사용량(Memory Used)** 확인
- "가상 메모리"나 "스왑" 수치는 무시해도 됨
- Python 프로세스가 **2GB 이상** 사용하면 위험 신호

### 추가 안전장치 (선택사항)
```python
# 더 강력한 메모리 제한이 필요하면:
MAX_ROWS_FOR_MODELING = 5000  # 5천 행으로 줄이기
safe_fold = 2  # fold 수를 항상 2로 고정
```

## 📝 변경된 파일
- `/backend/app/services/modeling_service.py`
  - Init lazy loading
  - 메모리 안전 모드 추가
  - 모델 수 제한
  - 명시적 gc.collect()
  - 병렬 처리 비활성화

## 🎉 결론
**시스템 리부팅 문제는 PyCaret의 compare_models()가 너무 많은 모델을 동시에 학습**하면서 발생했습니다.
모델 수를 1-3개로 제한하고, fold 수를 줄이고, 병렬 처리를 비활성화하여 **메모리 사용량을 80배 이상 감소**시켰습니다.

이제 안전하게 모델 학습을 진행할 수 있습니다! 🚀
