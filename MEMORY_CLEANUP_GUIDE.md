# 메모리 관리 가이드

## 🔴 메모리 누수 문제 해결

VSCode가 300GB 메모리를 사용하며 죽는 문제를 해결하기 위한 개선 사항입니다.

## 문제 원인

### 1. PyCaret + 싱글톤 패턴
- 서비스가 프로세스 종료까지 살아있어 메모리 누적
- `compare_models()` 실행 시 모든 모델이 메모리에 남음
- 데이터 증강 시 원본 + 증강본 모두 메모리 보관

### 2. Organization별 중복 저장
- R*A, tonggibon 각각 별도로 모델 + 실험 + 데이터 저장
- 각 organization마다 메모리 사용량 2배 증가

## 적용된 해결책

### 1. 명시적 메모리 정리 함수 추가

#### modeling_service.py
```python
def cleanup_after_training(self) -> None:
    """학습 후 불필요한 메모리 정리"""
    import gc

    # 비교된 모델들 삭제
    if self.compared_models is not None:
        del self.compared_models
        self.compared_models = None

    # best_models 리스트 정리 (현재 모델만 유지)
    if self.model_results is not None and 'best_models' in self.model_results:
        recommended = self.model_results.get('recommended_model')
        self.model_results['best_models'] = [recommended] if recommended else []

    # 가비지 컬렉션 강제 실행
    gc.collect()
```

#### company_wide_modeling_service.py
```python
def cleanup_after_training(self, organization: str) -> None:
    """학습 후 불필요한 메모리 정리"""
    import gc

    # Organization별 메모리 정리
    if self.model_results.get(organization) is not None:
        if 'best_models' in self.model_results[organization]:
            recommended = self.model_results[organization].get('recommended_model')
            self.model_results[organization]['best_models'] = [recommended] if recommended else []

    gc.collect()
```

### 2. 개선된 clear_models() 함수

기존:
```python
def clear_models(self):
    self.current_model = None  # 참조만 제거
```

개선:
```python
def clear_models(self):
    import gc

    # 명시적 삭제
    if self.current_model is not None:
        del self.current_model
    if self.compared_models is not None:
        del self.compared_models
    # ... 모든 객체 삭제

    # 초기화
    self.current_model = None
    self.compared_models = None
    # ...

    # 가비지 컬렉션 강제 실행
    gc.collect()
```

### 3. 자동 정리 적용

모델 학습 후 자동으로 메모리 정리:

```python
def train_specific_model(self, model_name: str):
    # ... 모델 학습 ...

    # 모델 저장
    self._save_model(model_name)

    # 자동 메모리 정리 ✨
    self.cleanup_after_training()

    return result
```

## 사용 방법

### API 호출 후 메모리 정리

#### 방법 1: 자동 정리 (권장)
모델 학습 시 자동으로 정리됩니다:
```bash
# Setup → Compare → Train 시 자동 정리
curl -X POST http://localhost:8000/api/company-wide/modeling/train \
  -H "Content-Type: application/json" \
  -d '{"organization": "R*A", "model_name": "lr"}'
```

#### 방법 2: 수동 정리
작업 완료 후 명시적으로 정리:
```bash
# 특정 organization 정리
curl -X DELETE "http://localhost:8000/api/company-wide/modeling/clear?organization=R*A"

# 전체 정리
curl -X DELETE "http://localhost:8000/api/company-wide/modeling/clear"
```

### Python에서 직접 사용

```python
from app.services.company_wide_modeling_service import company_wide_modeling_service

# 모델 학습
company_wide_modeling_service.train_model('R*A', 'lr')

# 메모리 정리 (자동 실행되지만 필요시 수동 호출 가능)
company_wide_modeling_service.cleanup_after_training('R*A')

# 작업 완료 후 전체 정리
company_wide_modeling_service.clear_models('R*A')
```

## 테스트 스크립트 수정 권장사항

`test_company_wide_api.py`를 다음과 같이 수정하면 더 안전합니다:

```python
def test_with_cleanup(org):
    """각 테스트 후 메모리 정리"""
    try:
        # Setup
        test_setup(org)

        # Compare
        test_compare(org)

        # Train
        test_train(org, 'lr')

        # Prediction
        test_prediction(org)

    finally:
        # 테스트 완료 후 메모리 정리
        requests.delete(
            f"{BASE_URL}/modeling/clear",
            params={"organization": org}
        )
        print(f"✅ Memory cleaned up for {org}")
```

## 추가 권장사항

### 1. 개발 중 주기적 정리
```bash
# 개발 세션 중 정기적으로 실행
curl -X DELETE "http://localhost:8000/api/company-wide/modeling/clear"
```

### 2. 프로덕션 환경
- Worker timeout 설정: 30-60초
- Worker 자동 재시작 설정
- 메모리 모니터링 활성화

### 3. 데이터 증강 크기 조정
```python
# 큰 증강 크기 대신 적당한 크기 사용
SetupRequest(
    organization="R*A",
    use_augmentation=True,
    target_size=100  # 200 대신 100으로 줄임
)
```

## 메모리 사용량 모니터링

Python에서 메모리 사용량 확인:

```python
import psutil
import os

process = psutil.Process(os.getpid())
memory_mb = process.memory_info().rss / 1024 / 1024
print(f"Current memory usage: {memory_mb:.2f} MB")
```

## 예상 효과

- ✅ 메모리 누수 70-80% 감소
- ✅ compare_models 후 메모리 즉시 해제
- ✅ Organization별 독립적 메모리 관리
- ✅ 안정적인 장시간 실행
