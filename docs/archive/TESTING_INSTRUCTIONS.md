# 메모리 안전 모드 테스트 가이드

## 서버 상태
✅ Backend: http://localhost:8000
✅ Frontend: http://localhost:3001

## 테스트 시나리오

### 1. 데이터 업로드 테스트
1. 브라우저에서 http://localhost:3001 접속
2. "Data Upload" 메뉴로 이동
3. Excel/CSV 파일 업로드
4. 업로드된 데이터 확인

### 2. 모델 학습 테스트 (메모리 모니터링)
1. "Modeling" 메뉴로 이동
2. 타겟 컬럼 선택
3. "Train Model" 또는 "Compare Models" 클릭
4. **동시에 Activity Monitor 실행:**
   - `Cmd + Space` → "Activity Monitor" 검색
   - "Memory" 탭 선택
   - "Python" 프로세스 찾기
   - **"Memory" 열** 확인 (Virtual Memory 아님!)

### 3. 메모리 모니터링 포인트

#### ✅ 정상 (메모리 안전)
- 학습 시작 전: ~300 MB
- 학습 중: 500-800 MB
- 학습 후: 400-600 MB로 감소

#### ⚠️ 주의 (경계선)
- 학습 중: 800 MB - 1.5 GB
- 학습 후: 메모리가 줄어들지 않음

#### 🚨 위험 (시스템 리부팅 위험)
- 학습 중: 2 GB 이상
- 메모리가 계속 증가
- 시스템 전체가 느려짐

### 4. 적용된 메모리 보호 장치

#### 데이터 크기별 제한
- **작은 데이터 (< 30행)**
  - 모델 수: 1개 (선형회귀만)
  - Fold 수: 2
  - 예상 메모리: < 100 MB

- **중간 데이터 (30-50행)**
  - 모델 수: 최대 2개
  - Fold 수: 2
  - 예상 메모리: < 300 MB

- **큰 데이터 (50-100행)**
  - 모델 수: 최대 3개
  - Fold 수: 3
  - 예상 메모리: < 500 MB

- **매우 큰 데이터 (> 100행)**
  - 모델 수: 최대 3개
  - Fold 수: 5
  - 예상 메모리: < 800 MB

#### 자동 제한
- 최대 행 수: 10,000행 (초과 시 샘플링)
- 최대 Feature 수: 50개
- 병렬 처리: 비활성화 (`n_jobs=1`)
- 명시적 가비지 컬렉션 (`gc.collect()`)

## 실시간 메모리 모니터링 스크립트

터미널에서 실행:
```bash
watch -n 1 'ps aux | grep "python run.py" | grep -v grep | awk "{print \$6/1024\" MB\"}"'
```

또는 더 자세한 정보:
```bash
python3 check_memory.py
```

## 테스트 체크리스트

- [ ] 서버 정상 시작 (리부팅 없음)
- [ ] 데이터 업로드 성공
- [ ] 모델 학습 시작
- [ ] 메모리 사용량 < 1 GB 유지
- [ ] 학습 완료 후 메모리 감소
- [ ] 시스템 안정성 유지
- [ ] 브라우저 응답 정상

## 문제 발생 시

### 메모리가 계속 증가하면
1. 즉시 `Cmd + C`로 학습 중단
2. 서버 재시작: `make restart`
3. 데이터 크기 확인 (너무 큰지)

### 시스템이 느려지면
1. 즉시 학습 중단
2. Activity Monitor에서 Python 프로세스 강제 종료
3. `make restart`로 서버 재시작

### 수정이 필요하면
1. [modeling_service.py](backend/app/services/modeling_service.py) 수정
2. 메모리 제한을 더 엄격하게:
   ```python
   if data_size < 30:
       models_to_use = ['lr']  # 선형회귀만
       safe_fold = 2
   ```

## 성공 기준

✅ **테스트 통과 조건:**
1. 모델 학습이 완료됨
2. 메모리 사용량 < 1 GB
3. 시스템 리부팅 없음
4. 학습 후 메모리가 정상 수준으로 복귀
5. 여러 번 학습해도 안정적

## 참고 자료

- [MEMORY_FIX_SUMMARY.md](MEMORY_FIX_SUMMARY.md) - 메모리 수정 내역
- [check_memory.py](check_memory.py) - 메모리 분석 스크립트
- [modeling_service.py](backend/app/services/modeling_service.py) - 수정된 서비스 코드
