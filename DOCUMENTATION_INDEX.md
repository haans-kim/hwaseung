# 프로젝트 문서 인덱스

## 📚 전체 문서 목록

### 🎯 핵심 구현 문서

#### 1. **IMPLEMENTATION_COMPLETE.md** ⭐ (NEW)
**최신 구현 완료 보고서**
- 메모리 안전 모드 적용 완료
- 전사 적정인력 산정 시스템 (R*A/tonggibon)
- API 8개 엔드포인트
- 프론트엔드 Dashboard 2개
- 테스트 결과 및 사용 방법

#### 2. **company_wide_modeling_plan.md** 📋
**전사 적정인력 산정 시스템 구현 계획**
- 위치: `claudedocs/company_wide_modeling_plan.md`
- R*A와 tonggibon 적정인력 산정 시스템 설계
- Phase별 구현 단계
- 데이터 구조 및 API 명세
- 검증 항목

### 🔧 메모리 최적화 문서

#### 3. **MEMORY_FIX_SUMMARY.md** 🚨
**메모리 폭발 문제 해결 완료**
- 시스템 리부팅 문제 원인 분석
- PyCaret compare_models() 메모리 폭발 해결
- 6가지 메모리 보호 장치
- 메모리 사용량 160배 감소 (8.6GB → 54MB)

#### 4. **MEMORY_CLEANUP_GUIDE.md**
**메모리 정리 가이드**
- 메모리 관리 best practices
- GC 사용법
- 메모리 누수 방지

### 🧪 테스트 문서

#### 5. **TEST_RESULTS.md**
**테스트 결과 보고서**
- 서버 상태 확인
- 메모리 사용량 측정
- API 테스트 결과

#### 6. **TESTING_INSTRUCTIONS.md**
**테스트 가이드**
- 메모리 모니터링 방법
- Activity Monitor 확인법
- 데이터 업로드 테스트
- 모델 학습 테스트

### 📊 데이터 관련 문서

#### 7. **데이터_업로드_재구성_분석.md** 📈
**데이터 업로드 시스템 설계**
- 위치: `claudedocs/데이터_업로드_재구성_분석.md`
- Template 파일 5개 구조 분석
- 페이지별 데이터 매핑
- 데이터베이스 스키마 제안

#### 8. **데이터_업로드_재구성_분석_v2.md**
**데이터 업로드 시스템 v2**
- 위치: `claudedocs/데이터_업로드_재구성_분석_v2.md`
- 개선된 업로드 시스템
- 상세 구현 계획

#### 9. **근무시간_산출_로직.md**
**근무시간 계산 로직**
- 위치: `data/근무시간_산출_로직.md`
- FTE 계산 방법
- 근무시간 산출 규칙

### 🚀 배포 문서

#### 10. **DEPLOYMENT.md** 🌐
**배포 가이드**
- Vercel (Frontend) 배포
- Railway/Render (Backend) 배포
- 환경 변수 설정

#### 11. **EASY_DEPLOYMENT.md**
**간편 배포 가이드**
- 빠른 배포 방법
- 원클릭 배포 옵션

#### 12. **README_DEPLOYMENT.md**
**배포 상세 문서**
- 단계별 배포 절차
- 트러블슈팅

#### 13. **README_MAC_MINI_DEPLOYMENT.md**
**Mac Mini 배포 가이드**
- Mac Mini 서버 설정
- 로컬 서버 운영

#### 14. **mac-mini-setup.md**
**Mac Mini 초기 설정**
- 서버 환경 구축
- 의존성 설치

### 📖 기본 문서

#### 15. **README.md** 📘
**프로젝트 메인 문서**
- 프로젝트 개요
- 기술 스택
- 주요 기능

#### 16. **CLAUDE.md** 🤖
**Claude Code 작업 규칙**
- 코딩 규칙 (DEFAULT 값 사용 금지 등)
- 프로젝트 가이드라인

#### 17. **Build_and_Run.md**
**빌드 및 실행 가이드**
- 로컬 개발 환경 설정
- 빌드 명령어

#### 18. **IMPORTANT_FILES.md**
**중요 파일 목록**
- 핵심 파일들의 위치와 역할

### 🎨 프론트엔드 문서

#### 19. **frontend/README.md**
**Frontend 문서**
- React 프로젝트 구조
- 컴포넌트 설명

#### 20. **frontend/README_FRONTEND.md**
**Frontend 상세 문서**
- UI 컴포넌트 가이드
- 페이지별 설명

### 🔌 백엔드 문서

#### 21. **backend/README.md**
**Backend 문서**
- FastAPI 프로젝트 구조
- API 엔드포인트 설명

---

## 📁 문서 디렉토리 구조

```
/Users/hanskim/Projects/Hwaseung/
├── IMPLEMENTATION_COMPLETE.md ⭐ (최신)
├── MEMORY_FIX_SUMMARY.md
├── TEST_RESULTS.md
├── TESTING_INSTRUCTIONS.md
├── MEMORY_CLEANUP_GUIDE.md
├── README.md
├── CLAUDE.md
├── Build_and_Run.md
├── DEPLOYMENT.md
├── EASY_DEPLOYMENT.md
├── README_DEPLOYMENT.md
├── README_MAC_MINI_DEPLOYMENT.md
├── mac-mini-setup.md
├── IMPORTANT_FILES.md
│
├── claudedocs/
│   ├── company_wide_modeling_plan.md 📋 (구현 계획)
│   ├── 데이터_업로드_재구성_분석.md 📈
│   └── 데이터_업로드_재구성_분석_v2.md
│
├── data/
│   └── 근무시간_산출_로직.md
│
├── frontend/
│   ├── README.md
│   └── README_FRONTEND.md
│
└── backend/
    └── README.md
```

---

## 🎯 빠른 참조

### 새로 작업 시작할 때
1. **IMPLEMENTATION_COMPLETE.md** - 현재 구현 상태 확인
2. **CLAUDE.md** - 코딩 규칙 확인
3. **company_wide_modeling_plan.md** - 구현 계획 확인

### 문제 해결할 때
1. **MEMORY_FIX_SUMMARY.md** - 메모리 문제
2. **TESTING_INSTRUCTIONS.md** - 테스트 방법
3. **TEST_RESULTS.md** - 테스트 결과

### 배포할 때
1. **DEPLOYMENT.md** - 기본 배포 가이드
2. **EASY_DEPLOYMENT.md** - 빠른 배포
3. **README_MAC_MINI_DEPLOYMENT.md** - Mac Mini 배포

### 데이터 작업할 때
1. **데이터_업로드_재구성_분석.md** - 업로드 시스템
2. **근무시간_산출_로직.md** - FTE 계산
3. **company_wide_modeling_plan.md** - 데이터 구조

---

## 📝 문서 작성 규칙

### 문서 위치 규칙
- **프로젝트 루트**: 전체 프로젝트 관련 문서
- **claudedocs/**: 구현 계획, 분석 문서
- **frontend/**: Frontend 관련 문서
- **backend/**: Backend 관련 문서
- **data/**: 데이터 처리 관련 문서

### 문서 명명 규칙
- **대문자 시작**: README.md, DEPLOYMENT.md
- **한글 문서**: `데이터_업로드_재구성_분석.md`
- **버전 표시**: `_v2.md`, `_v3.md`
- **날짜 표시**: 문서 내 "작성일" 필드

### 문서 업데이트
- 새로운 기능 구현 시: IMPLEMENTATION_COMPLETE.md 업데이트
- 문제 해결 시: 해당 문서에 해결 방법 추가
- 계획 변경 시: 계획 문서 버전업

---

## 🔄 최근 업데이트

### 2025-10-10
- ✅ **IMPLEMENTATION_COMPLETE.md** 생성
- ✅ **MEMORY_FIX_SUMMARY.md** 완성
- ✅ **TEST_RESULTS.md** 생성
- ✅ **TESTING_INSTRUCTIONS.md** 생성
- ✅ 메모리 안전 모드 적용 완료
- ✅ R*A/tonggibon 시스템 구현 완료

---

## 📌 중요 링크

- **최신 구현 현황**: [IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)
- **구현 계획**: [company_wide_modeling_plan.md](claudedocs/company_wide_modeling_plan.md)
- **메모리 문제 해결**: [MEMORY_FIX_SUMMARY.md](MEMORY_FIX_SUMMARY.md)
- **테스트 가이드**: [TESTING_INSTRUCTIONS.md](TESTING_INSTRUCTIONS.md)
- **코딩 규칙**: [CLAUDE.md](CLAUDE.md)

---

**마지막 업데이트**: 2025-10-10
**문서 개수**: 21개 (프로젝트 관련 문서만)
