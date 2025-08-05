# SambioWage Frontend

React + TypeScript + Tailwind CSS 기반 프론트엔드 애플리케이션

## 기술 스택

- **React 18** - UI 라이브러리
- **TypeScript** - 타입 안정성
- **Tailwind CSS** - 스타일링
- **React Router** - 라우팅
- **Lucide React** - 아이콘
- **Chart.js** - 차트 및 데이터 시각화

## 설치 및 실행

### 1. 의존성 설치
```bash
npm install
```

### 2. 개발 서버 실행
```bash
npm start
```

애플리케이션이 http://localhost:3000 에서 실행됩니다.

### 3. 빌드
```bash
npm run build
```

## 프로젝트 구조

```
src/
├── components/
│   ├── ui/                 # 재사용 가능한 UI 컴포넌트
│   │   ├── button.tsx
│   │   └── card.tsx
│   └── layout/             # 레이아웃 컴포넌트
│       ├── Layout.tsx
│       └── Sidebar.tsx
├── pages/                  # 페이지 컴포넌트
│   ├── DataUpload.tsx
│   ├── Modeling.tsx
│   ├── Analysis.tsx
│   ├── Dashboard.tsx
│   └── Effects.tsx
├── lib/
│   └── utils.ts           # 유틸리티 함수
├── App.tsx                # 메인 애플리케이션
└── index.tsx              # 진입점
```

## 주요 기능

### 1. 사이드바 네비게이션
- Data 업로드
- 모델링
- Analysis
- Dashboard
- 기대효과

### 2. 테마 전환
- Light/Dark 모드 지원
- 시스템 설정 저장

### 3. 반응형 디자인
- 모바일 및 태블릿 친화적
- shadcn/ui 디자인 시스템

## API 연동

백엔드 API와의 통신:
- Base URL: `http://localhost:8000`
- 파일 업로드: `POST /api/data/upload`
- 모델링: `POST /api/modeling/*`
- 분석: `GET /api/analysis/*`
- 대시보드: `POST /api/dashboard/*`

## 환경 설정

`.env` 파일에서 환경변수 설정:

```env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_VERSION=1.0.0
```