# SambioWage 배포 가이드

## 프론트엔드 배포 (Vercel)

### 1. Vercel 계정 생성 및 프로젝트 연결
1. [Vercel](https://vercel.com) 계정 생성
2. GitHub 리포지토리 연결
3. Import Project 선택

### 2. 빌드 설정
- **Framework Preset**: Create React App
- **Root Directory**: `frontend`
- **Build Command**: `npm run build`
- **Output Directory**: `build`
- **Install Command**: `npm install`

### 3. 환경 변수 설정
Vercel 대시보드에서 다음 환경 변수 추가:
```
REACT_APP_API_URL=https://your-backend-url.railway.app
```

### 4. 배포
- 자동 배포 활성화 시 main 브랜치 push 시 자동 배포
- 수동 배포: Vercel 대시보드에서 "Redeploy" 클릭

## 백엔드 배포 (Railway)

### 1. Railway 계정 생성 및 프로젝트 생성
1. [Railway](https://railway.app) 계정 생성
2. New Project → Deploy from GitHub repo 선택
3. 리포지토리 선택

### 2. 배포 설정
- **Root Directory**: `backend`
- **Build Command**: 자동 감지 (railway.json 참조)
- **Start Command**: `python run.py`

### 3. 환경 변수 설정
Railway 대시보드에서 다음 환경 변수 추가:
```
ENVIRONMENT=production
PORT=8000
CORS_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000
```

### 4. 영구 스토리지 설정
- Railway 대시보드에서 Volume 추가
- Mount path: `/app/uploads` (업로드 파일용)
- Mount path: `/app/data` (데이터 파일용)
- Mount path: `/app/models` (모델 파일용)

## 백엔드 배포 (Render) - 대안

### 1. Render 계정 생성 및 서비스 생성
1. [Render](https://render.com) 계정 생성
2. New → Web Service 선택
3. GitHub 리포지토리 연결

### 2. 배포 설정
- **Name**: sambiowage-backend
- **Root Directory**: `backend`
- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python run.py`

### 3. 환경 변수 설정
```
ENVIRONMENT=production
PORT=10000
CORS_ORIGINS=https://your-frontend.vercel.app
```

## 배포 후 확인사항

### 프론트엔드
1. 배포된 URL 접속 확인
2. API 연결 테스트 (개발자 도구 Network 탭 확인)
3. 모든 페이지 라우팅 동작 확인

### 백엔드
1. Health check 엔드포인트 확인: `https://your-backend-url/health`
2. CORS 설정 확인 (프론트엔드에서 API 호출 시)
3. 파일 업로드 기능 테스트
4. 모델 학습 및 예측 기능 테스트

## 문제 해결

### CORS 오류
- 백엔드 환경 변수 `CORS_ORIGINS`에 프론트엔드 URL 추가
- 프로토콜(https://)과 도메인 정확히 입력

### 파일 업로드 실패
- Railway/Render에서 영구 스토리지 설정 확인
- 파일 권한 확인

### API 연결 실패
- 프론트엔드 환경 변수 `REACT_APP_API_URL` 확인
- 백엔드 서비스 상태 확인

## 배포 명령어

### 로컬 테스트
```bash
# 프론트엔드
cd frontend
npm run build
serve -s build

# 백엔드
cd backend
ENVIRONMENT=production python run.py
```

### Git 푸시로 자동 배포
```bash
git add .
git commit -m "Deploy to production"
git push origin main
```