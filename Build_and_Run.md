# Build and Run Guide

## 📋 목차
- [환경별 설정](#환경별-설정)
- [Quick Start](#quick-start)
- [개발 환경](#개발-환경)
- [프로덕션 환경](#프로덕션-환경)
- [Makefile 명령어](#makefile-명령어)
- [도메인 설정](#도메인-설정)
- [문제 해결](#문제-해결)

## 환경별 설정

### API URL 구성
| 환경 | Frontend URL | Backend URL | 설정 파일 |
|------|-------------|-------------|-----------|
| 개발 | http://localhost:3001 | http://localhost:8000 | `.env.development` |
| 프로덕션 | http://dashboard.insightgroup.biz:3000 | http://dashboard.insightgroup.biz:8000 | `.env.production` |

### 환경 파일 구조
```
frontend/
├── .env                    # 기본 설정 (개발용)
├── .env.development        # 개발 환경 설정
└── .env.production         # 프로덕션 환경 설정
```

## Quick Start

### 가장 빠른 시작 방법
```bash
# 의존성 설치 및 개발 서버 시작
make install
make start
```

### 프로덕션 모드로 시작 (외부 접속 가능)
```bash
make start-production
```

## 개발 환경

### 1. 백엔드 실행
```bash
cd backend
source venv/bin/activate  # 가상환경 활성화
pip install -r requirements.txt  # 의존성 설치 (최초 1회)
python run.py  # localhost:8000에서 실행
```

### 2. 프론트엔드 실행
```bash
cd frontend
npm install  # 의존성 설치 (최초 1회)
npm start    # localhost:3001에서 실행 (개발 모드)
```

### 3. Makefile 사용 (권장)
```bash
# 스마트 시작 (포트 자동 감지)
make start

# 고정 포트로 시작
make start-fixed

# 강제 시작 (기존 프로세스 정리 후)
make start-force
```

## 프로덕션 환경

### 1. 프론트엔드 빌드
```bash
cd frontend
npm run build  # build/ 폴더에 최적화된 파일 생성
```

빌드 시 자동으로 `.env.production` 파일의 설정이 적용됩니다:
- `REACT_APP_API_URL=http://dashboard.insightgroup.biz:8000`

### 2. 빌드된 파일 실행
```bash
# serve 패키지로 실행
npx serve -s build -l 3000

# 또는 다른 포트 지정
npx serve -s build -l 5000
```

### 3. 백엔드 외부 접속 설정
```bash
cd backend
# 모든 네트워크 인터페이스에서 접속 가능
python run.py --host 0.0.0.0 --port 8000
```

### 4. Makefile로 프로덕션 실행 (권장)
```bash
# 프로덕션 모드로 전체 시스템 시작
make start-production

# 개별 작업
make build-frontend   # 프론트엔드 빌드만
make serve-frontend   # 빌드된 프론트엔드 실행
```

## Makefile 명령어

### 시작/중지
| 명령어 | 설명 |
|--------|------|
| `make start` | 스마트 포트 감지로 시작 |
| `make start-fixed` | 고정 포트로 시작 (로컬 개발) |
| `make start-production` | 프로덕션 모드 (외부 접속 가능) |
| `make start-force` | 강제로 포트 정리 후 시작 |
| `make restart` | 서비스 재시작 |
| `make stop` | 모든 서비스 중지 |

### 빌드/배포
| 명령어 | 설명 |
|--------|------|
| `make build-frontend` | 프론트엔드 빌드 |
| `make serve-frontend` | 빌드된 프론트엔드 실행 |
| `make install` | 의존성 설치 |

### 유틸리티
| 명령어 | 설명 |
|--------|------|
| `make check-ports` | 포트 사용 현황 확인 |
| `make logs` | 백엔드 로그 확인 |
| `make clean` | 개발 환경 초기화 |
| `make kill-all` | 모든 관련 프로세스 강제 종료 |
| `make help` | 도움말 표시 |

## 도메인 설정

### dashboard.insightgroup.biz 설정

1. **DNS 설정**
   - 도메인이 맥미니의 IP를 가리키도록 설정
   - A 레코드: `dashboard.insightgroup.biz` → 맥미니 IP

2. **방화벽 설정**
   - 포트 3000 (프론트엔드)
   - 포트 8000 (백엔드)

3. **네트워크 확인**
   ```bash
   # 현재 IP 확인 (macOS)
   ifconfig | grep "inet " | grep -v 127.0.0.1

   # 포트 열림 확인
   lsof -i:3000
   lsof -i:8000
   ```

## 문제 해결

### 포트가 이미 사용 중일 때
```bash
# 포트 확인
make check-ports

# 강제 정리
make kill-all

# 재시작
make restart
```

### 백엔드 연결 실패
1. 백엔드가 실행 중인지 확인
   ```bash
   ps aux | grep "python.*run.py"
   ```

2. 백엔드가 올바른 호스트로 실행 중인지 확인
   ```bash
   # 외부 접속이 필요한 경우
   cd backend
   python run.py --host 0.0.0.0 --port 8000
   ```

3. API URL 설정 확인
   ```bash
   # 개발 환경
   cat frontend/.env.development
   # REACT_APP_API_URL=http://localhost:8000

   # 프로덕션 환경
   cat frontend/.env.production
   # REACT_APP_API_URL=http://dashboard.insightgroup.biz:8000
   ```

### 프론트엔드 빌드 에러
```bash
# node_modules 재설치
cd frontend
rm -rf node_modules
npm install

# 빌드 재시도
npm run build
```

### 외부에서 접속 안 될 때
1. 맥미니 방화벽 설정 확인
2. 공유기 포트 포워딩 설정
3. 백엔드가 `0.0.0.0`으로 실행 중인지 확인
4. DNS가 올바른 IP를 가리키는지 확인

## 환경 변수 우선순위

React 앱에서 환경 변수 로딩 순서:
1. `process.env.REACT_APP_API_URL` (환경별 .env 파일)
2. 하드코딩된 기본값 (`http://localhost:8000`)

빌드 시:
- `npm start` → `.env.development` 사용
- `npm run build` → `.env.production` 사용

## 배포 체크리스트

프로덕션 배포 전 확인사항:

- [ ] `.env.production`에 올바른 API URL 설정
- [ ] 백엔드가 외부 접속 가능하도록 설정 (`--host 0.0.0.0`)
- [ ] DNS 설정 완료
- [ ] 방화벽 포트 오픈 (3000, 8000)
- [ ] 프론트엔드 빌드 완료 (`npm run build`)
- [ ] 로그 모니터링 설정

## 서버 운영 팁

### 백그라운드 실행
```bash
# nohup 사용
nohup python run.py --host 0.0.0.0 --port 8000 > backend.log 2>&1 &
nohup npx serve -s build -l 3000 > frontend.log 2>&1 &

# tmux/screen 사용 (권장)
tmux new -s backend
python run.py --host 0.0.0.0 --port 8000
# Ctrl+B, D로 분리

tmux new -s frontend
npx serve -s build -l 3000
# Ctrl+B, D로 분리
```

### 로그 확인
```bash
# 백엔드 로그
tail -f backend/logs.log

# 실시간 모니터링
make logs
```

### 자동 재시작 (PM2 사용)
```bash
# PM2 설치
npm install -g pm2

# 백엔드 실행
pm2 start backend/run.py --name backend --interpreter python3 -- --host 0.0.0.0 --port 8000

# 프론트엔드 실행
pm2 serve frontend/build 3000 --name frontend

# 상태 확인
pm2 status

# 재시작
pm2 restart all
```