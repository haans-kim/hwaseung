# 쉬운 배포 방법들

## 1. 🚀 Railway (가장 쉬운 방법)

전체 프로젝트를 한 번에 배포할 수 있습니다.

### 배포 단계:
1. [Railway](https://railway.app) 가입
2. GitHub 저장소 연결
3. "Deploy from GitHub repo" 클릭
4. Railway가 자동으로 서비스 감지:
   - Frontend 서비스
   - Backend 서비스
5. 환경 변수만 설정하면 완료!

railway.json 파일이 이미 생성되어 있어 자동으로 설정됩니다!

## 2. 🐳 Docker Compose (로컬처럼 배포)

Docker를 사용하면 로컬 환경과 동일하게 배포할 수 있습니다.

### 로컬에서 실행:
```bash
docker-compose up -d
```

### 클라우드 배포 옵션:

#### a) DigitalOcean App Platform
```bash
doctl apps create --spec app.yaml
```

#### b) AWS ECS/Fargate
```bash
docker-compose up
ecs-cli compose up
```

#### c) Google Cloud Run
```bash
gcloud run deploy --source .
```

## 3. 🔥 Render (Blueprint로 한 번에 배포)

Render는 모노레포를 자동으로 인식합니다.

### 배포 방법:
1. [Render](https://render.com) 가입
2. "New Blueprint Instance" 클릭
3. GitHub 저장소 연결
4. render.yaml 파일이 자동으로 감지됨
5. "Apply" 클릭하면 완료!

## 4. 🎯 Fly.io (글로벌 배포)

전 세계에 자동으로 배포됩니다.

### 배포 명령어:
```bash
# Fly CLI 설치
curl -L https://fly.io/install.sh | sh

# 배포
fly launch
fly deploy
```

## 5. 🚄 한 줄 배포 스크립트

가장 쉬운 방법을 위한 자동 배포 스크립트입니다.