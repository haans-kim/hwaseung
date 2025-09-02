#!/bin/bash

# 맥미니 서버 설정
MAC_MINI_HOST="your-mac-mini.local"  # 맥미니 호스트명 또는 IP
MAC_MINI_USER="hanskim"  # 맥미니 사용자명
REMOTE_DIR="/Users/$MAC_MINI_USER/sambiowage"

echo "🚀 맥미니 서버에 SambioWage 배포 시작..."

# 1. 프론트엔드 빌드
echo "📦 프론트엔드 빌드 중..."
cd frontend
npm run build
cd ..

# 2. 파일 동기화 (rsync 사용)
echo "📤 파일을 맥미니로 전송 중..."
rsync -avz --exclude 'node_modules' \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude 'logs' \
    ./ $MAC_MINI_USER@$MAC_MINI_HOST:$REMOTE_DIR/

# 3. 맥미니에서 Docker 컨테이너 실행
echo "🐳 맥미니에서 Docker 컨테이너 시작..."
ssh $MAC_MINI_USER@$MAC_MINI_HOST << 'ENDSSH'
cd /Users/hanskim/sambiowage

# 기존 컨테이너 중지 및 제거
docker-compose -f docker-compose.production.yml down

# 새 이미지 빌드 및 컨테이너 시작
docker-compose -f docker-compose.production.yml build --no-cache
docker-compose -f docker-compose.production.yml up -d

# 상태 확인
docker-compose -f docker-compose.production.yml ps

echo "✅ 배포 완료!"
echo "🌐 http://your-mac-mini.local 에서 접속 가능합니다."
ENDSSH