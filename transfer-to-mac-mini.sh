#!/bin/bash

# 맥미니로 모든 필요한 파일 전송 스크립트

echo "📦 맥미니로 전체 프로젝트 전송"
echo "=============================="

# 설정 (수정 필요)
MAC_MINI_HOST="mac-mini.local"  # 맥미니 호스트명 또는 IP
MAC_MINI_USER="hanskim"          # 맥미니 사용자명
REMOTE_DIR="/Users/$MAC_MINI_USER/sambiowage"

echo "🎯 대상: $MAC_MINI_USER@$MAC_MINI_HOST:$REMOTE_DIR"

# 1. Git으로 관리되는 파일들 업데이트
echo ""
echo "1️⃣ Git 파일 업데이트..."
ssh $MAC_MINI_USER@$MAC_MINI_HOST << EOF
  cd $REMOTE_DIR 2>/dev/null || git clone https://github.com/haans-kim/2025_Wage_Prediction.git $REMOTE_DIR
  cd $REMOTE_DIR
  git pull origin main
EOF

# 2. Git에서 제외된 중요 파일들 전송
echo ""
echo "2️⃣ 데이터 및 모델 파일 전송..."

# backend/data 폴더 (pkl 파일들)
echo "   - 데이터 파일..."
rsync -avz --progress \
  backend/data/*.pkl \
  $MAC_MINI_USER@$MAC_MINI_HOST:$REMOTE_DIR/backend/data/

# backend/models 폴더 (학습된 모델)
echo "   - 모델 파일..."
rsync -avz --progress \
  backend/models/*.pkl \
  $MAC_MINI_USER@$MAC_MINI_HOST:$REMOTE_DIR/backend/models/

# backend/uploads 폴더 (업로드된 파일들)
echo "   - 업로드 파일..."
rsync -avz --progress \
  backend/uploads/ \
  $MAC_MINI_USER@$MAC_MINI_HOST:$REMOTE_DIR/backend/uploads/

# saved_models 폴더 (있는 경우)
if [ -d "backend/saved_models" ]; then
  echo "   - 저장된 모델..."
  rsync -avz --progress \
    backend/saved_models/ \
    $MAC_MINI_USER@$MAC_MINI_HOST:$REMOTE_DIR/backend/saved_models/
fi

# 3. 환경 설정 파일 (선택적)
echo ""
echo "3️⃣ 환경 설정..."

# .env 파일들 (있는 경우)
if [ -f "backend/.env" ]; then
  scp backend/.env $MAC_MINI_USER@$MAC_MINI_HOST:$REMOTE_DIR/backend/
fi

if [ -f "frontend/.env" ]; then
  scp frontend/.env $MAC_MINI_USER@$MAC_MINI_HOST:$REMOTE_DIR/frontend/
fi

# 4. 디렉토리 권한 설정
echo ""
echo "4️⃣ 디렉토리 생성 및 권한 설정..."
ssh $MAC_MINI_USER@$MAC_MINI_HOST << EOF
  cd $REMOTE_DIR
  mkdir -p backend/data backend/models backend/uploads backend/saved_models
  mkdir -p frontend/build
  mkdir -p logs
EOF

echo ""
echo "✅ 전송 완료!"
echo ""
echo "이제 맥미니에서 다음 명령을 실행하세요:"
echo "  cd $REMOTE_DIR"
echo "  ./quick-start-mac-mini.sh"
echo ""
echo "또는 직접 Docker 실행:"
echo "  docker-compose -f docker-compose.production.yml up -d --build"