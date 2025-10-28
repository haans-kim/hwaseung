#!/bin/bash

echo "🚀 스마트 포트 감지 시작 스크립트"
echo "================================"

# 사용 가능한 포트 찾기 함수
find_port() {
    local port=$1
    while lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; do
        echo "⚠️  포트 $port 사용 중... 다음 포트 확인" >&2
        port=$((port + 1))
    done
    echo "✅ 포트 $port 사용 가능!" >&2
    echo $port
}

# 백엔드 포트 찾기
echo "🔍 백엔드 포트 확인 중..."
BACKEND_PORT=$(find_port 8000)

# 프론트엔드 포트 찾기  
echo "🔍 프론트엔드 포트 확인 중..."
FRONTEND_PORT=$(find_port 3000)

# 백엔드 시작
echo "📦 백엔드 서버 시작 (포트: $BACKEND_PORT)..."
cd backend
if [ -d "venv" ]; then
    source venv/bin/activate
else
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
fi

# FastAPI 서버에 포트 전달
uvicorn app.main:app --host 0.0.0.0 --port $BACKEND_PORT --reload &
BACKEND_PID=$!

# 프론트엔드 시작
cd ../frontend
echo "📦 프론트엔드 서버 시작 (포트: $FRONTEND_PORT)..."

# Node.js 22.15 버전 사용
source ~/.nvm/nvm.sh
nvm use 22.15.0

# React 앱에 백엔드 URL 전달
export REACT_APP_API_URL=http://localhost:$BACKEND_PORT
export PORT=$FRONTEND_PORT

npm start &
FRONTEND_PID=$!

echo "================================"
echo "✨ 모든 서비스가 시작되었습니다!"
echo "🌐 프론트엔드: http://localhost:$FRONTEND_PORT"
echo "🔧 백엔드 API: http://localhost:$BACKEND_PORT"
echo "================================"

# Ctrl+C 시 종료
trap "echo '종료 중...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT

wait