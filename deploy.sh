#!/bin/bash

echo "🚀 Wage Prediction App 배포 스크립트"
echo "=================================="
echo ""
echo "배포 플랫폼을 선택하세요:"
echo "1) Railway (추천 - 가장 쉬움)"
echo "2) Render"
echo "3) Docker Compose (로컬)"
echo "4) Fly.io"
echo "5) Vercel + 별도 Backend"
echo ""

read -p "선택 (1-5): " choice

case $choice in
    1)
        echo "🚂 Railway 배포 시작..."
        echo "1. https://railway.app 에서 계정 생성"
        echo "2. GitHub 저장소를 Railway에 연결"
        echo "3. railway.json 파일이 자동으로 설정을 적용합니다"
        echo ""
        echo "Railway CLI 설치 (선택사항):"
        echo "npm install -g @railway/cli"
        echo "railway login"
        echo "railway up"
        ;;
    
    2)
        echo "🔥 Render 배포 시작..."
        echo "1. https://render.com 에서 계정 생성"
        echo "2. 'New Blueprint Instance' 클릭"
        echo "3. GitHub 저장소 연결"
        echo "4. render.yaml이 자동으로 감지됩니다"
        ;;
    
    3)
        echo "🐳 Docker Compose 배포 시작..."
        docker --version > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "Docker 실행 중..."
            docker-compose up -d
            echo "✅ 배포 완료!"
            echo "Frontend: http://localhost:2000"
            echo "Backend: http://localhost:8000"
        else
            echo "❌ Docker가 설치되어 있지 않습니다."
            echo "https://www.docker.com/get-started 에서 설치하세요."
        fi
        ;;
    
    4)
        echo "🎯 Fly.io 배포 시작..."
        fly version > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            fly launch
            fly deploy
        else
            echo "Fly CLI 설치가 필요합니다:"
            echo "curl -L https://fly.io/install.sh | sh"
        fi
        ;;
    
    5)
        echo "⚡ Vercel + 별도 Backend 배포..."
        echo ""
        echo "Frontend (Vercel):"
        echo "cd frontend && vercel"
        echo ""
        echo "Backend 옵션:"
        echo "- Railway: railway up (backend 디렉토리에서)"
        echo "- Render: render.yaml 사용"
        echo "- Heroku: git push heroku main"
        ;;
    
    *)
        echo "잘못된 선택입니다."
        exit 1
        ;;
esac

echo ""
echo "📚 자세한 내용은 DEPLOYMENT.md 또는 EASY_DEPLOYMENT.md를 참조하세요."