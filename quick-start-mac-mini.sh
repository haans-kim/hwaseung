#!/bin/bash

# 맥미니에서 실행할 빠른 시작 스크립트

echo "🚀 SambioWage 서버 시작 스크립트"
echo "================================"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Git 최신 버전 가져오기
echo -e "${YELLOW}📥 최신 코드 가져오는 중...${NC}"
git pull origin main

# 2. Docker 실행 상태 확인
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker가 실행되고 있지 않습니다.${NC}"
    echo "Docker Desktop을 시작해주세요."
    open -a Docker
    echo "Docker가 시작될 때까지 대기 중..."
    sleep 10
fi

# 3. 기존 컨테이너 중지
echo -e "${YELLOW}🛑 기존 컨테이너 중지 중...${NC}"
docker-compose -f docker-compose.production.yml down

# 4. 새 컨테이너 빌드 및 시작
echo -e "${YELLOW}🔨 컨테이너 빌드 및 시작 중...${NC}"
docker-compose -f docker-compose.production.yml up -d --build

# 5. 상태 확인
echo -e "${GREEN}✅ 서비스 상태:${NC}"
docker-compose -f docker-compose.production.yml ps

# 6. IP 정보 출력
echo ""
echo -e "${GREEN}🌐 접속 정보:${NC}"
echo "================================"

# 내부 IP
INTERNAL_IP=$(ipconfig getifaddr en0)
if [ -z "$INTERNAL_IP" ]; then
    INTERNAL_IP=$(ipconfig getifaddr en1)
fi
echo -e "내부 네트워크: ${GREEN}http://$INTERNAL_IP${NC}"

# 외부 IP
EXTERNAL_IP=$(curl -s ifconfig.me)
echo -e "외부 네트워크: ${GREEN}http://$EXTERNAL_IP${NC}"

echo ""
echo -e "${YELLOW}📌 참고사항:${NC}"
echo "- 외부 접속을 위해서는 공유기에서 포트포워딩 설정이 필요합니다"
echo "- 포트 80 → 맥미니 IP ($INTERNAL_IP)"
echo ""
echo -e "로그 확인: ${YELLOW}docker-compose -f docker-compose.production.yml logs -f${NC}"
echo -e "서비스 중지: ${YELLOW}docker-compose -f docker-compose.production.yml down${NC}"