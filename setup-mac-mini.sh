#!/bin/bash

# 맥미니 초기 설정 스크립트
# 맥미니에서 한 번만 실행

echo "🔧 맥미니 서버 초기 설정 시작..."

# 1. 프로젝트 디렉토리 생성
mkdir -p ~/hwaseung
cd ~/hwaseung

# 2. Docker 설치 확인
if ! command -v docker &> /dev/null; then
    echo "Docker가 설치되어 있지 않습니다."
    echo "https://www.docker.com/products/docker-desktop 에서 Docker Desktop을 설치해주세요."
    exit 1
fi

# 3. 로그 디렉토리 생성
mkdir -p logs

# 4. 데이터 영구 저장 디렉토리 생성
mkdir -p backend/uploads
mkdir -p backend/data
mkdir -p backend/models

# 5. 자동 시작 설정 (LaunchAgent)
cat > ~/Library/LaunchAgents/com.hwaseung.docker.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.hwaseung.docker</string>

    <key>ProgramArguments</key>
    <array>
        <string>/usr/local/bin/docker-compose</string>
        <string>-f</string>
        <string>/Users/hanskim/hwaseung/docker-compose.production.yml</string>
        <string>up</string>
        <string>-d</string>
    </array>

    <key>WorkingDirectory</key>
    <string>/Users/hanskim/hwaseung</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>/Users/hanskim/hwaseung/logs/hwaseung.log</string>

    <key>StandardErrorPath</key>
    <string>/Users/hanskim/hwaseung/logs/hwaseung.error.log</string>
</dict>
</plist>
EOF

# LaunchAgent 등록
launchctl load ~/Library/LaunchAgents/com.hwaseung.docker.plist

echo "✅ 맥미니 서버 설정 완료!"
echo ""
echo "다음 단계:"
echo "1. 맥북에서 deploy-to-mac-mini.sh 스크립트의 MAC_MINI_HOST를 수정하세요"
echo "2. 맥북에서 ./deploy-to-mac-mini.sh 실행하여 배포하세요"