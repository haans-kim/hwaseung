# 맥미니 Docker 배포 가이드

## 사전 준비사항

### 맥미니 (서버)
1. Docker Desktop 설치
2. SSH 접속 활성화 (시스템 환경설정 > 공유 > 원격 로그인)
3. 고정 IP 설정 또는 호스트명 확인

### 맥북 (개발 환경)
1. SSH 키 설정 (비밀번호 없이 접속)
   ```bash
   ssh-copy-id hanskim@your-mac-mini.local
   ```

## 초기 설정 (맥미니에서 1회 실행)

1. setup-mac-mini.sh 파일을 맥미니로 복사
   ```bash
   scp setup-mac-mini.sh hanskim@your-mac-mini.local:~/
   ```

2. 맥미니에서 설정 스크립트 실행
   ```bash
   ssh hanskim@your-mac-mini.local
   chmod +x setup-mac-mini.sh
   ./setup-mac-mini.sh
   ```

## 배포 방법 (맥북에서 실행)

1. deploy-to-mac-mini.sh 파일 수정
   ```bash
   # MAC_MINI_HOST를 실제 맥미니 주소로 변경
   MAC_MINI_HOST="192.168.1.100"  # 또는 "mac-mini.local"
   ```

2. 배포 스크립트 실행
   ```bash
   chmod +x deploy-to-mac-mini.sh
   ./deploy-to-mac-mini.sh
   ```

## 서비스 관리

### 서비스 시작
```bash
ssh hanskim@mac-mini.local "cd ~/sambiowage && docker-compose -f docker-compose.production.yml up -d"
```

### 서비스 중지
```bash
ssh hanskim@mac-mini.local "cd ~/sambiowage && docker-compose -f docker-compose.production.yml down"
```

### 로그 확인
```bash
ssh hanskim@mac-mini.local "cd ~/sambiowage && docker-compose -f docker-compose.production.yml logs -f"
```

### 컨테이너 상태 확인
```bash
ssh hanskim@mac-mini.local "cd ~/sambiowage && docker-compose -f docker-compose.production.yml ps"
```

## 접속 정보

- **웹 애플리케이션**: http://mac-mini.local 또는 http://[맥미니-IP]
- **API 엔드포인트**: http://mac-mini.local/api
- **헬스체크**: http://mac-mini.local/health

## 자동 시작 설정

맥미니가 재시작되면 자동으로 서비스가 시작됩니다.

### 자동 시작 비활성화
```bash
launchctl unload ~/Library/LaunchAgents/com.sambiowage.docker.plist
```

### 자동 시작 재활성화
```bash
launchctl load ~/Library/LaunchAgents/com.sambiowage.docker.plist
```

## 문제 해결

### Docker 컨테이너가 시작되지 않을 때
```bash
# 로그 확인
cat ~/sambiowage/logs/sambiowage.error.log

# Docker 재시작
killall Docker && open -a Docker

# 컨테이너 강제 정리
docker system prune -af
```

### 포트 충돌
```bash
# 80 포트 사용 중인 프로세스 확인
sudo lsof -i :80

# nginx가 이미 실행 중이면 중지
sudo nginx -s stop
```

### 디스크 공간 부족
```bash
# Docker 이미지 정리
docker image prune -af

# 오래된 컨테이너 정리  
docker container prune -f
```

## 보안 설정

### 방화벽 설정 (옵션)
시스템 환경설정 > 보안 및 개인 정보 보호 > 방화벽에서 80 포트 허용

### HTTPS 설정 (옵션)
1. Let's Encrypt 인증서 발급
2. nginx/nginx.conf에 SSL 설정 추가
3. docker-compose.production.yml에서 443 포트 노출

## 백업

### 데이터 백업
```bash
# 맥북에서 실행
rsync -avz hanskim@mac-mini.local:~/sambiowage/backend/data/ ./backup/data/
rsync -avz hanskim@mac-mini.local:~/sambiowage/backend/models/ ./backup/models/
```

### 자동 백업 설정
Time Machine 또는 cron job을 사용하여 정기 백업 설정 가능