# 맥미니 서버 설정 가이드

## 1. 맥미니에서 프로젝트 실행

### 1.1 프로젝트 클론 및 업데이트
```bash
# 처음 설치하는 경우
cd ~
git clone https://github.com/haans-kim/2025_Wage_Prediction.git sambiowage
cd sambiowage

# 이미 클론되어 있는 경우 최신 버전으로 업데이트
cd ~/sambiowage
git pull origin main
```

### 1.2 Docker Compose로 실행
```bash
# 프로덕션 모드로 실행 (백그라운드)
docker-compose -f docker-compose.production.yml up -d --build

# 로그 확인
docker-compose -f docker-compose.production.yml logs -f

# 상태 확인
docker-compose -f docker-compose.production.yml ps
```

### 1.3 서비스 중지/재시작
```bash
# 중지
docker-compose -f docker-compose.production.yml down

# 재시작
docker-compose -f docker-compose.production.yml restart
```

## 2. 포트포워딩 설정

### 2.1 공유기 설정 (일반적인 경우)
1. 공유기 관리 페이지 접속 (보통 192.168.1.1 또는 192.168.0.1)
2. 포트포워딩 메뉴 찾기
3. 다음 규칙 추가:
   - 외부 포트: 80
   - 내부 IP: 맥미니의 내부 IP (예: 192.168.1.100)
   - 내부 포트: 80
   - 프로토콜: TCP

### 2.2 맥미니 고정 IP 설정
```bash
# 맥미니의 현재 IP 확인
ifconfig | grep "inet " | grep -v 127.0.0.1

# 시스템 환경설정 > 네트워크 > 고급 > TCP/IP
# IPv4 구성: 수동으로 설정
# IP 주소: 192.168.1.100 (예시)
# 서브넷 마스크: 255.255.255.0
# 라우터: 192.168.1.1 (공유기 IP)
```

### 2.3 DDNS 설정 (동적 IP인 경우)
1. No-IP, DuckDNS 등 무료 DDNS 서비스 가입
2. 도메인 생성 (예: sambiowage.ddns.net)
3. 맥미니에 DDNS 클라이언트 설치:
```bash
# DuckDNS 예시
# crontab에 추가
crontab -e
# 5분마다 IP 업데이트
*/5 * * * * curl "https://www.duckdns.org/update?domains=sambiowage&token=YOUR_TOKEN&ip="
```

## 3. 보안 설정

### 3.1 방화벽 설정
```bash
# macOS 방화벽 설정
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate on
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add /usr/local/bin/docker
sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp /usr/local/bin/docker
```

### 3.2 Basic Auth 추가 (nginx)
```bash
# htpasswd 파일 생성
htpasswd -c nginx/.htpasswd admin

# nginx.conf 수정
# location / 블록에 추가:
#   auth_basic "Restricted Access";
#   auth_basic_user_file /etc/nginx/.htpasswd;
```

### 3.3 HTTPS 설정 (선택사항)
```bash
# Let's Encrypt 인증서 발급 (도메인이 있는 경우)
docker run -it --rm \
  -v ~/sambiowage/nginx/ssl:/etc/letsencrypt \
  certbot/certbot certonly \
  --standalone \
  -d yourdomain.com
```

## 4. 모니터링

### 4.1 시스템 리소스 모니터링
```bash
# CPU/메모리 사용량
docker stats

# 디스크 사용량
df -h
```

### 4.2 자동 재시작 설정
```bash
# Docker 자동 시작 설정
sudo systemctl enable docker

# 컨테이너 자동 재시작 (docker-compose.yml에 이미 설정됨)
# restart: always
```

## 5. 접속 테스트

### 5.1 내부 네트워크에서 테스트
```bash
# 맥미니 로컬에서
curl http://localhost

# 같은 네트워크의 다른 기기에서
curl http://192.168.1.100
```

### 5.2 외부에서 테스트
```bash
# 외부 IP 확인
curl ifconfig.me

# 외부에서 접속 (친구에게 테스트 요청)
# http://YOUR_EXTERNAL_IP
# 또는
# http://sambiowage.ddns.net
```

## 6. 문제 해결

### Docker 실행 오류
```bash
# Docker Desktop 재시작
killall Docker && open -a Docker

# 컨테이너 정리
docker system prune -af
```

### 포트 충돌
```bash
# 80 포트 사용 중인 프로세스 확인
sudo lsof -i :80

# Apache/nginx 중지
sudo apachectl stop
sudo nginx -s stop
```

### 외부 접속 안 될 때
1. 공유기 포트포워딩 설정 확인
2. 맥미니 방화벽 설정 확인
3. ISP가 80 포트를 차단하는지 확인 (다른 포트 사용)
4. Docker 컨테이너 실행 상태 확인

## 7. 성능 최적화

### 7.1 Docker 리소스 제한
```yaml
# docker-compose.production.yml에 추가
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
```

### 7.2 로그 로테이션 설정
```yaml
# docker-compose.production.yml에 추가
services:
  nginx:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```