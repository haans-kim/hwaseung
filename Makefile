.PHONY: start stop install clean logs

# 스마트 시작 (포트 자동 감지)
start:
	@chmod +x start_smart.sh
	@./start_smart.sh

# 일반 시작 (고정 포트)
start-fixed:
	@echo "🚀 고정 포트로 서비스 시작..."
	@cd backend && source venv/bin/activate && python run.py &
	@cd frontend && npm start

# 강제 시작 (기존 프로세스 종료 후)
start-force:
	@echo "⚠️  기존 프로세스 종료 중..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@sleep 1
	@make start-fixed

# 재시작 (기존 프로세스 정리 후 시작)
restart:
	@echo "🔄 서비스 재시작 중..."
	@make stop
	@sleep 2
	@make start-fixed
	@echo "✅ 재시작 완료!"

# 서비스 중지
stop:
	@echo "🛑 서비스 중지 중..."
	@pkill -f "uvicorn" || true
	@pkill -f "python run.py" || true
	@pkill -f "npm start" || true
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	@echo "✅ 모든 서비스가 중지되었습니다"

# 의존성 설치
install:
	@echo "📦 의존성 설치 중..."
	@cd backend && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
	@cd frontend && npm install
	@echo "✅ 설치 완료"

# 로그 확인
logs:
	@tail -f backend/logs.log

# 포트 상태 확인
check-ports:
	@echo "🔍 포트 상태 확인..."
	@echo "포트 8000:" && lsof -i:8000 || echo "  ✅ 사용 가능"
	@echo "포트 3000:" && lsof -i:3000 || echo "  ✅ 사용 가능"

# 개발 환경 리셋
clean:
	@echo "🧹 개발 환경 정리 중..."
	@make stop
	@rm -rf backend/venv
	@rm -rf frontend/node_modules
	@rm -rf backend/__pycache__
	@rm -rf backend/app/__pycache__
	@echo "✅ 정리 완료"

# 도움말
help:
	@echo "사용 가능한 명령어:"
	@echo "  make start       - 스마트 포트 감지로 시작"
	@echo "  make start-force - 강제로 포트 정리 후 시작"
	@echo "  make stop        - 모든 서비스 중지"
	@echo "  make install     - 의존성 설치"
	@echo "  make logs        - 로그 확인"
	@echo "  make check-ports - 포트 상태 확인"
	@echo "  make clean       - 개발 환경 초기화"