.PHONY: start stop install clean logs

# 스마트 시작 (포트 자동 감지)
start:
	@chmod +x start_smart.sh
	@./start_smart.sh

# 일반 시작 (고정 포트 - 로컬 개발용)
start-fixed:
	@echo "🚀 고정 포트로 서비스 시작 (로컬 개발)..."
	@cd backend && source venv/bin/activate && python run.py &
	@cd frontend && PORT=2000 npm start

# 프로덕션 모드 (외부 접속 가능)
start-production:
	@echo "🌐 프로덕션 모드로 시작 (외부 접속 가능)..."
	@cd backend && source venv/bin/activate && NETWORK_MODE=external python run.py --host 0.0.0.0 --port 8000 &
	@cd frontend && npm run build && npx serve -s build -l 2000 &
	@sleep 2
	@echo "✅ 서비스 실행 중:"
	@echo "   Frontend: http://$(shell ipconfig getifaddr en0 2>/dev/null || echo localhost):2000"
	@echo "   Backend:  http://$(shell ipconfig getifaddr en0 2>/dev/null || echo localhost):8000"
	@echo ""
	@echo "💡 외부에서 접속하려면 위 IP 주소를 사용하세요"

# 강제 시작 (기존 프로세스 종료 후)
start-force:
	@echo "⚠️  기존 프로세스 종료 중..."
	@lsof -ti:8000 | xargs kill -9 2>/dev/null || true
	@lsof -ti:2000 | xargs kill -9 2>/dev/null || true
	@sleep 1
	@make start-fixed

# 재시작 (기존 프로세스 정리 후 시작)
restart:
	@echo "🔄 서비스 재시작 중..."
	@make stop
	@sleep 3
	@echo "🔍 포트 상태 재확인..."
	@lsof -ti:2000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@sleep 1
	@make start-fixed
	@echo "✅ 재시작 완료!"

# 서비스 중지
stop:
	@echo "🛑 서비스 중지 중..."
	# Python/Backend 프로세스 종료
	@pkill -9 -f "uvicorn" 2>/dev/null || true
	@pkill -9 -f "python.*run.py" 2>/dev/null || true
	@pkill -9 -f "python.*backend" 2>/dev/null || true
	# Node/Frontend 프로세스 종료
	@pkill -9 -f "react-scripts start" 2>/dev/null || true
	@pkill -9 -f "react-app-rewired start" 2>/dev/null || true
	@pkill -9 -f "webpack-dev-server" 2>/dev/null || true
	@pkill -9 -f "node.*frontend" 2>/dev/null || true
	# 포트 기반 종료 (남아있는 것들 처리)
	@lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@lsof -ti:2000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@sleep 1
	# 확인
	@if lsof -ti:8000 >/dev/null 2>&1; then echo "⚠️  포트 8000이 아직 사용 중입니다"; else echo "✅ 포트 8000 해제됨"; fi
	@if lsof -ti:2000 >/dev/null 2>&1; then echo "⚠️  포트 2000이 아직 사용 중입니다"; else echo "✅ 포트 2000 해제됨"; fi
	@echo "✅ 모든 서비스 중지 완료"

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
	@echo "🔍 포트 사용 현황 확인..."
	@echo "--- 포트 2000 (Frontend) ---"
	@lsof -i:2000 2>/dev/null || echo "✅ 포트 2000 사용 안함"
	@echo ""
	@echo "--- 포트 8000 (Backend) ---"
	@lsof -i:8000 2>/dev/null || echo "✅ 포트 8000 사용 안함"
	@echo ""
	@echo "--- 관련 프로세스 ---"
	@ps aux | grep -E "(react|webpack|node.*frontend|python.*run|uvicorn)" | grep -v grep || echo "✅ 관련 프로세스 없음"

# 강제 정리 (모든 관련 프로세스 제거)
kill-all:
	@echo "⚠️  모든 관련 프로세스 강제 종료..."
	@killall -9 node 2>/dev/null || true
	@killall -9 Python 2>/dev/null || true
	@killall -9 python3 2>/dev/null || true
	@lsof -ti:8000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@lsof -ti:2000 2>/dev/null | xargs kill -9 2>/dev/null || true
	@echo "✅ 강제 정리 완료"
	@echo "포트 8000:" && lsof -i:8000 || echo "  ✅ 사용 가능"
	@echo "포트 2000:" && lsof -i:2000 || echo "  ✅ 사용 가능"

# 개발 환경 리셋
clean:
	@echo "🧹 개발 환경 정리 중..."
	@make stop
	@rm -rf backend/venv
	@rm -rf frontend/node_modules
	@rm -rf backend/__pycache__
	@rm -rf backend/app/__pycache__
	@echo "✅ 정리 완료"

# 프론트엔드 빌드
build-frontend:
	@echo "🔨 프론트엔드 빌드 중..."
	@cd frontend && npm run build
	@echo "✅ 빌드 완료 (frontend/build)"

# 빌드된 프론트엔드 실행
serve-frontend:
	@echo "🌐 빌드된 프론트엔드 실행..."
	@cd frontend && npx serve -s build -l 2000

# 도움말
help:
	@echo "사용 가능한 명령어:"
	@echo "  make start            - 스마트 포트 감지로 시작"
	@echo "  make start-fixed      - 고정 포트로 시작 (로컬 개발)"
	@echo "  make start-production - 프로덕션 모드 (외부 접속 가능)"
	@echo "  make start-force      - 강제로 포트 정리 후 시작"
	@echo "  make stop             - 모든 서비스 중지"
	@echo "  make build-frontend   - 프론트엔드 빌드"
	@echo "  make serve-frontend   - 빌드된 프론트엔드 실행"
	@echo "  make install          - 의존성 설치"
	@echo "  make logs             - 로그 확인"
	@echo "  make check-ports      - 포트 상태 확인"
	@echo "  make clean            - 개발 환경 초기화"