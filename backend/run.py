import uvicorn
import os
import sys
from app.main import app

if __name__ == "__main__":
    # UTF-8 인코딩 강제 설정 (Windows에서 중요)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')

    # Electron 모드 감지
    is_electron = os.getenv("ELECTRON_MODE") == "1"

    port = int(os.getenv("PORT", 8000))

    # 명령줄 인자로 --host 옵션 지원
    if "--host" in sys.argv:
        host_idx = sys.argv.index("--host")
        if host_idx + 1 < len(sys.argv):
            host = sys.argv[host_idx + 1]
        else:
            host = "0.0.0.0"
    else:
        # Electron 모드에서는 localhost만 사용
        if is_electron:
            host = "localhost"
        else:
            # 환경변수 또는 로컬 네트워크에서 접속 가능하도록 0.0.0.0 사용
            host = "0.0.0.0" if os.getenv("ENVIRONMENT") == "production" or os.getenv("NETWORK_MODE") == "external" else "127.0.0.1"

    # --port 옵션 지원
    if "--port" in sys.argv:
        port_idx = sys.argv.index("--port")
        if port_idx + 1 < len(sys.argv):
            port = int(sys.argv[port_idx + 1])

    mode_info = " [Electron Mode]" if is_electron else ""
    print(f"Starting server on {host}:{port}{mode_info}")

    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") != "production" and not is_electron,
        log_level="info"
    )