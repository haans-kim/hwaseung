"""
Database Configuration
프로젝트 전체에서 사용할 데이터베이스 경로 설정

환경별 DB 경로:
- 로컬 개발: <project_root>/hwaseung_RnD.db
- Docker: /app/hwaseung_RnD.db (볼륨 마운트)
- Electron: app.getPath('userData')/hwaseung_RnD.db (향후 지원)
"""

import os
from pathlib import Path

def get_db_path():
    """
    데이터베이스 경로 반환

    우선순위:
    1. DB_PATH 환경변수 (Docker, Electron에서 사용)
    2. 프로젝트 루트의 hwaseung_RnD.db (로컬 개발)
    """
    # 환경변수 확인 (최우선)
    db_path = os.getenv('DB_PATH')
    if db_path:
        if not os.path.exists(db_path):
            print(f"Warning: DB_PATH environment variable set but file not found: {db_path}")
        return db_path

    # 로컬 개발 환경: 프로젝트 루트 찾기
    # backend/db_config.py -> backend/ -> project_root/
    current_file = Path(__file__)
    backend_dir = current_file.parent
    project_root = backend_dir.parent

    # 프로젝트 루트에 있는 DB 파일
    db_file = project_root / 'hwaseung_RnD.db'

    if not db_file.exists():
        print(f"Warning: Database file not found at expected location: {db_file}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Backend directory: {backend_dir}")
        print(f"Project root: {project_root}")

    return str(db_file)

# 전역 상수로 export
DB_PATH = get_db_path()

# 디버그 정보 출력 (개발 시에만)
if os.getenv('DEBUG'):
    print(f"[DB_CONFIG] Using database at: {DB_PATH}")
    print(f"[DB_CONFIG] File exists: {os.path.exists(DB_PATH)}")
