"""
Database Configuration
프로젝트 전체에서 사용할 데이터베이스 경로 설정
"""

import os
from pathlib import Path

def get_db_path():
    """
    데이터베이스 경로 반환

    우선순위:
    1. DB_PATH 환경변수
    2. 프로젝트 루트의 hwaseung_RnD.db
    """
    # 환경변수 확인
    db_path = os.getenv('DB_PATH')
    if db_path:
        return db_path

    # 현재 파일의 위치를 기준으로 프로젝트 루트 찾기
    # backend/db_config.py -> backend/ -> project_root/
    current_file = Path(__file__)
    backend_dir = current_file.parent
    project_root = backend_dir.parent

    # 프로젝트 루트에 있는 DB 파일
    db_file = project_root / 'hwaseung_RnD.db'

    return str(db_file)

# 전역 상수로 export
DB_PATH = get_db_path()
