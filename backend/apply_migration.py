#!/usr/bin/env python3
"""
데이터베이스 마이그레이션 적용 스크립트
"""
import sqlite3
import os
from pathlib import Path

# 🔧 FIX: Docker 환경을 위해 환경 변수 또는 상대 경로 사용
DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), 'hwaseung_RnD.db'))
MIGRATIONS_DIR = Path(__file__).parent / 'migrations'

def apply_migration(migration_file: Path):
    """마이그레이션 SQL 파일 적용"""
    print(f"📄 Applying migration: {migration_file.name}")
    print(f"   Using DB path: {DB_PATH}")

    # DB 파일이 없으면 자동 생성됨 (sqlite3.connect가 생성함)
    if not os.path.exists(os.path.dirname(DB_PATH)):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        print(f"   Created directory: {os.path.dirname(DB_PATH)}")

    try:
        with open(migration_file, 'r', encoding='utf-8') as f:
            sql_script = f.read()

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Execute the entire script
        cursor.executescript(sql_script)

        conn.commit()
        conn.close()

        print(f"✅ Migration {migration_file.name} applied successfully")
        return True

    except Exception as e:
        print(f"❌ Error applying migration {migration_file.name}: {e}")
        return False

def main():
    """모든 마이그레이션 파일 적용"""
    print("🚀 Starting database migration...\n")

    if not MIGRATIONS_DIR.exists():
        print(f"❌ Migrations directory not found: {MIGRATIONS_DIR}")
        return

    # Get all .sql files in migrations directory, sorted
    migration_files = sorted(MIGRATIONS_DIR.glob('*.sql'))

    if not migration_files:
        print("⚠️  No migration files found")
        return

    print(f"📋 Found {len(migration_files)} migration file(s)\n")

    success_count = 0
    for migration_file in migration_files:
        if apply_migration(migration_file):
            success_count += 1
        print()  # Empty line for readability

    print(f"🎉 Migration complete: {success_count}/{len(migration_files)} successful")

if __name__ == '__main__':
    main()
