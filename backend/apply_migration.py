#!/usr/bin/env python3
"""
데이터베이스 마이그레이션 적용 스크립트
"""
import sqlite3
import os
from pathlib import Path

DB_PATH = '/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db'
MIGRATIONS_DIR = Path(__file__).parent / 'migrations'

def apply_migration(migration_file: Path):
    """마이그레이션 SQL 파일 적용"""
    print(f"📄 Applying migration: {migration_file.name}")

    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return False

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
