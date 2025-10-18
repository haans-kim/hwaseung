#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3

def check_team_fte():
    """시스템개발 운영팀의 FTE 데이터 확인"""

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    print("=== 시스템개발 운영팀 FTE 데이터 ===")

    # FTE 테이블 구조 확인
    cursor.execute("PRAGMA table_info(FTE)")
    columns = cursor.fetchall()
    print("FTE 테이블 컬럼:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

    # 시스템개발 운영팀 FTE 데이터
    cursor.execute("""
        SELECT * FROM FTE
        WHERE 팀명 = '시스템개발 운영팀'
        ORDER BY id DESC
        LIMIT 5
    """)

    fte_data = cursor.fetchall()
    print(f"\n시스템개발 운영팀 최근 FTE 데이터 ({len(fte_data)}건):")
    for row in fte_data:
        print(f"  {row}")

    # team_headcount 테이블도 확인
    print("\n=== team_headcount 테이블 ===")
    cursor.execute("PRAGMA table_info(team_headcount)")
    columns = cursor.fetchall()
    print("team_headcount 테이블 컬럼:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

    # 시스템개발 운영팀 headcount 데이터
    cursor.execute("""
        SELECT * FROM team_headcount
        WHERE team_name = '시스템개발 운영팀'
        ORDER BY year DESC, month DESC
        LIMIT 5
    """)

    headcount_data = cursor.fetchall()
    print(f"\n시스템개발 운영팀 최근 headcount 데이터 ({len(headcount_data)}건):")
    for row in headcount_data:
        print(f"  {row}")

    # 2025년 최신 데이터 확인
    cursor.execute("""
        SELECT year, month, position, headcount
        FROM team_headcount
        WHERE team_name = '시스템개발 운영팀' AND year = 25
        ORDER BY month DESC, position
    """)

    latest_data = cursor.fetchall()
    print(f"\n시스템개발 운영팀 2025년 데이터:")
    for row in latest_data:
        year, month, position, headcount = row
        print(f"  25년 {month}월 {position}: {headcount}명")

    conn.close()

if __name__ == "__main__":
    check_team_fte()