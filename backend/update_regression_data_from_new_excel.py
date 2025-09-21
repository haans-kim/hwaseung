#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
from datetime import datetime

def update_team_headcount_from_new_excel():
    """새로운 엑셀 파일에서 25년도 6/7/8월 인력 데이터 업데이트"""

    # 데이터베이스 연결
    conn = sqlite3.connect('/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db')
    cursor = conn.cursor()

    # 엑셀 파일 경로
    excel_file = '/Users/hanskim/Projects/Hwaseung/data/화승 회귀분석_250919.xlsx'

    # 팀별 시트 이름들
    teams = ['시스템개발 운영팀', 'SPECIALTY개발팀', 'SL운영팀', '관체사업팀']

    print("=== 25년도 6/7/8월 인력 데이터 업데이트 ===")

    # 기존 25년도 데이터 삭제
    cursor.execute("DELETE FROM team_headcount WHERE year = 25")

    for team_name in teams:
        print(f"\n처리 중: {team_name}")

        try:
            df = pd.read_excel(excel_file, sheet_name=team_name)

            # 25년도 6, 7, 8월 데이터가 있는 행들 찾기
            for month in [6, 7, 8]:
                # 해당 월의 데이터 중 인력규모가 있는 행들 찾기
                month_data = df[
                    (df['월'] == month) &
                    pd.notna(df['인력규모 (총)'])
                ]

                if len(month_data) > 0:
                    # 첫 번째로 찾은 데이터 사용
                    row = month_data.iloc[0]

                    total_headcount = int(row['인력규모 (총)']) if pd.notna(row['인력규모 (총)']) else 0
                    manager_count = int(row['인력규모 (책임)']) if pd.notna(row['인력규모 (책임)']) else 0
                    senior_count = int(row['인력규모 (선임)']) if pd.notna(row['인력규모 (선임)']) else 0
                    junior_count = int(row['인력규모 (사원)']) if pd.notna(row['인력규모 (사원)']) else 0

                    print(f"  {month}월: 총 {total_headcount}명 (책임:{manager_count}, 선임:{senior_count}, 사원:{junior_count})")

                    # 총 인력 데이터 삽입 (총합)
                    cursor.execute('''
                        INSERT INTO team_headcount (team_name, year, month, position, headcount, flow_logins)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (team_name, 25, month, '총합', total_headcount, 0))

                    # 직급별 데이터 삽입
                    cursor.execute('''
                        INSERT INTO team_headcount (team_name, year, month, position, headcount, flow_logins)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (team_name, 25, month, '책임', manager_count, 0))

                    cursor.execute('''
                        INSERT INTO team_headcount (team_name, year, month, position, headcount, flow_logins)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (team_name, 25, month, '선임', senior_count, 0))

                    cursor.execute('''
                        INSERT INTO team_headcount (team_name, year, month, position, headcount, flow_logins)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (team_name, 25, month, '사원', junior_count, 0))

                else:
                    print(f"  {month}월: 데이터 없음")

        except Exception as e:
            print(f"  오류 발생: {e}")

    # 변경사항 저장
    conn.commit()

    # 결과 확인
    print("\n=== 업데이트된 25년도 데이터 확인 ===")
    cursor.execute('''
        SELECT team_name, year, month, position, headcount
        FROM team_headcount
        WHERE year = 25
        ORDER BY team_name, month,
        CASE position
            WHEN '총합' THEN 1
            WHEN '책임' THEN 2
            WHEN '선임' THEN 3
            WHEN '사원' THEN 4
        END
    ''')

    results = cursor.fetchall()
    for row in results:
        print(f"{row[0]} - {row[1]}년 {row[2]}월 {row[3]}: {row[4]}명")

    conn.close()
    print("\n25년도 인력 데이터 업데이트 완료!")

if __name__ == "__main__":
    update_team_headcount_from_new_excel()