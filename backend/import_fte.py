#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3
import os
from datetime import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def import_fte_to_db():
    """FTE 엑셀 파일을 읽어서 DB에 저장"""

    # 파일 경로
    excel_path = '../data/FTE계산_2025.0.21.xlsx'
    db_path = '../hwaseung_RnD.db'

    print("FTE 데이터 가져오기 시작...")
    print(f"엑셀 파일: {excel_path}")
    print(f"데이터베이스: {db_path}")

    try:
        # FTE 정리 시트 읽기
        df = pd.read_excel(excel_path, sheet_name='FTE 정리')
        print(f"\n✅ 엑셀 파일 읽기 완료")
        print(f"  - 총 {len(df)} 행")

        # 첫 번째 행을 컬럼 헤더로 사용하고, 실제 데이터는 두 번째 행부터
        # 첫 번째 행이 헤더 정보를 포함하고 있음
        header_row = df.iloc[0]
        df_data = df.iloc[1:].copy()  # 실제 데이터는 2번째 행부터

        # 컬럼 이름 정리
        df_data.columns = [
            '팀명',                    # Unnamed: 0
            'FTE_전체',               # 평균 FTE (6월 ~ 8월)
            'FTE_책임',               # Unnamed: 2
            'FTE_선임',               # Unnamed: 3
            'FTE_사원',               # Unnamed: 4
            '인원수_전체',            # 현재 인원수
            '인원수_책임',            # Unnamed: 6
            '인원수_선임',            # Unnamed: 7
            '인원수_사원',            # Unnamed: 8
            'FTE_per_인원_전체',      # FTE/인원수
            'FTE_per_인원_책임',      # Unnamed: 10
            'FTE_per_인원_선임',      # Unnamed: 11
            'FTE_per_인원_사원'       # Unnamed: 12
        ]

        # 팀명이 있는 행만 필터링
        df_data = df_data[df_data['팀명'].notna()].copy()
        df_data = df_data[df_data['팀명'] != '팀'].copy()  # 헤더 행 제외

        print(f"  - 유효한 팀 데이터: {len(df_data)} 행")

        # 데이터 타입 변환 및 정리
        numeric_columns = [col for col in df_data.columns if col != '팀명']

        for col in numeric_columns:
            # 숫자로 변환 시도, 실패하면 0으로 처리
            df_data[col] = pd.to_numeric(df_data[col], errors='coerce').fillna(0)

        # 0으로 나누기 방지를 위한 FTE/인원수 재계산
        for position in ['전체', '책임', '선임', '사원']:
            fte_col = f'FTE_{position}'
            headcount_col = f'인원수_{position}'
            ratio_col = f'FTE_per_인원_{position}'

            # 인원수가 0이 아닌 경우만 계산, 0인 경우 0으로 설정
            df_data[ratio_col] = df_data.apply(
                lambda row: row[fte_col] / row[headcount_col] if row[headcount_col] > 0 else 0,
                axis=1
            )

        # 기간 정보 추가 (파일명에서 추출)
        df_data['기간'] = '2025.06-08'  # 6월~8월 평균

        # 타임스탬프 추가
        df_data['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df_data['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # SQLite 연결
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 기존 테이블 삭제
        cursor.execute("DROP TABLE IF EXISTS fte")
        print("\n✅ 기존 FTE 테이블 삭제")

        # 테이블 생성
        create_table_sql = """
        CREATE TABLE fte (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            팀명 TEXT,
            기간 TEXT,
            FTE_전체 REAL,
            FTE_책임 REAL,
            FTE_선임 REAL,
            FTE_사원 REAL,
            인원수_전체 INTEGER,
            인원수_책임 INTEGER,
            인원수_선임 INTEGER,
            인원수_사원 INTEGER,
            FTE_per_인원_전체 REAL,
            FTE_per_인원_책임 REAL,
            FTE_per_인원_선임 REAL,
            FTE_per_인원_사원 REAL,
            created_at TEXT,
            updated_at TEXT
        )
        """
        cursor.execute(create_table_sql)
        print("✅ 새 FTE 테이블 생성")

        # 데이터 삽입
        insert_count = 0
        for _, row in df_data.iterrows():
            insert_sql = """
            INSERT INTO fte (
                팀명, 기간,
                FTE_전체, FTE_책임, FTE_선임, FTE_사원,
                인원수_전체, 인원수_책임, 인원수_선임, 인원수_사원,
                FTE_per_인원_전체, FTE_per_인원_책임, FTE_per_인원_선임, FTE_per_인원_사원,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            values = (
                row['팀명'], row['기간'],
                round(row['FTE_전체'], 2),
                round(row['FTE_책임'], 2),
                round(row['FTE_선임'], 2),
                round(row['FTE_사원'], 2),
                int(row['인원수_전체']),
                int(row['인원수_책임']),
                int(row['인원수_선임']),
                int(row['인원수_사원']),
                round(row['FTE_per_인원_전체'], 3),
                round(row['FTE_per_인원_책임'], 3),
                round(row['FTE_per_인원_선임'], 3),
                round(row['FTE_per_인원_사원'], 3),
                row['created_at'], row['updated_at']
            )
            cursor.execute(insert_sql, values)
            insert_count += 1

        print(f"✅ {insert_count}개 레코드 삽입 완료")

        # 인덱스 생성
        print("\n인덱스 생성 중...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fte_team ON fte(팀명)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fte_period ON fte(기간)")

        # 커밋
        conn.commit()

        # 통계 정보 출력
        print("\n" + "="*60)
        print("📊 FTE 데이터 가져오기 완료!")
        print("="*60)

        # 전체 통계
        cursor.execute("""
            SELECT
                COUNT(*) as team_count,
                AVG(FTE_전체) as avg_fte,
                SUM(인원수_전체) as total_headcount,
                AVG(FTE_per_인원_전체) as avg_fte_per_person
            FROM fte
        """)
        stats = cursor.fetchone()
        print(f"  - 총 팀 수: {stats[0]}")
        print(f"  - 평균 FTE (팀당): {stats[1]:.2f}")
        print(f"  - 전체 인원수: {stats[2]}")
        print(f"  - 평균 FTE/인원: {stats[3]:.3f}")

        # FTE가 높은 상위 10개 팀
        print("\n📈 FTE 상위 10개 팀:")
        cursor.execute("""
            SELECT 팀명, FTE_전체, 인원수_전체, FTE_per_인원_전체
            FROM fte
            ORDER BY FTE_전체 DESC
            LIMIT 10
        """)
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"  {i:2}. {row[0]:<15} | FTE: {row[1]:>7.2f} | 인원: {row[2]:>3} | FTE/인원: {row[3]:.3f}")

        # FTE/인원 비율이 높은 팀 (인원 5명 이상)
        print("\n⚡ FTE/인원 비율 상위 10개 팀 (인원 5명 이상):")
        cursor.execute("""
            SELECT 팀명, FTE_per_인원_전체, FTE_전체, 인원수_전체
            FROM fte
            WHERE 인원수_전체 >= 5
            ORDER BY FTE_per_인원_전체 DESC
            LIMIT 10
        """)
        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"  {i:2}. {row[0]:<15} | 비율: {row[1]:.3f} | FTE: {row[2]:>7.2f} | 인원: {row[3]:>3}")

        # 인원이 0인 팀들 확인
        cursor.execute("""
            SELECT 팀명, FTE_전체
            FROM fte
            WHERE 인원수_전체 = 0 AND FTE_전체 > 0
        """)
        zero_headcount = cursor.fetchall()
        if zero_headcount:
            print("\n⚠️ 인원수가 0이지만 FTE가 있는 팀:")
            for row in zero_headcount:
                print(f"  - {row[0]}: FTE {row[1]:.2f}")

        # 직급별 통계
        print("\n👥 직급별 전체 통계:")
        cursor.execute("""
            SELECT
                SUM(FTE_책임) as total_fte_책임,
                SUM(FTE_선임) as total_fte_선임,
                SUM(FTE_사원) as total_fte_사원,
                SUM(인원수_책임) as total_인원_책임,
                SUM(인원수_선임) as total_인원_선임,
                SUM(인원수_사원) as total_인원_사원
            FROM fte
        """)
        position_stats = cursor.fetchone()
        print(f"  - 책임: FTE {position_stats[0]:>7.2f} / 인원 {position_stats[3]:>4}")
        print(f"  - 선임: FTE {position_stats[1]:>7.2f} / 인원 {position_stats[4]:>4}")
        print(f"  - 사원: FTE {position_stats[2]:>7.2f} / 인원 {position_stats[5]:>4}")

        # 연결 종료
        conn.close()
        print("\n✅ 데이터베이스 연결 종료")
        print(f"\n💾 FTE 데이터가 {os.path.abspath(db_path)}에 저장되었습니다.")

        return True

    except FileNotFoundError:
        print(f"\n❌ 엑셀 파일을 찾을 수 없습니다: {excel_path}")
        return False
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = import_fte_to_db()
    if success:
        print("\n✨ 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️ 작업 중 오류가 발생했습니다. 로그를 확인해주세요.")