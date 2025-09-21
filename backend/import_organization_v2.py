#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3
import os
from datetime import datetime

def import_organization_to_db():
    """조직도 엑셀 파일을 읽어서 DB에 저장"""

    # 파일 경로
    excel_path = '../data/화승_조직도_정리_2025.09.21.xlsx'
    db_path = '../hwaseung_RnD.db'

    print("조직도 데이터 가져오기 시작...")
    print(f"엑셀 파일: {excel_path}")
    print(f"데이터베이스: {db_path}")

    try:
        # 엑셀 파일 읽기
        df = pd.read_excel(excel_path, engine='openpyxl')
        print(f"\n✅ 엑셀 파일 읽기 완료")
        print(f"  - 총 {len(df)} 행")
        print(f"  - 컬럼: {list(df.columns)}")

        # 데이터 정리
        df = df.fillna('')  # NaN 값을 빈 문자열로 변경

        # 데이터 타입 정리
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
            else:
                df[col] = df[col].astype(str)

        # 타임스탬프 추가
        df['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # SQLite 연결
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 기존 테이블 삭제 (있는 경우)
        cursor.execute("DROP TABLE IF EXISTS organization")
        print("\n✅ 기존 organization 테이블 삭제")

        # 테이블 생성
        create_table_sql = """
        CREATE TABLE organization (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            회사 TEXT,
            본부 TEXT,
            담당_사업단_센터 TEXT,
            실 TEXT,
            팀 TEXT,
            비고 TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
        cursor.execute(create_table_sql)
        print("✅ 새 organization 테이블 생성")

        # 데이터 삽입
        insert_count = 0
        for _, row in df.iterrows():
            insert_sql = """
            INSERT INTO organization
            (회사, 본부, 담당_사업단_센터, 실, 팀, 비고, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            values = (
                row.get('HQ', ''),
                row.get('본부', ''),
                row.get('담당/사업단/센터', ''),
                row.get('실', ''),
                row.get('팀', ''),
                row.get('비고', ''),
                row.get('created_at', ''),
                row.get('updated_at', '')
            )
            cursor.execute(insert_sql, values)
            insert_count += 1

        print(f"✅ {insert_count}개 레코드 삽입 완료")

        # 인덱스 생성
        print("\n인덱스 생성 중...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_org_company ON organization(회사)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_org_dept ON organization(본부)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_org_team ON organization(팀)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_org_division ON organization(담당_사업단_센터)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_org_section ON organization(실)")

        # 커밋
        conn.commit()

        # 통계 정보 확인
        cursor.execute("SELECT COUNT(*) FROM organization")
        total_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT 회사) FROM organization")
        company_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT 본부) FROM organization WHERE 본부 != ''")
        dept_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT 팀) FROM organization WHERE 팀 != ''")
        team_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT 담당_사업단_센터) FROM organization WHERE 담당_사업단_센터 != ''")
        division_count = cursor.fetchone()[0]

        print("\n" + "="*60)
        print("📊 조직도 데이터 가져오기 완료!")
        print("="*60)
        print(f"  - 총 레코드 수: {total_count}")
        print(f"  - 회사 수: {company_count}")
        print(f"  - 본부 수: {dept_count}")
        print(f"  - 담당/사업단/센터 수: {division_count}")
        print(f"  - 팀 수: {team_count}")

        # 샘플 데이터 출력
        print("\n📋 샘플 데이터 (처음 10개 팀):")
        cursor.execute("""
            SELECT 회사, 본부, 담당_사업단_센터, 실, 팀
            FROM organization
            WHERE 팀 != ''
            LIMIT 10
        """)

        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"  {i}. {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")

        # 본부별 팀 수 통계
        print("\n📊 본부별 팀 수:")
        cursor.execute("""
            SELECT 본부, COUNT(DISTINCT 팀) as team_count
            FROM organization
            WHERE 본부 != '' AND 팀 != ''
            GROUP BY 본부
            ORDER BY team_count DESC
        """)

        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]}개 팀")

        # 담당/사업단/센터별 팀 수 통계
        print("\n📈 담당/사업단/센터별 팀 수:")
        cursor.execute("""
            SELECT 담당_사업단_센터, COUNT(DISTINCT 팀) as team_count
            FROM organization
            WHERE 담당_사업단_센터 != '' AND 팀 != ''
            GROUP BY 담당_사업단_센터
            ORDER BY team_count DESC
            LIMIT 10
        """)

        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]}개 팀")

        # 전체 조직 구조 확인
        print("\n🏢 전체 조직 구조 요약:")
        cursor.execute("""
            SELECT
                본부,
                COUNT(DISTINCT 담당_사업단_센터) as division_count,
                COUNT(DISTINCT 실) as section_count,
                COUNT(DISTINCT 팀) as team_count
            FROM organization
            WHERE 본부 != ''
            GROUP BY 본부
            ORDER BY 본부
        """)

        print("\n  본부 | 담당/사업단/센터 수 | 실 수 | 팀 수")
        print("  " + "-"*50)
        for row in cursor.fetchall():
            print(f"  {row[0]:<15} | {row[1]:^18} | {row[2]:^5} | {row[3]:^5}")

        # 연결 종료
        conn.close()
        print("\n✅ 데이터베이스 연결 종료")
        print(f"\n💾 조직도 데이터가 {os.path.abspath(db_path)}에 저장되었습니다.")

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
    success = import_organization_to_db()
    if success:
        print("\n✨ 모든 작업이 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️ 작업 중 오류가 발생했습니다. 로그를 확인해주세요.")