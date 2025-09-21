#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3
import os
from datetime import datetime

def import_organization_to_db():
    """조직도 CSV 파일을 읽어서 DB에 저장"""

    # 파일 경로
    csv_path = '../data/화승_조직도_정리_2025_v2.csv'
    db_path = '../hwaseung_RnD.db'

    print("조직도 데이터 가져오기 시작...")
    print(f"CSV 파일: {csv_path}")
    print(f"데이터베이스: {db_path}")

    try:
        # CSV 파일 읽기
        df = pd.read_csv(csv_path, encoding='utf-8')
        print(f"\n✅ CSV 파일 읽기 완료")
        print(f"  - 총 {len(df)} 행")
        print(f"  - 컬럼: {list(df.columns)}")

        # 데이터 정리
        df = df.fillna('')  # NaN 값을 빈 문자열로 변경

        # 데이터 타입 정리
        for col in df.columns:
            df[col] = df[col].astype(str).str.strip()

        # 타임스탬프 추가
        df['created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df['updated_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # SQLite 연결
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # 기존 테이블 삭제 (있는 경우)
        cursor.execute("DROP TABLE IF EXISTS organization")
        print("\n기존 organization 테이블 삭제")

        # 테이블 생성
        create_table_sql = """
        CREATE TABLE organization (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            회사 TEXT,
            최상위조직 TEXT,
            본부_사업본부 TEXT,
            담당_실_센터 TEXT,
            팀 TEXT,
            비고 TEXT,
            created_at TEXT,
            updated_at TEXT
        )
        """
        cursor.execute(create_table_sql)
        print("새 organization 테이블 생성")

        # 컬럼명 변경 (슬래시를 언더스코어로)
        df_renamed = df.copy()
        df_renamed.columns = [col.replace('/', '_').replace(' ', '_') for col in df.columns]

        # 데이터 삽입
        for _, row in df.iterrows():
            insert_sql = """
            INSERT INTO organization
            (회사, 최상위조직, 본부_사업본부, 담당_실_센터, 팀, 비고, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            values = (
                row.get('회사', ''),
                row.get('최상위조직', ''),
                row.get('본부/사업본부', ''),
                row.get('담당/실/센터', ''),
                row.get('팀', ''),
                row.get('비고', ''),
                row.get('created_at', ''),
                row.get('updated_at', '')
            )
            cursor.execute(insert_sql, values)

        # 인덱스 생성
        print("\n인덱스 생성 중...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_org_company ON organization(회사)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_org_department ON organization(본부_사업본부)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_org_team ON organization(팀)")

        # 커밋
        conn.commit()

        # 통계 정보 확인
        cursor.execute("SELECT COUNT(*) FROM organization")
        total_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT 회사) FROM organization")
        company_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT 본부_사업본부) FROM organization WHERE 본부_사업본부 != ''")
        dept_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT 팀) FROM organization WHERE 팀 != ''")
        team_count = cursor.fetchone()[0]

        print("\n" + "="*60)
        print("📊 조직도 데이터 가져오기 완료!")
        print("="*60)
        print(f"  - 총 레코드 수: {total_count}")
        print(f"  - 회사 수: {company_count}")
        print(f"  - 본부/사업본부 수: {dept_count}")
        print(f"  - 팀 수: {team_count}")

        # 샘플 데이터 출력
        print("\n📋 샘플 데이터 (처음 10개):")
        cursor.execute("""
            SELECT 회사, 최상위조직, 본부_사업본부, 팀
            FROM organization
            WHERE 팀 != ''
            LIMIT 10
        """)

        for i, row in enumerate(cursor.fetchall(), 1):
            print(f"  {i}. {row[0]} | {row[1]} | {row[2]} | {row[3]}")

        # 회사별 팀 수 통계
        print("\n📈 회사별 팀 수:")
        cursor.execute("""
            SELECT 회사, COUNT(DISTINCT 팀) as team_count
            FROM organization
            WHERE 팀 != ''
            GROUP BY 회사
            ORDER BY team_count DESC
        """)

        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]}개 팀")

        # 본부별 팀 수 통계
        print("\n📊 본부별 팀 수 (상위 10개):")
        cursor.execute("""
            SELECT 본부_사업본부, COUNT(DISTINCT 팀) as team_count
            FROM organization
            WHERE 본부_사업본부 != '' AND 팀 != ''
            GROUP BY 본부_사업본부
            ORDER BY team_count DESC
            LIMIT 10
        """)

        for row in cursor.fetchall():
            print(f"  - {row[0]}: {row[1]}개 팀")

        # 연결 종료
        conn.close()
        print("\n✅ 데이터베이스 연결 종료")
        print(f"\n💾 조직도 데이터가 {os.path.abspath(db_path)}에 저장되었습니다.")

    except FileNotFoundError:
        print(f"\n❌ CSV 파일을 찾을 수 없습니다: {csv_path}")
    except Exception as e:
        print(f"\n❌ 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import_organization_to_db()