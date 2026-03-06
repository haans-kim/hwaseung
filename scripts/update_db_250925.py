#!/usr/bin/env python3
import pandas as pd
import sqlite3
from datetime import datetime

DB_PATH = '/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db'
ORG_FILE = '/Users/hanskim/Projects/Hwaseung/data/화승 조직도 정리_250925.xlsx'
FTE_RA_FILE = '/Users/hanskim/Projects/Hwaseung/data/FTE계산_R*A_250924.xlsx'
FTE_CORP_FILE = '/Users/hanskim/Projects/Hwaseung/data/FTE계산_통합기술본부_250924.xlsx'

def update_organization_table(conn):
    """
    조직도 데이터 업데이트
    - R*A 시트: 30개 팀
    - 통합기술본부 시트: 20개 팀
    - 총 50개 행 (중복 8개 팀 포함)
    """
    print("=" * 80)
    print("1. Organization 테이블 업데이트 시작")
    print("=" * 80)

    cursor = conn.cursor()
    cursor.execute("DELETE FROM organization")
    conn.commit()
    print("기존 organization 데이터 삭제 완료")

    org_ra = pd.read_excel(ORG_FILE, sheet_name='R*A')
    org_corp = pd.read_excel(ORG_FILE, sheet_name='통합기술본부')

    print(f"\nR*A 시트: {len(org_ra)}개 팀")
    print(f"통합기술본부 시트: {len(org_corp)}개 팀")

    now = datetime.now().isoformat()

    for _, row in org_ra.iterrows():
        cursor.execute("""
            INSERT INTO organization (회사, 본부, 담당_사업단_센터, 실, 팀, 비고, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['HQ'],
            row['본부'],
            row['담당/사업단/센터'],
            row['실'],
            row['팀'],
            row['비고'] if pd.notna(row['비고']) else None,
            now,
            now
        ))

    for _, row in org_corp.iterrows():
        cursor.execute("""
            INSERT INTO organization (회사, 본부, 담당_사업단_센터, 실, 팀, 비고, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['HQ'],
            row['본부'],
            row['담당/사업단/센터'],
            row['실'],
            row['팀'],
            row['비고'] if pd.notna(row['비고']) else None,
            now,
            now
        ))

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM organization")
    total_count = cursor.fetchone()[0]
    cursor.execute("SELECT 회사, COUNT(*) FROM organization GROUP BY 회사")
    company_counts = cursor.fetchall()

    print(f"\n삽입 완료: 총 {total_count}개 행")
    for company, count in company_counts:
        print(f"  - {company}: {count}개 팀")

    print("\n✅ Organization 테이블 업데이트 완료\n")

def add_company_column_to_fte(conn):
    """
    FTE 테이블에 회사 컬럼 추가
    """
    print("=" * 80)
    print("2. FTE 테이블 스키마 수정 시작")
    print("=" * 80)

    cursor = conn.cursor()

    cursor.execute("PRAGMA table_info(fte)")
    columns = [col[1] for col in cursor.fetchall()]

    if '회사' not in columns:
        print("회사 컬럼 추가 중...")
        cursor.execute("ALTER TABLE fte ADD COLUMN 회사 TEXT")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_fte_company ON fte(회사)")
        conn.commit()
        print("✅ 회사 컬럼 추가 완료")
    else:
        print("회사 컬럼이 이미 존재합니다")

    print()

def parse_fte_data(file_path, company_name):
    """
    FTE 엑셀 파일의 'FTE 정리' 시트를 파싱하여 DataFrame 반환
    """
    df = pd.read_excel(file_path, sheet_name='FTE 정리', header=None)

    header_row = 0
    data_start_row = 1

    teams = []
    for idx in range(data_start_row, len(df)):
        team_name = df.iloc[idx, 0]

        if pd.isna(team_name) or team_name == '팀':
            continue

        fte_total = df.iloc[idx, 1] if pd.notna(df.iloc[idx, 1]) else 0
        fte_책임 = df.iloc[idx, 2] if pd.notna(df.iloc[idx, 2]) else 0
        fte_선임 = df.iloc[idx, 3] if pd.notna(df.iloc[idx, 3]) else 0
        fte_사원 = df.iloc[idx, 4] if pd.notna(df.iloc[idx, 4]) else 0

        인원_total = df.iloc[idx, 5] if pd.notna(df.iloc[idx, 5]) else 0
        인원_책임 = df.iloc[idx, 6] if pd.notna(df.iloc[idx, 6]) else 0
        인원_선임 = df.iloc[idx, 7] if pd.notna(df.iloc[idx, 7]) else 0
        인원_사원 = df.iloc[idx, 8] if pd.notna(df.iloc[idx, 8]) else 0

        fte_per_인원_total = df.iloc[idx, 9] if pd.notna(df.iloc[idx, 9]) else 0
        fte_per_인원_책임 = df.iloc[idx, 10] if pd.notna(df.iloc[idx, 10]) else 0
        fte_per_인원_선임 = df.iloc[idx, 11] if pd.notna(df.iloc[idx, 11]) else 0
        fte_per_인원_사원 = df.iloc[idx, 12] if pd.notna(df.iloc[idx, 12]) else 0

        teams.append({
            '팀명': team_name,
            '회사': company_name,
            'FTE_전체': float(fte_total),
            'FTE_책임': float(fte_책임),
            'FTE_선임': float(fte_선임),
            'FTE_사원': float(fte_사원),
            '인원수_전체': int(인원_total),
            '인원수_책임': int(인원_책임),
            '인원수_선임': int(인원_선임),
            '인원수_사원': int(인원_사원),
            'FTE_per_인원_전체': float(fte_per_인원_total),
            'FTE_per_인원_책임': float(fte_per_인원_책임),
            'FTE_per_인원_선임': float(fte_per_인원_선임),
            'FTE_per_인원_사원': float(fte_per_인원_사원)
        })

    return pd.DataFrame(teams)

def update_fte_table(conn):
    """
    FTE 데이터 업데이트
    - R*A: 31개 팀
    - 화승 Corp.: 20개 팀
    - 중복 8개 팀은 회사별로 별도 행으로 저장
    """
    print("=" * 80)
    print("3. FTE 테이블 데이터 업데이트 시작")
    print("=" * 80)

    cursor = conn.cursor()
    cursor.execute("DELETE FROM fte")
    conn.commit()
    print("기존 FTE 데이터 삭제 완료")

    fte_ra = parse_fte_data(FTE_RA_FILE, '화승 R*A')
    fte_corp = parse_fte_data(FTE_CORP_FILE, '화승 Corp.')

    print(f"\nR*A FTE: {len(fte_ra)}개 팀")
    print(f"Corp FTE: {len(fte_corp)}개 팀")

    now = datetime.now().isoformat()
    period = "2024.06-08"

    for _, row in fte_ra.iterrows():
        cursor.execute("""
            INSERT INTO fte (
                팀명, 회사, 기간,
                FTE_전체, FTE_책임, FTE_선임, FTE_사원,
                인원수_전체, 인원수_책임, 인원수_선임, 인원수_사원,
                FTE_per_인원_전체, FTE_per_인원_책임, FTE_per_인원_선임, FTE_per_인원_사원,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['팀명'], row['회사'], period,
            row['FTE_전체'], row['FTE_책임'], row['FTE_선임'], row['FTE_사원'],
            row['인원수_전체'], row['인원수_책임'], row['인원수_선임'], row['인원수_사원'],
            row['FTE_per_인원_전체'], row['FTE_per_인원_책임'], row['FTE_per_인원_선임'], row['FTE_per_인원_사원'],
            now, now
        ))

    for _, row in fte_corp.iterrows():
        cursor.execute("""
            INSERT INTO fte (
                팀명, 회사, 기간,
                FTE_전체, FTE_책임, FTE_선임, FTE_사원,
                인원수_전체, 인원수_책임, 인원수_선임, 인원수_사원,
                FTE_per_인원_전체, FTE_per_인원_책임, FTE_per_인원_선임, FTE_per_인원_사원,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row['팀명'], row['회사'], period,
            row['FTE_전체'], row['FTE_책임'], row['FTE_선임'], row['FTE_사원'],
            row['인원수_전체'], row['인원수_책임'], row['인원수_선임'], row['인원수_사원'],
            row['FTE_per_인원_전체'], row['FTE_per_인원_책임'], row['FTE_per_인원_선임'], row['FTE_per_인원_사원'],
            now, now
        ))

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM fte")
    total_count = cursor.fetchone()[0]
    cursor.execute("SELECT 회사, COUNT(*) FROM fte GROUP BY 회사")
    company_counts = cursor.fetchall()

    print(f"\n삽입 완료: 총 {total_count}개 행")
    for company, count in company_counts:
        print(f"  - {company}: {count}개 팀")

    print("\n✅ FTE 테이블 업데이트 완료\n")

def verify_data(conn):
    """
    데이터 검증 및 매칭률 확인
    """
    print("=" * 80)
    print("4. 데이터 검증")
    print("=" * 80)

    cursor = conn.cursor()

    cursor.execute("""
        SELECT o.회사, o.팀, f.팀명, f.회사
        FROM organization o
        LEFT JOIN fte f ON o.팀 = f.팀명 AND o.회사 = f.회사
    """)

    results = cursor.fetchall()
    matched = sum(1 for r in results if r[2] is not None)
    total = len(results)

    print(f"\n조직도-FTE 매칭률: {matched}/{total} ({matched/total*100:.1f}%)")

    unmatched = [r for r in results if r[2] is None]
    if unmatched:
        print(f"\n매칭 안된 팀 ({len(unmatched)}개):")
        for org_company, org_team, _, _ in unmatched:
            print(f"  - {org_company} / {org_team}")

    cursor.execute("""
        SELECT 팀명, 회사
        FROM fte
        WHERE 팀명 NOT IN (SELECT DISTINCT 팀 FROM organization)
    """)
    unmatched_fte = cursor.fetchall()

    if unmatched_fte:
        print(f"\n조직도에 없는 FTE 데이터 ({len(unmatched_fte)}개):")
        for team, company in unmatched_fte:
            print(f"  - {company} / {team}")

    cursor.execute("""
        SELECT 팀명, COUNT(*) as cnt
        FROM fte
        GROUP BY 팀명
        HAVING cnt > 1
    """)
    duplicates = cursor.fetchall()

    if duplicates:
        print(f"\n중복된 팀 ({len(duplicates)}개) - 회사별로 다른 FTE 값:")
        for team, count in duplicates:
            print(f"  - {team}: {count}개 행")
            cursor.execute("SELECT 회사, FTE_per_인원_전체 FROM fte WHERE 팀명 = ?", (team,))
            details = cursor.fetchall()
            for company, fte_val in details:
                print(f"      {company}: {fte_val:.2f}")

    print("\n✅ 검증 완료\n")

def main():
    print("\n" + "=" * 80)
    print("화승 조직도 및 FTE 데이터 업데이트 (2025.09.25)")
    print("=" * 80 + "\n")

    try:
        conn = sqlite3.connect(DB_PATH)

        update_organization_table(conn)
        add_company_column_to_fte(conn)
        update_fte_table(conn)
        verify_data(conn)

        conn.close()

        print("=" * 80)
        print("✅ 모든 업데이트 완료!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()