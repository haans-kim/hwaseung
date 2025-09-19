#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime

def connect_db():
    """데이터베이스 연결"""
    db_path = '../data/work_history.db'
    return sqlite3.connect(db_path)

def print_basic_stats():
    """기본 통계 정보 출력"""
    conn = connect_db()
    cursor = conn.cursor()

    print("="*60)
    print("📊 근무 이력 데이터 분석")
    print("="*60)

    # 전체 레코드 수
    cursor.execute("SELECT COUNT(*) FROM work_history")
    total_records = cursor.fetchone()[0]
    print(f"\n📌 전체 레코드 수: {total_records:,}")

    # 기간별 통계
    print("\n📅 기간별 데이터:")
    cursor.execute("""
        SELECT period, COUNT(*) as count,
               COUNT(DISTINCT user_id) as unique_users
        FROM work_history
        GROUP BY period
        ORDER BY period
    """)
    for row in cursor.fetchall():
        print(f"  - {row[0]}: {row[1]:,} 레코드, {row[2]} 명")

    # 부서별 인원 통계
    print("\n🏢 상위 10개 부서별 인원수:")
    cursor.execute("""
        SELECT department, COUNT(DISTINCT user_id) as user_count
        FROM work_history
        GROUP BY department
        ORDER BY user_count DESC
        LIMIT 10
    """)
    for row in cursor.fetchall():
        print(f"  - {row[0]}: {row[1]} 명")

    # 직위별 통계
    print("\n👤 직위별 분포:")
    cursor.execute("""
        SELECT user_position, COUNT(DISTINCT user_id) as user_count
        FROM work_history
        GROUP BY user_position
        ORDER BY user_count DESC
    """)
    for row in cursor.fetchall():
        print(f"  - {row[0]}: {row[1]} 명")

    conn.close()

def analyze_work_patterns():
    """근무 패턴 분석"""
    conn = connect_db()

    print("\n" + "="*60)
    print("⏰ 근무 패턴 분석")
    print("="*60)

    # 요일별 근무 현황
    print("\n📅 요일별 활동 현황:")
    query = """
        SELECT weekday, COUNT(*) as activity_count,
               COUNT(DISTINCT user_id) as unique_users
        FROM work_history
        GROUP BY weekday
        ORDER BY
            CASE weekday
                WHEN '월요일' THEN 1
                WHEN '화요일' THEN 2
                WHEN '수요일' THEN 3
                WHEN '목요일' THEN 4
                WHEN '금요일' THEN 5
                WHEN '토요일' THEN 6
                WHEN '일요일' THEN 7
            END
    """
    df = pd.read_sql_query(query, conn)
    for _, row in df.iterrows():
        print(f"  - {row['weekday']}: {row['activity_count']:,} 활동, {row['unique_users']} 명")

    # 주말 근무자 분석
    print("\n🔴 주말 근무 현황:")
    query_weekend = """
        SELECT period,
               COUNT(DISTINCT CASE WHEN weekday IN ('토요일', '일요일') THEN user_id END) as weekend_workers,
               COUNT(DISTINCT user_id) as total_workers
        FROM work_history
        GROUP BY period
        ORDER BY period
    """
    df_weekend = pd.read_sql_query(query_weekend, conn)
    for _, row in df_weekend.iterrows():
        weekend_ratio = (row['weekend_workers'] / row['total_workers']) * 100 if row['total_workers'] > 0 else 0
        print(f"  - {row['period']}: {row['weekend_workers']}명 / {row['total_workers']}명 ({weekend_ratio:.1f}%)")

    conn.close()

def analyze_department_activity():
    """부서별 활동 분석"""
    conn = connect_db()

    print("\n" + "="*60)
    print("🏢 부서별 활동 분석")
    print("="*60)

    # 부서별 평균 활동량
    query = """
        SELECT department,
               COUNT(DISTINCT user_id) as users,
               COUNT(*) as total_activities,
               ROUND(CAST(COUNT(*) AS FLOAT) / COUNT(DISTINCT user_id), 1) as avg_activities_per_user
        FROM work_history
        WHERE department IS NOT NULL
        GROUP BY department
        HAVING COUNT(DISTINCT user_id) >= 5
        ORDER BY avg_activities_per_user DESC
        LIMIT 15
    """

    print("\n📊 부서별 평균 활동량 (인원 5명 이상):")
    df = pd.read_sql_query(query, conn)
    for _, row in df.iterrows():
        print(f"  - {row['department']}: 평균 {row['avg_activities_per_user']:.1f} 활동/인 (총 {row['users']}명)")

    conn.close()

def analyze_user_activity():
    """개인별 활동 분석"""
    conn = connect_db()

    print("\n" + "="*60)
    print("👤 개인별 활동 분석")
    print("="*60)

    # 가장 활발한 사용자
    query_active = """
        SELECT user_name, user_position, department,
               COUNT(*) as activity_count,
               COUNT(DISTINCT work_date) as work_days
        FROM work_history
        GROUP BY user_id, user_name, user_position, department
        ORDER BY activity_count DESC
        LIMIT 10
    """

    print("\n🏆 가장 활발한 사용자 TOP 10:")
    df = pd.read_sql_query(query_active, conn)
    for i, row in df.iterrows():
        print(f"  {i+1}. {row['user_name']} ({row['user_position']}, {row['department']})")
        print(f"     활동: {row['activity_count']:,}회, 근무일: {row['work_days']}일")

    conn.close()

def export_summary_to_csv():
    """주요 분석 결과를 CSV로 내보내기"""
    conn = connect_db()

    # 부서별 통계
    dept_query = """
        SELECT department,
               period,
               COUNT(DISTINCT user_id) as user_count,
               COUNT(*) as activity_count,
               COUNT(DISTINCT work_date) as work_days
        FROM work_history
        GROUP BY department, period
        ORDER BY department, period
    """
    df_dept = pd.read_sql_query(dept_query, conn)

    # 년도와 월 분리
    df_dept['년'] = df_dept['period'].str[:4].astype(int)
    df_dept['월'] = df_dept['period'].str[5:7].astype(int)

    # period 컬럼 제거하고 컬럼 순서 재배열
    df_dept = df_dept[['department', '년', '월', 'user_count', 'activity_count', 'work_days']]

    # 컬럼명 한글로 변경
    df_dept.columns = ['소속팀', '년', '월', '인원수', '활동횟수', '근무일수']

    df_dept.to_csv('../data/department_summary.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 부서별 통계 저장: ../data/department_summary.csv")

    # 사용자별 통계
    user_query = """
        SELECT user_id, user_name, user_position, department,
               period,
               COUNT(*) as activity_count,
               COUNT(DISTINCT work_date) as work_days,
               COUNT(DISTINCT CASE WHEN weekday IN ('토요일', '일요일') THEN work_date END) as weekend_days
        FROM work_history
        GROUP BY user_id, user_name, user_position, department, period
        ORDER BY user_id, period
    """
    df_user = pd.read_sql_query(user_query, conn)

    # 년도와 월 분리
    df_user['년'] = df_user['period'].str[:4].astype(int)
    df_user['월'] = df_user['period'].str[5:7].astype(int)

    # period 컬럼 제거하고 컬럼 순서 재배열
    df_user = df_user[['user_id', 'user_name', 'user_position', 'department', '년', '월',
                       'activity_count', 'work_days', 'weekend_days']]

    # 컬럼명 한글로 변경
    df_user.columns = ['사번', '성명', '직위', '소속팀', '년', '월', '활동횟수', '근무일수', '주말근무일수']

    df_user.to_csv('../data/user_summary.csv', index=False, encoding='utf-8-sig')
    print("✅ 사용자별 통계 저장: ../data/user_summary.csv")

    conn.close()

def main():
    """메인 분석 실행"""
    print("\n🔍 근무 이력 데이터 분석 시작...")
    print("="*60)

    # 기본 통계
    print_basic_stats()

    # 근무 패턴 분석
    analyze_work_patterns()

    # 부서별 분석
    analyze_department_activity()

    # 개인별 분석
    analyze_user_activity()

    # CSV 내보내기
    print("\n" + "="*60)
    print("💾 분석 결과 내보내기")
    print("="*60)
    export_summary_to_csv()

    print("\n✨ 분석 완료!")
    print("\n💡 추가 분석을 원하시면 analyze_work_data.py 파일을 수정하거나")
    print("   SQLite 데이터베이스 '../data/work_history.db'를 직접 쿼리하실 수 있습니다.")

if __name__ == "__main__":
    main()