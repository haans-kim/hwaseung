#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def connect_db():
    """데이터베이스 연결"""
    db_path = '../data/work_history.db'
    return sqlite3.connect(db_path)

def calculate_daily_work_hours():
    """일별 근무시간 계산"""
    conn = connect_db()

    print("일별 근무시간 계산 중...")

    # 각 사용자의 일별 첫 활동과 마지막 활동 시간 찾기
    query = """
    WITH daily_activities AS (
        SELECT
            user_id,
            user_name,
            department,
            user_position,
            work_date,
            period,
            weekday,
            MIN(CASE
                WHEN 시작시각 IS NOT NULL AND 시작시각 != 'None'
                THEN 시작시각
                ELSE NULL
            END) as first_activity,
            MAX(CASE
                WHEN 종료시각 IS NOT NULL AND 종료시각 != 'None'
                THEN 종료시각
                WHEN 시작시각 IS NOT NULL AND 시작시각 != 'None'
                THEN 시작시각
                ELSE NULL
            END) as last_activity,
            -- 초과근무 시간 집계
            SUM(CASE
                WHEN 항목 = '초과' AND 사용시간 IS NOT NULL
                THEN CAST(사용시간 AS INTEGER)
                ELSE 0
            END) as overtime_minutes,
            -- 시스템 ON/OFF 체크
            MAX(CASE WHEN 항목 = '시스템 ON' THEN 1 ELSE 0 END) as has_system_on,
            MAX(CASE WHEN 항목 = '시스템 OFF' THEN 1 ELSE 0 END) as has_system_off,
            -- 출퇴근 체크
            MAX(CASE WHEN 항목 IN ('출근', '출근버튼 클릭') THEN 1 ELSE 0 END) as has_checkin,
            MAX(CASE WHEN 항목 IN ('퇴근', '퇴근버튼 클릭') THEN 1 ELSE 0 END) as has_checkout
        FROM work_history
        WHERE work_date IS NOT NULL
        GROUP BY user_id, user_name, department, user_position, work_date, period, weekday
    )
    SELECT
        user_id,
        user_name,
        department,
        user_position,
        work_date,
        period,
        weekday,
        first_activity,
        last_activity,
        overtime_minutes,
        has_system_on,
        has_system_off,
        has_checkin,
        has_checkout,
        -- 근무시간 계산 (분 단위)
        CASE
            WHEN first_activity IS NOT NULL AND last_activity IS NOT NULL
                 AND first_activity != last_activity
            THEN
                CAST((julianday(last_activity) - julianday(first_activity)) * 24 * 60 AS INTEGER)
            WHEN has_checkin = 1 OR has_system_on = 1
            THEN 480  -- 기본 8시간 (480분)
            ELSE 0
        END as work_minutes
    FROM daily_activities
    """

    df_daily = pd.read_sql_query(query, conn)

    # 비정상적인 값 처리 (24시간 이상 또는 음수)
    df_daily['work_minutes'] = df_daily['work_minutes'].apply(
        lambda x: min(max(x, 0), 1440)  # 0~24시간(1440분) 범위로 제한
    )

    # 주말 구분
    df_daily['is_weekend'] = df_daily['weekday'].isin(['토요일', '일요일'])

    conn.close()
    return df_daily

def aggregate_monthly_hours(df_daily):
    """월별 집계"""
    print("월별 집계 중...")

    # 년도와 월 분리
    df_daily['년'] = df_daily['period'].str[:4].astype(int)
    df_daily['월'] = df_daily['period'].str[5:7].astype(int)

    # 월별 집계
    monthly_summary = df_daily.groupby(['user_id', 'user_name', 'department', 'user_position', '년', '월']).agg({
        'work_date': 'count',  # 근무일수
        'work_minutes': 'sum',  # 총 근무시간(분)
        'overtime_minutes': 'sum',  # 총 초과근무시간(분)
        'is_weekend': 'sum'  # 주말 근무일수
    }).reset_index()

    # 컬럼명 변경
    monthly_summary.columns = [
        '사번', '성명', '소속팀', '직위', '년', '월',
        '근무일수', '총근무시간_분', '초과근무시간_분', '주말근무일수'
    ]

    # 시간 단위로 변환 및 반올림
    monthly_summary['총근무시간'] = (monthly_summary['총근무시간_분'] / 60).round(1)
    monthly_summary['초과근무시간'] = (monthly_summary['초과근무시간_분'] / 60).round(1)
    monthly_summary['일평균근무시간'] = (monthly_summary['총근무시간'] / monthly_summary['근무일수']).round(1)

    # 정규근무시간 계산 (평일 8시간 기준)
    monthly_summary['예상정규근무시간'] = (monthly_summary['근무일수'] - monthly_summary['주말근무일수']) * 8
    monthly_summary['근무시간차이'] = monthly_summary['총근무시간'] - monthly_summary['예상정규근무시간']

    return monthly_summary

def save_results(df_daily, monthly_summary):
    """결과 저장"""
    print("결과 저장 중...")

    # 일별 상세 데이터 저장
    df_daily.to_csv('../data/daily_work_hours.csv', index=False, encoding='utf-8-sig')
    print("✅ 일별 근무시간 저장: ../data/daily_work_hours.csv")

    # 월별 집계 데이터 저장
    monthly_summary.to_csv('../data/monthly_work_hours.csv', index=False, encoding='utf-8-sig')
    print("✅ 월별 근무시간 저장: ../data/monthly_work_hours.csv")

    # 엑셀 파일로도 저장
    with pd.ExcelWriter('../data/work_hours_analysis.xlsx', engine='openpyxl') as writer:
        monthly_summary.to_excel(writer, sheet_name='월별집계', index=False)

        # 부서별 평균 추가
        dept_summary = monthly_summary.groupby(['소속팀', '년', '월']).agg({
            '근무일수': 'mean',
            '총근무시간': 'mean',
            '초과근무시간': 'mean',
            '일평균근무시간': 'mean'
        }).round(1).reset_index()
        dept_summary.to_excel(writer, sheet_name='부서별평균', index=False)

    print("✅ 엑셀 파일 저장: ../data/work_hours_analysis.xlsx")

def print_summary(monthly_summary):
    """요약 통계 출력"""
    print("\n" + "="*60)
    print("📊 근무시간 집계 결과")
    print("="*60)

    # 전체 평균
    print("\n📌 전체 평균 (월 기준):")
    print(f"  - 평균 근무일수: {monthly_summary['근무일수'].mean():.1f}일")
    print(f"  - 평균 총근무시간: {monthly_summary['총근무시간'].mean():.1f}시간")
    print(f"  - 평균 초과근무시간: {monthly_summary['초과근무시간'].mean():.1f}시간")
    print(f"  - 평균 일평균근무시간: {monthly_summary['일평균근무시간'].mean():.1f}시간")

    # 월별 통계
    print("\n📅 월별 평균:")
    monthly_avg = monthly_summary.groupby(['년', '월']).agg({
        '근무일수': 'mean',
        '총근무시간': 'mean',
        '초과근무시간': 'mean',
        '일평균근무시간': 'mean'
    }).round(1)

    for (year, month) in monthly_avg.index:
        row = monthly_avg.loc[(year, month)]
        print(f"\n  {year}년 {month}월:")
        print(f"    - 평균 근무일수: {row['근무일수']:.1f}일")
        print(f"    - 평균 총근무시간: {row['총근무시간']:.1f}시간")
        print(f"    - 평균 초과근무시간: {row['초과근무시간']:.1f}시간")
        print(f"    - 일평균근무시간: {row['일평균근무시간']:.1f}시간")

    # 초과근무 TOP 10
    print("\n🔥 월평균 초과근무시간 TOP 10:")
    top_overtime = monthly_summary.groupby(['사번', '성명', '소속팀', '직위']).agg({
        '초과근무시간': 'mean'
    }).round(1).sort_values('초과근무시간', ascending=False).head(10)

    for i, (idx, row) in enumerate(top_overtime.iterrows(), 1):
        print(f"  {i}. {idx[1]} ({idx[3]}, {idx[2]}): {row['초과근무시간']:.1f}시간/월")

    # 부서별 평균 근무시간
    print("\n🏢 부서별 평균 근무시간 (월 기준, 상위 10개):")
    dept_avg = monthly_summary.groupby('소속팀').agg({
        '총근무시간': 'mean',
        '사번': 'count'
    }).round(1).sort_values('총근무시간', ascending=False).head(10)

    for dept in dept_avg.index:
        row = dept_avg.loc[dept]
        print(f"  - {dept}: {row['총근무시간']:.1f}시간 (인원: {int(row['사번']/3)}명)")

def main():
    """메인 실행 함수"""
    print("🔍 근무시간 계산 시작...")
    print("="*60)

    # 일별 근무시간 계산
    df_daily = calculate_daily_work_hours()
    print(f"✅ 일별 데이터 처리 완료: {len(df_daily)}개 레코드")

    # 월별 집계
    monthly_summary = aggregate_monthly_hours(df_daily)
    print(f"✅ 월별 집계 완료: {len(monthly_summary)}개 레코드")

    # 결과 저장
    save_results(df_daily, monthly_summary)

    # 요약 출력
    print_summary(monthly_summary)

    print("\n✨ 근무시간 계산 완료!")
    print("\n💡 생성된 파일:")
    print("  - daily_work_hours.csv: 일별 상세 근무시간")
    print("  - monthly_work_hours.csv: 월별 집계 데이터")
    print("  - work_hours_analysis.xlsx: 엑셀 분석 파일 (월별, 부서별)")

if __name__ == "__main__":
    main()