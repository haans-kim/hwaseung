#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3
import os
from datetime import datetime

def standardize_column_names(df):
    """표준화된 컬럼명으로 변경"""
    # 컬럼명 매핑 (실제 데이터의 컬럼명을 표준 형태로)
    column_mapping = {
        '사용자ID': 'user_id',
        '사용자': 'user_name',
        '시스템명': 'system_name',
        'mid': 'mid',
        '사용자직위': 'user_position',
        '소속그룹': 'department',
        '근무구분': 'work_type',
        '항목구분': 'item_type',
        '년월일': 'work_date',
        '요일': 'weekday',
        '근무기준시간': 'standard_work_hours',
        '1차근태_상태': 'attendance_status_1',
        '1차근태_출근시각': 'check_in_time_1',
        '1차근태_퇴근시각': 'check_out_time_1',
        '2차근태_상태': 'attendance_status_2',
        '2차근태_출근시각': 'check_in_time_2',
        '2차근태_퇴근시각': 'check_out_time_2'
    }

    # 컬럼명 변경
    new_columns = {}
    for col in df.columns:
        col_clean = col.strip()
        if col_clean in column_mapping:
            new_columns[col] = column_mapping[col_clean]
        else:
            # 매핑되지 않은 컬럼은 한글과 특수문자를 처리하여 영어로 변환
            new_col = col.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            # 한글이 포함된 컬럼명은 그대로 유지하되 공백만 언더스코어로 변경
            if any(ord(char) > 127 for char in col):
                new_col = col.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
            new_columns[col] = new_col

    df = df.rename(columns=new_columns)
    return df

def main():
    # 데이터 파일 경로
    data_dir = '../data'
    db_path = '../data/work_history.db'

    # 파일 목록
    excel_files = [
        '25년 06월 근무이력.xlsx',
        '25년 07월 근무이력.xlsx',
        '25년 08월 근무이력.xlsx'
    ]

    # SQLite3 연결
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    print("SQLite3 데이터베이스 생성 중...")

    all_data = []

    for idx, file in enumerate(excel_files):
        file_path = os.path.join(data_dir, file)
        print(f"\n파일 읽는 중: {file}")

        # 월 정보 추출
        month = file.split('년')[1].split('월')[0].strip()
        year = '20' + file.split('년')[0].strip()
        period = f"{year}-{month.zfill(2)}"

        try:
            # Excel 파일 읽기
            df = pd.read_excel(file_path, engine='openpyxl')
            print(f"  - 원본 행 수: {len(df)}")
            print(f"  - 원본 컬럼: {list(df.columns)[:10]}...")  # 첫 10개 컬럼만 표시

            # 컬럼명 표준화
            df = standardize_column_names(df)

            # 기간 컬럼 추가
            df['period'] = period
            df['year_month'] = period

            # 데이터 타입 정리
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.strip()

            print(f"  - 표준화된 컬럼: {list(df.columns)[:10]}...")

            all_data.append(df)

        except Exception as e:
            print(f"  오류 발생: {str(e)}")
            continue

    if all_data:
        # 모든 데이터 합치기
        print("\n모든 데이터 합치는 중...")
        combined_df = pd.concat(all_data, ignore_index=True)

        print(f"전체 데이터: {len(combined_df)} 행, {len(combined_df.columns)} 컬럼")

        # SQLite에 저장
        print("\nSQLite3 데이터베이스에 저장 중...")
        combined_df.to_sql('work_history', conn, if_exists='replace', index=False)

        # 인덱스 생성
        print("인덱스 생성 중...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON work_history(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_period ON work_history(period)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_department ON work_history(department)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_work_date ON work_history(work_date)")

        # 통계 정보
        cursor.execute("SELECT COUNT(DISTINCT user_id) as unique_users FROM work_history")
        unique_users = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT period) as unique_periods FROM work_history")
        unique_periods = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT department) as unique_departments FROM work_history")
        unique_departments = cursor.fetchone()[0]

        print(f"\n=== 데이터베이스 생성 완료 ===")
        print(f"데이터베이스 경로: {os.path.abspath(db_path)}")
        print(f"총 레코드 수: {len(combined_df)}")
        print(f"고유 사용자 수: {unique_users}")
        print(f"기간 수: {unique_periods}")
        print(f"부서 수: {unique_departments}")

        # 샘플 데이터 확인
        print(f"\n=== 샘플 데이터 (처음 5행) ===")
        cursor.execute("SELECT * FROM work_history LIMIT 5")
        columns = [description[0] for description in cursor.description]
        print(f"컬럼: {columns}")

        for row in cursor.fetchall():
            print(dict(zip(columns, row)))

    else:
        print("읽을 수 있는 데이터가 없습니다.")

    # 연결 종료
    conn.commit()
    conn.close()
    print("\n완료!")

if __name__ == "__main__":
    main()