#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def find_coefficients():
    """엑셀 파일에서 회귀 계수 위치 찾기"""

    file_path = '/Users/hanskim/Projects/Hwaseung/data/FTE계산_2025.0.21.xlsx'

    # FTE 계산 시트에서 회귀 관련 데이터 찾기
    print("=== FTE 계산 시트 분석 ===")
    df = pd.read_excel(file_path, sheet_name='FTE 계산', header=None)
    print(f"시트 크기: {df.shape[0]}행 x {df.shape[1]}열")

    # "회귀" 키워드가 있는 행 찾기
    print("\n'회귀' 키워드가 포함된 셀:")
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            cell_value = df.iloc[row, col]
            if pd.notna(cell_value) and isinstance(cell_value, str) and '회귀' in cell_value:
                print(f"  행{row+1}, 열{chr(65+col)}: {cell_value}")

    # "계수" 키워드가 있는 행 찾기
    print("\n'계수' 키워드가 포함된 셀:")
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            cell_value = df.iloc[row, col]
            if pd.notna(cell_value) and isinstance(cell_value, str) and '계수' in cell_value:
                print(f"  행{row+1}, 열{chr(65+col)}: {cell_value}")

    # 팀 이름이 있는 행 찾기
    print("\n팀 이름이 포함된 셀:")
    team_names = ['시스템개발 운영팀', 'SPECIALTY개발팀', 'SL운영팀', '관체사업팀']
    for team in team_names:
        print(f"\n[{team}]:")
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                cell_value = df.iloc[row, col]
                if pd.notna(cell_value) and isinstance(cell_value, str) and team in cell_value:
                    print(f"  행{row+1}, 열{chr(65+col)}: {cell_value}")

    # 숫자 데이터가 많은 영역 확인 (회귀 계수일 가능성)
    print("\n숫자 데이터가 집중된 영역:")
    for start_row in range(0, df.shape[0], 50):
        end_row = min(start_row + 50, df.shape[0])
        numeric_count = 0
        for row in range(start_row, end_row):
            for col in range(df.shape[1]):
                cell_value = df.iloc[row, col]
                if pd.notna(cell_value) and isinstance(cell_value, (int, float)):
                    numeric_count += 1
        if numeric_count > 100:  # 숫자가 많은 영역
            print(f"  행{start_row+1}~{end_row}: {numeric_count}개의 숫자")

    # Sheet3도 확인
    print("\n=== Sheet3 분석 ===")
    df3 = pd.read_excel(file_path, sheet_name='Sheet3', header=None)
    print(f"시트 크기: {df3.shape[0]}행 x {df3.shape[1]}열")

    # 전체 데이터 샘플 확인
    print("\n전체 데이터 샘플 (처음 20행):")
    for row in range(min(20, df3.shape[0])):
        row_data = []
        for col in range(min(10, df3.shape[1])):
            val = df3.iloc[row, col]
            if pd.notna(val):
                if isinstance(val, str):
                    row_data.append(val[:20])  # 문자열은 20자까지만
                else:
                    row_data.append(str(val))
            else:
                row_data.append('')
        print(f"  행{row+1}: {row_data}")

if __name__ == "__main__":
    find_coefficients()