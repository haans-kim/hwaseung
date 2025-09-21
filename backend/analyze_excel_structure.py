#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def analyze_excel_structure():
    """엑셀 파일의 구조를 자세히 분석"""

    file_path = '/Users/hanskim/Projects/Hwaseung/data/FTE계산_2025.0.21.xlsx'

    # FTE 계산 시트 분석
    print("=== FTE 계산 시트 구조 분석 ===")
    df = pd.read_excel(file_path, sheet_name='FTE 계산', header=None)
    print(f"크기: {df.shape[0]}행 x {df.shape[1]}열")

    # 각 팀별로 위치 찾기
    teams = {
        '시스템개발 운영팀': [],
        'SPECIALTY개발팀': [],
        'SL운영팀': [],
        '관체사업팀': []
    }

    # 전체 시트에서 팀 이름이 있는 위치 찾기
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            cell_value = df.iloc[row, col]
            if pd.notna(cell_value) and isinstance(cell_value, str):
                for team_name in teams.keys():
                    if team_name in cell_value:
                        teams[team_name].append((row+1, chr(65+col), cell_value))

    # 각 팀별 위치 출력
    for team_name, positions in teams.items():
        print(f"\n[{team_name}] 위치:")
        for row, col, value in positions:
            print(f"  행{row}, 열{col}: {value}")

    # 숫자 패턴 분석 - 회귀 계수일 가능성이 있는 영역
    print("\n=== 숫자 패턴 분석 ===")
    for team_name, positions in teams.items():
        if positions:
            print(f"\n[{team_name}] 주변 숫자 패턴:")
            # 첫 번째 위치 근처의 숫자들 확인
            first_pos = positions[0]
            start_row = first_pos[0] - 1  # 0-based

            # 해당 행부터 +50행 정도까지 숫자 확인
            for check_row in range(start_row, min(start_row + 50, df.shape[0])):
                numeric_cells = []
                for col in range(min(20, df.shape[1])):  # A~T열까지
                    cell_value = df.iloc[check_row, col]
                    if pd.notna(cell_value) and isinstance(cell_value, (int, float)) and cell_value != 0:
                        numeric_cells.append(f"{chr(65+col)}{check_row+1}:{cell_value}")

                if len(numeric_cells) > 2:  # 숫자가 3개 이상 있는 행
                    print(f"  행{check_row+1}: {numeric_cells[:10]}")  # 처음 10개만

    # 특정 패턴 찾기 - "회귀", "계수", "절편" 키워드
    print("\n=== 회귀 관련 키워드 찾기 ===")
    keywords = ['회귀', '계수', '절편', 'coefficient', 'intercept', '상수', 'R²', 'R-squared']

    for keyword in keywords:
        print(f"\n'{keyword}' 포함 셀:")
        found = False
        for row in range(df.shape[0]):
            for col in range(df.shape[1]):
                cell_value = df.iloc[row, col]
                if pd.notna(cell_value) and isinstance(cell_value, str) and keyword.lower() in str(cell_value).lower():
                    print(f"  행{row+1}, 열{chr(65+col)}: {cell_value}")
                    found = True
        if not found:
            print(f"  '{keyword}' 키워드를 찾을 수 없음")

if __name__ == "__main__":
    analyze_excel_structure()