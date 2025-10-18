#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

def check_sheets():
    """엑셀 파일의 시트 이름 확인"""

    file_path = str(Path(__file__).parent.parent / 'data/FTE계산_2025.0.21.xlsx'

    # 모든 시트 이름 확인
    xl_file = pd.ExcelFile(file_path)
    print("=== 엑셀 파일 시트 목록 ===")
    for i, sheet_name in enumerate(xl_file.sheet_names):
        print(f"{i+1}. {sheet_name}")

    # 각 시트의 데이터 구조 확인
    print("\n=== 각 시트 데이터 구조 ===")
    for sheet_name in xl_file.sheet_names[:6]:  # 처음 6개 시트만
        print(f"\n[{sheet_name}]")
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
            print(f"크기: {df.shape[0]}행 x {df.shape[1]}열")

            # 80~90행 정도의 M~P열 데이터 확인
            if df.shape[0] > 80 and df.shape[1] > 15:
                print("80~85행, M~P열 데이터:")
                for row in range(79, min(85, df.shape[0])):
                    row_data = []
                    for col in range(12, min(16, df.shape[1])):  # M(12)~P(15)
                        val = df.iloc[row, col]
                        row_data.append(str(val) if pd.notna(val) else 'NaN')
                    print(f"  행{row+1}: {row_data}")
        except Exception as e:
            print(f"  오류: {e}")

if __name__ == "__main__":
    check_sheets()