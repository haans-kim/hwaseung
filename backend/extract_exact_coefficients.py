#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3

def extract_exact_coefficients():
    """스크린샷에서 확인한 정확한 회귀 계수를 추출해서 DB에 저장"""

    conn = sqlite3.connect('/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db')
    cursor = conn.cursor()

    print("=== 엑셀에서 정확한 회귀 계수 추출 ===")

    # 엑셀 파일 읽기
    file_path = '/Users/hanskim/Projects/Hwaseung/data/FTE계산_2025.0.21.xlsx'

    # FTE 계산 시트에서 회귀 계수 찾기
    df = pd.read_excel(file_path, sheet_name='FTE 계산', header=None)
    print(f"시트 크기: {df.shape[0]}행 x {df.shape[1]}열")

    # 회귀 분석 결과가 있는 영역 찾기
    print("\n회귀 계수 영역 탐색...")

    # Y 절편, X1~X9를 찾기 위해 전체 시트 스캔
    regression_sections = []

    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            cell_value = df.iloc[row, col]

            # "Y 절편" 텍스트를 찾으면 회귀 분석 섹션 시작
            if pd.notna(cell_value) and isinstance(cell_value, str) and 'Y 절편' in cell_value:
                print(f"  'Y 절편' 발견: 행{row+1}, 열{chr(65+col)}")

                # 해당 섹션의 계수들 추출
                coefficients = {}

                try:
                    # Y 절편 값 (한 칸 오른쪽)
                    y_intercept = df.iloc[row, col + 1]
                    if pd.notna(y_intercept) and isinstance(y_intercept, (int, float)):
                        coefficients['intercept'] = float(y_intercept)
                        print(f"    Y 절편: {y_intercept}")

                    # X1~X9 계수들 (아래 행들에서)
                    feature_names = [
                        'IT SW/HW 구매',           # X1
                        '기타 지원',               # X2
                        '네트워크/보안 지원',      # X3
                        '데이터 수정/변경',        # X4
                        '시스템 구축',             # X5
                        '시스템 권한',             # X6
                        '시스템 트러블슈팅',       # X7
                        '프로그램 수정/개발',      # X8
                        'FLOW 로그인 수 (총)'     # X9
                    ]

                    for i, feature_name in enumerate(feature_names):
                        x_row = row + 1 + i  # X1은 Y 절편 다음 행
                        if x_row < df.shape[0]:
                            # X 라벨 확인
                            x_label = df.iloc[x_row, col]
                            if pd.notna(x_label) and f'X {i+1}' in str(x_label):
                                # 계수 값 (한 칸 오른쪽)
                                x_coeff = df.iloc[x_row, col + 1]
                                if pd.notna(x_coeff) and isinstance(x_coeff, (int, float)):
                                    coefficients[feature_name] = float(x_coeff)
                                    print(f"    X{i+1} ({feature_name}): {x_coeff}")

                    # 어느 팀/직급의 계수인지 추정 (위쪽에서 팀 이름 찾기)
                    team_name = None
                    model_type = None

                    # 위쪽 50행 정도에서 팀 이름 찾기
                    teams = ['시스템개발 운영팀', 'SPECIALTY개발팀', 'SL운영팀', '관체사업팀']
                    for check_row in range(max(0, row - 50), row):
                        for check_col in range(df.shape[1]):
                            check_value = df.iloc[check_row, check_col]
                            if pd.notna(check_value) and isinstance(check_value, str):
                                for team in teams:
                                    if team in check_value:
                                        team_name = team
                                        break
                                if team_name:
                                    break
                        if team_name:
                            break

                    # 직급 추정 (주변에서 "전체", "책임", "선임", "사원" 찾기)
                    positions = ['총', '책임', '선임', '사원']
                    for check_row in range(max(0, row - 10), min(df.shape[0], row + 10)):
                        for check_col in range(max(0, col - 5), min(df.shape[1], col + 10)):
                            check_value = df.iloc[check_row, check_col]
                            if pd.notna(check_value) and isinstance(check_value, str):
                                if '전체' in check_value:
                                    model_type = '총'
                                    break
                                elif '책임' in check_value:
                                    model_type = '책임'
                                    break
                                elif '선임' in check_value:
                                    model_type = '선임'
                                    break
                                elif '사원' in check_value:
                                    model_type = '사원'
                                    break
                        if model_type:
                            break

                    # DB에 저장
                    if team_name and model_type and coefficients:
                        print(f"  → {team_name} {model_type} 모델로 저장")

                        # 모델 ID 찾기
                        cursor.execute('''
                            SELECT id FROM regression_models
                            WHERE org_name = ? AND model_type = ?
                        ''', (team_name, model_type))

                        model_result = cursor.fetchone()
                        if model_result:
                            model_id = model_result[0]

                            # 계수들 업데이트
                            for param_name, coefficient in coefficients.items():
                                cursor.execute('''
                                    UPDATE regression_parameters
                                    SET coefficient = ?
                                    WHERE model_id = ? AND parameter_name = ?
                                ''', (coefficient, model_id, param_name))

                            print(f"    저장 완료: {len(coefficients)}개 계수")
                        else:
                            print(f"    오류: {team_name} {model_type} 모델을 찾을 수 없음")
                    else:
                        print(f"    건너뜀: team={team_name}, model={model_type}, coeffs={len(coefficients)}")

                except Exception as e:
                    print(f"    계수 추출 오류: {e}")

    conn.commit()
    conn.close()
    print("\n정확한 회귀 계수 추출 완료!")

if __name__ == "__main__":
    extract_exact_coefficients()