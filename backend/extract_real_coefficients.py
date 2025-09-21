#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sqlite3

def extract_real_coefficients():
    """엑셀 파일에서 실제 회귀 계수를 추출해서 DB에 저장"""

    conn = sqlite3.connect('/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db')
    cursor = conn.cursor()

    print("=== 엑셀에서 실제 회귀 계수 추출 ===")

    # 엑셀 파일 읽기
    file_path = '/Users/hanskim/Projects/Hwaseung/data/FTE계산_2025.0.21.xlsx'

    # 각 팀별 시트와 회귀 계수 위치 정의
    teams_config = {
        '시스템개발 운영팀': {
            'sheet': 'Sheet3',
            'positions': {
                '총': {'row': 80, 'col': 'M'},     # M80
                '책임': {'row': 80, 'col': 'N'},   # N80
                '선임': {'row': 80, 'col': 'O'},   # O80
                '사원': {'row': 80, 'col': 'P'}    # P80
            }
        },
        'SPECIALTY개발팀': {
            'sheet': 'Sheet4',
            'positions': {
                '총': {'row': 80, 'col': 'M'},
                '책임': {'row': 80, 'col': 'N'},
                '선임': {'row': 80, 'col': 'O'},
                '사원': {'row': 80, 'col': 'P'}
            }
        },
        'SL운영팀': {
            'sheet': 'Sheet5',
            'positions': {
                '총': {'row': 80, 'col': 'M'},
                '책임': {'row': 80, 'col': 'N'},
                '선임': {'row': 80, 'col': 'O'},
                '사원': {'row': 80, 'col': 'P'}
            }
        },
        '관체사업팀': {
            'sheet': 'Sheet6',
            'positions': {
                '총': {'row': 80, 'col': 'M'},
                '책임': {'row': 80, 'col': 'N'},
                '선임': {'row': 80, 'col': 'O'},
                '사원': {'row': 80, 'col': 'P'}
            }
        }
    }

    # 각 팀별로 처리
    for team_name, config in teams_config.items():
        print(f"\n{team_name} 회귀 계수 추출 중...")

        try:
            # 시트 읽기
            df = pd.read_excel(file_path, sheet_name=config['sheet'], header=None)

            for position, pos_config in config['positions'].items():
                row_idx = pos_config['row'] - 1  # 0-based index
                col = pos_config['col']

                # 모델 ID 찾기
                cursor.execute('''
                    SELECT id FROM regression_models
                    WHERE org_name = ? AND model_type = ?
                ''', (team_name, position))

                model_result = cursor.fetchone()
                if not model_result:
                    print(f"  {position}: 모델을 찾을 수 없음")
                    continue

                model_id = model_result[0]

                # Y절편 (intercept) 추출
                intercept_value = df.iloc[row_idx, ord(col) - ord('A')]
                if pd.notna(intercept_value):
                    cursor.execute('''
                        UPDATE regression_parameters
                        SET coefficient = ?
                        WHERE model_id = ? AND parameter_name = 'intercept'
                    ''', (float(intercept_value), model_id))
                    print(f"  {position}: intercept = {intercept_value}")

                # X1~X9 계수 추출 (M81~M89, N81~N89 등)
                feature_names = [
                    'IT SW/HW 구매',
                    '기타 지원',
                    '네트워크/보안 지원',
                    '데이터 수정/변경',
                    '시스템 구축',
                    '시스템 권한',
                    '시스템 트러블슈팅',
                    '프로그램 수정/개발',
                    'FLOW 로그인 수 (총)'
                ]

                for i, feature_name in enumerate(feature_names):
                    coeff_row_idx = row_idx + 1 + i  # M81, M82, ... M89
                    if coeff_row_idx < len(df):
                        coeff_value = df.iloc[coeff_row_idx, ord(col) - ord('A')]
                        if pd.notna(coeff_value):
                            cursor.execute('''
                                UPDATE regression_parameters
                                SET coefficient = ?
                                WHERE model_id = ? AND parameter_name = ?
                            ''', (float(coeff_value), model_id, feature_name))
                            print(f"    {feature_name}: {coeff_value}")

                # 관체사업팀의 경우 추가 feature들
                if team_name == '관체사업팀':
                    additional_features = [
                        'LINE별 설비CAPA분석 (반기, 중장기)',
                        ' 저압 / 고압 / 외주 공정지시 (ERP 업로드)',
                        ' 후가공 KD / 수출 제품 납입율 점검',
                        ' 생산계획대비 실적 분석 (생산금액, 생산수량)',
                        ' 월간 생산실적 분석 (인건비, 생산량, CAPA, 재료비, 경비 , 생산지표)',
                        ' 주요지표 일일정산(결원율, 라인가동, 설비종합효율, 수율 등)',
                        ' 생산 공정, 설비, 자재 등 양산 공정 현장 순회 점검',
                        ' 현장 공정 개선 업무 지도 ( 생산성,수율,품질,가동율 등)'
                    ]

                    # X10~X17까지 계속 처리 (M90~M97)
                    for i, feature_name in enumerate(additional_features):
                        coeff_row_idx = row_idx + 10 + i  # M90, M91, ... M97
                        if coeff_row_idx < len(df):
                            coeff_value = df.iloc[coeff_row_idx, ord(col) - ord('A')]
                            if pd.notna(coeff_value):
                                cursor.execute('''
                                    UPDATE regression_parameters
                                    SET coefficient = ?
                                    WHERE model_id = ? AND parameter_name = ?
                                ''', (float(coeff_value), model_id, feature_name))
                                print(f"    {feature_name}: {coeff_value}")

        except Exception as e:
            print(f"  {team_name} 처리 중 오류: {e}")

    conn.commit()
    conn.close()
    print("\n실제 회귀 계수 추출 완료!")

if __name__ == "__main__":
    extract_real_coefficients()