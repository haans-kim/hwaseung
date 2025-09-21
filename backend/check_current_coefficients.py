#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3

def check_current_coefficients():
    """현재 DB에 저장된 회귀 계수 확인"""

    conn = sqlite3.connect('/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db')
    cursor = conn.cursor()

    print("=== 현재 저장된 회귀 계수 ===")

    # 각 팀별로 확인
    teams = ['시스템개발 운영팀', 'SPECIALTY개발팀', 'SL운영팀', '관체사업팀']

    for team in teams:
        print(f"\n[{team}]")

        # 팀의 모든 모델과 계수 가져오기
        cursor.execute('''
            SELECT rm.model_type, rp.parameter_name, rp.coefficient
            FROM regression_models rm
            JOIN regression_parameters rp ON rm.id = rp.model_id
            WHERE rm.org_name = ?
            ORDER BY rm.model_type, rp.parameter_name
        ''', (team,))

        results = cursor.fetchall()
        current_model = None

        for model_type, param_name, coefficient in results:
            if model_type != current_model:
                print(f"  {model_type}:")
                current_model = model_type

            print(f"    {param_name}: {coefficient}")

    # 전체 계수 개수 확인
    cursor.execute('SELECT COUNT(*) FROM regression_parameters')
    total_params = cursor.fetchone()[0]
    print(f"\n총 회귀 계수 개수: {total_params}")

    # 65535 값이 있는지 확인
    cursor.execute('SELECT COUNT(*) FROM regression_parameters WHERE coefficient = 65535')
    invalid_count = cursor.fetchone()[0]
    print(f"65535 값 개수: {invalid_count}")

    # 0 값이 있는지 확인
    cursor.execute('SELECT COUNT(*) FROM regression_parameters WHERE coefficient = 0')
    zero_count = cursor.fetchone()[0]
    print(f"0 값 개수: {zero_count}")

    conn.close()

if __name__ == "__main__":
    check_current_coefficients()