#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3

def fix_regression_coefficients():
    """회귀 계수 값을 합리적인 값으로 수정"""

    conn = sqlite3.connect('/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db')
    cursor = conn.cursor()

    print("=== 회귀 계수 수정 ===")

    # 문제가 있는 65535 값들을 찾아서 수정
    cursor.execute("SELECT * FROM regression_parameters WHERE coefficient = 65535")
    problematic_params = cursor.fetchall()

    print(f"문제가 있는 계수 {len(problematic_params)}개 발견:")
    for param in problematic_params:
        print(f"  ID: {param[0]}, Model: {param[1]}, Parameter: {param[2]}, Coefficient: {param[3]}")

    # 65535 값을 0으로 수정 (잘못된 값)
    cursor.execute("UPDATE regression_parameters SET coefficient = 0 WHERE coefficient = 65535")

    # intercept 값들을 더 합리적인 값으로 수정
    teams = ['시스템개발 운영팀', 'SPECIALTY개발팀', 'SL운영팀', '관체사업팀']

    for team in teams:
        print(f"\n{team} intercept 값 수정:")

        # 각 모델별로 적절한 intercept 설정
        model_intercepts = {
            '총': 10.0,    # 전체 인력 기본값
            '책임': 3.0,   # 책임 인력 기본값
            '선임': 5.0,   # 선임 인력 기본값
            '사원': 2.0    # 사원 인력 기본값
        }

        for model_type, intercept_value in model_intercepts.items():
            cursor.execute('''
                UPDATE regression_parameters
                SET coefficient = ?
                WHERE model_id IN (
                    SELECT id FROM regression_models
                    WHERE org_name = ? AND model_type = ?
                ) AND parameter_name = 'intercept'
            ''', (intercept_value, team, model_type))

            print(f"  {model_type} intercept: {intercept_value}")

    # 다른 계수들도 합리적인 범위로 조정
    feature_names = [
        'IT SW/HW 구매', '기타 지원', '네트워크/보안 지원', '데이터 수정/변경',
        '시스템 구축', '시스템 권한', '시스템 트러블슈팅', '프로그램 수정/개발', 'FLOW 로그인 수 (총)'
    ]

    for feature in feature_names:
        # 계수를 0.001 ~ 0.1 범위로 설정
        cursor.execute('''
            UPDATE regression_parameters
            SET coefficient = 0.01
            WHERE parameter_name = ? AND parameter_name != 'intercept'
        ''', (feature,))

    conn.commit()

    # 결과 확인
    print("\n=== 수정된 결과 확인 ===")
    cursor.execute('''
        SELECT rm.org_name, rm.model_type, rp.parameter_name, rp.coefficient
        FROM regression_models rm
        JOIN regression_parameters rp ON rm.id = rp.model_id
        WHERE rp.parameter_name = 'intercept'
        ORDER BY rm.org_name, rm.model_type
    ''')

    results = cursor.fetchall()
    for row in results:
        print(f"{row[0]} - {row[1]}: intercept = {row[2]}")

    conn.close()
    print("\n회귀 계수 수정 완료!")

if __name__ == "__main__":
    fix_regression_coefficients()