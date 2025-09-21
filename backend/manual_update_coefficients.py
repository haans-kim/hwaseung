#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3

def manual_update_coefficients():
    """스크린샷에서 확인한 회귀 계수를 수동으로 입력"""

    conn = sqlite3.connect('/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db')
    cursor = conn.cursor()

    print("=== 스크린샷 계수 수동 입력 ===")

    # 스크린샷에서 확인한 계수들
    # Y 절편: 19.5046
    # X1: 0
    # X2: 0.02412
    # X3: 0
    # X4: 0
    # X5: 0
    # X6: 0
    # X7: 0
    # X8: 0
    # X9: 0.00544

    screenshot_coefficients = {
        'intercept': 19.5046,
        'IT SW/HW 구매': 0,           # X1
        '기타 지원': 0.02412,         # X2
        '네트워크/보안 지원': 0,      # X3
        '데이터 수정/변경': 0,        # X4
        '시스템 구축': 0,             # X5
        '시스템 권한': 0,             # X6
        '시스템 트러블슈팅': 0,       # X7
        '프로그램 수정/개발': 0,      # X8
        'FLOW 로그인 수 (총)': 0.00544  # X9
    }

    # 어느 팀의 어떤 모델인지 추측 - 스크린샷 위치상 시스템개발 운영팀의 총 모델로 보임
    team_name = '시스템개발 운영팀'
    model_type = '총'

    print(f"{team_name} {model_type} 모델 계수 업데이트:")

    # 모델 ID 찾기
    cursor.execute('''
        SELECT id FROM regression_models
        WHERE org_name = ? AND model_type = ?
    ''', (team_name, model_type))

    model_result = cursor.fetchone()
    if model_result:
        model_id = model_result[0]

        # 계수들 업데이트
        for param_name, coefficient in screenshot_coefficients.items():
            cursor.execute('''
                UPDATE regression_parameters
                SET coefficient = ?
                WHERE model_id = ? AND parameter_name = ?
            ''', (coefficient, model_id, param_name))
            print(f"  {param_name}: {coefficient}")

        print(f"\n계수 업데이트 완료!")

        # 예상 계산 확인
        base_metrics = {
            'IT SW/HW 구매': 50,
            '기타 지원': 30,
            '네트워크/보안 지원': 40,
            '데이터 수정/변경': 35,
            '시스템 구축': 45,
            '시스템 권한': 25,
            '시스템 트러블슈팅': 55,
            '프로그램 수정/개발': 40,
            'FLOW 로그인 수 (총)': 1000
        }

        predicted_value = screenshot_coefficients['intercept']
        for param_name, coefficient in screenshot_coefficients.items():
            if param_name != 'intercept' and param_name in base_metrics:
                contribution = coefficient * base_metrics[param_name]
                predicted_value += contribution
                if contribution != 0:
                    print(f"  {param_name}: {coefficient} * {base_metrics[param_name]} = {contribution}")

        print(f"\n기본 메트릭으로 예상값: {predicted_value:.1f}")

    else:
        print(f"오류: {team_name} {model_type} 모델을 찾을 수 없음")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    manual_update_coefficients()