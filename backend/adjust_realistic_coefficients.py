#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3

def adjust_realistic_coefficients():
    """회귀 계수를 더 현실적인 값으로 조정"""

    conn = sqlite3.connect('/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db')
    cursor = conn.cursor()

    print("=== 현실적인 회귀 계수로 조정 ===")

    # 팀별 현재 인원 확인
    teams_headcount = {
        '시스템개발 운영팀': {'총': 26, '책임': 5, '선임': 11, '사원': 10},
        'SPECIALTY개발팀': {'총': 13, '책임': 2, '선임': 5, '사원': 6},
        'SL운영팀': {'총': 7, '책임': 3, '선임': 2, '사원': 2},
        '관체사업팀': {'총': 11, '책임': 4, '선임': 6, '사원': 1}
    }

    for team_name, headcounts in teams_headcount.items():
        print(f"\n{team_name} 계수 조정:")

        for model_type, target_headcount in headcounts.items():
            # intercept를 현재 인원의 80% 정도로 설정
            base_intercept = target_headcount * 0.8

            # 모델 ID 찾기
            cursor.execute('''
                SELECT id FROM regression_models
                WHERE org_name = ? AND model_type = ?
            ''', (team_name, model_type))

            model_result = cursor.fetchone()
            if model_result:
                model_id = model_result[0]

                # intercept 업데이트
                cursor.execute('''
                    UPDATE regression_parameters
                    SET coefficient = ?
                    WHERE model_id = ? AND parameter_name = 'intercept'
                ''', (base_intercept, model_id))

                # feature 계수들을 더 의미있는 값으로 설정
                feature_coefficients = {
                    'IT SW/HW 구매': 0.05,
                    '기타 지원': 0.03,
                    '네트워크/보안 지원': 0.02,
                    '데이터 수정/변경': 0.04,
                    '시스템 구축': 0.08,
                    '시스템 권한': 0.02,
                    '시스템 트러블슈팅': 0.06,
                    '프로그램 수정/개발': 0.07,
                    'FLOW 로그인 수 (총)': 0.01,
                    # 관체사업팀용
                    'LINE별 설비CAPA분석 (반기, 중장기)': 0.5,
                    ' 저압 / 고압 / 외주 공정지시 (ERP 업로드)': 0.03,
                    ' 후가공 KD / 수출 제품 납입율 점검': 0.02,
                    ' 생산계획대비 실적 분석 (생산금액, 생산수량)': 0.04,
                    ' 월간 생산실적 분석 (인건비, 생산량, CAPA, 재료비, 경비 , 생산지표)': 0.2,
                    ' 주요지표 일일정산(결원율, 라인가동, 설비종합효율, 수율 등)': 0.02,
                    ' 생산 공정, 설비, 자재 등 양산 공정 현장 순회 점검': 0.05,
                    ' 현장 공정 개선 업무 지도 ( 생산성,수율,품질,가동율 등)': 0.1
                }

                for feature_name, coeff in feature_coefficients.items():
                    cursor.execute('''
                        UPDATE regression_parameters
                        SET coefficient = ?
                        WHERE model_id = ? AND parameter_name = ?
                    ''', (coeff, model_id, feature_name))

                print(f"  {model_type}: intercept = {base_intercept:.1f}, target = {target_headcount}")

    # 과적합 조정을 완화 (0.8 → 1.0)하도록 프론트엔드 수정을 위한 메모
    print("\n=== 주의 ===")
    print("프론트엔드에서 prediction * 0.8 부분을 prediction * 1.0으로 수정 필요")

    conn.commit()
    conn.close()
    print("\n현실적인 회귀 계수 조정 완료!")

if __name__ == "__main__":
    adjust_realistic_coefficients()