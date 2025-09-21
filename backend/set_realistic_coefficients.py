#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3

def set_realistic_coefficients():
    """현실적인 회귀 계수 설정"""

    conn = sqlite3.connect('/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db')
    cursor = conn.cursor()

    print("=== 현실적인 회귀 계수 설정 ===")

    # 팀별 현재 인원 (실제 데이터)
    teams_headcount = {
        '시스템개발 운영팀': {'총': 26, '책임': 5, '선임': 11, '사원': 10},
        'SPECIALTY개발팀': {'총': 13, '책임': 2, '선임': 5, '사원': 6},
        'SL운영팀': {'총': 7, '책임': 3, '선임': 2, '사원': 2},
        '관체사업팀': {'총': 11, '책임': 4, '선임': 6, '사원': 1}
    }

    # 기본 메트릭 값들 (평균적인 업무량)
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

    for team_name, headcounts in teams_headcount.items():
        print(f"\n{team_name} 계수 설정:")

        for model_type, target_headcount in headcounts.items():
            # 모델 ID 찾기
            cursor.execute('''
                SELECT id FROM regression_models
                WHERE org_name = ? AND model_type = ?
            ''', (team_name, model_type))

            model_result = cursor.fetchone()
            if model_result:
                model_id = model_result[0]

                # intercept를 목표 인원의 60% 정도로 설정
                base_intercept = target_headcount * 0.6

                # intercept 업데이트
                cursor.execute('''
                    UPDATE regression_parameters
                    SET coefficient = ?
                    WHERE model_id = ? AND parameter_name = 'intercept'
                ''', (base_intercept, model_id))

                # 나머지 40%를 feature들로 분배
                remaining_headcount = target_headcount * 0.4
                total_base_metric_value = sum(base_metrics.values())

                # 각 feature의 비중에 따라 계수 설정
                feature_coefficients = {}
                for feature_name, base_value in base_metrics.items():
                    # feature의 기여도 = (나머지 인원 * feature 비중) / feature 기본값
                    feature_contribution = remaining_headcount * (base_value / total_base_metric_value)
                    coefficient = feature_contribution / base_value
                    feature_coefficients[feature_name] = round(coefficient, 4)

                # 계수 업데이트
                for feature_name, coeff in feature_coefficients.items():
                    cursor.execute('''
                        UPDATE regression_parameters
                        SET coefficient = ?
                        WHERE model_id = ? AND parameter_name = ?
                    ''', (coeff, model_id, feature_name))

                print(f"  {model_type}: intercept = {base_intercept:.1f}")
                print(f"    예상 합계 = {base_intercept + sum(coeff * base_metrics[fname] for fname, coeff in feature_coefficients.items()):.1f}")
                print(f"    목표 = {target_headcount}")

                # 관체사업팀의 추가 feature들도 설정
                if team_name == '관체사업팀':
                    additional_features = {
                        'LINE별 설비CAPA분석 (반기, 중장기)': 10,
                        ' 저압 / 고압 / 외주 공정지시 (ERP 업로드)': 50,
                        ' 후가공 KD / 수출 제품 납입율 점검': 30,
                        ' 생산계획대비 실적 분석 (생산금액, 생산수량)': 40,
                        ' 월간 생산실적 분석 (인건비, 생산량, CAPA, 재료비, 경비 , 생산지표)': 60,
                        ' 주요지표 일일정산(결원율, 라인가동, 설비종합효율, 수율 등)': 20,
                        ' 생산 공정, 설비, 자재 등 양산 공정 현장 순회 점검': 35,
                        ' 현장 공정 개선 업무 지도 ( 생산성,수율,품질,가동율 등)': 25
                    }

                    # 관체사업팀 추가 feature 계수 설정 (작은 값들)
                    for feature_name, base_value in additional_features.items():
                        coeff = 0.001  # 매우 작은 기여도
                        cursor.execute('''
                            UPDATE regression_parameters
                            SET coefficient = ?
                            WHERE model_id = ? AND parameter_name = ?
                        ''', (coeff, model_id, feature_name))

    conn.commit()
    conn.close()
    print(f"\n현실적인 회귀 계수 설정 완료!")

if __name__ == "__main__":
    set_realistic_coefficients()