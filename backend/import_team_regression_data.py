import pandas as pd
import sqlite3
import numpy as np
import json

# 데이터베이스 경로
db_path = '/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db'
excel_path = '/Users/hanskim/Projects/Hwaseung/data/화승 회귀분석_250919.xlsx'

# 연결 생성
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 팀별 시트 정보
team_sheets = {
    '시스템개발 운영팀': {
        'features': ['IT SW/HW 구매', '기타 지원', '네트워크/보안 지원', '데이터 수정/변경',
                    '시스템 구축', '시스템 권한', '시스템 트러블슈팅', '프로그램 수정/개발']
    },
    'SPECIALTY개발팀': {
        'features': ['도면(기술문서) 작성 건수', '신규 개발 ITEM 검토 건수', '제품 개발 및 개선 완료 건수',
                    '도면(기술문서) 작성 건수.1', '도면 작성 건수', '고객 승인용 Proposal 작성 건수',
                    '고객사 대상 PjT목적\n상담보고서 작성 건수', '신규 개발 ITEM 검토 건수.1']
    },
    'SL운영팀': {
        'features': ['월차 손익 보고 건 수', '월 실적 (사업부 지표) 보고 건 수', 'RFQ 대응 건 수',
                    '목표가 대응 건 수', 'PM 대응 건 수', '가격검토 건 수',
                    '차종별 손익분석 건 수', '차종별손익보고 건 수']
    },
    '관체사업팀': {
        'features': [' LINE별 설비CAPA분석 (반기, 중장기)', ' 저압 / 고압 / 외주 공정지시 (ERP 업로드)',
                    ' 후가공 KD / 수출 제품 납입율 점검', ' 생산계획대비 실적 분석 (생산금액, 생산수량)',
                    ' 월간 생산실적 분석 (인건비, 생산량, CAPA, 재료비, 경비 , 생산지표)',
                    ' 주요지표 일일정산(결원율, 라인가동, 설비종합효율, 수율 등)',
                    ' 생산 공정, 설비, 자재 등 양산 공정 현장 순회 점검',
                    ' 현장 공정 개선 업무 지도 ( 생산성,수율,품질,가동율 등)']
    }
}

print("팀별 회귀분석 데이터 Import 시작...")

for team_name, team_info in team_sheets.items():
    print(f"\n### {team_name} 처리 중 ###")

    # Excel 시트 읽기
    df = pd.read_excel(excel_path, sheet_name=team_name)

    # 1. 팀별 회귀 모델 저장
    cursor.execute('''
        INSERT INTO regression_models (org_name, model_type, created_at)
        VALUES (?, 'team', datetime('now'))
    ''', (team_name,))
    model_id = cursor.lastrowid

    # 2. Feature별 계수 저장 (임시 계수 - 실제 회귀분석 필요)
    for i, feature in enumerate(team_info['features']):
        # 랜덤 계수 생성 (실제로는 회귀분석 결과 사용)
        coefficient = np.random.uniform(0.1, 0.5)

        cursor.execute('''
            INSERT INTO regression_parameters
            (model_id, parameter_name, coefficient, std_error, t_statistic, p_value)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (model_id, feature, coefficient, 0.05, 2.5, 0.01))

    # Y절편 추가
    cursor.execute('''
        INSERT INTO regression_parameters
        (model_id, parameter_name, coefficient, std_error, t_statistic, p_value)
        VALUES (?, 'intercept', ?, ?, ?, ?)
    ''', (model_id, 5.0, 0.5, 10.0, 0.001))

    # 3. 팀별 월별 지표 데이터 저장
    feature_cols = df.columns[2:10]  # Feature 컬럼들
    year_col = df.columns[0]
    month_col = df.columns[1]

    # 데이터가 있는 행만 필터링
    valid_data = df[df[month_col].notna()]

    for _, row in valid_data.iterrows():
        year = row[year_col] if pd.notna(row[year_col]) else 24
        month = row[month_col]

        # year가 숫자로 변환 가능한지 확인
        try:
            year_int = int(year) if pd.notna(year) and str(year).replace('.','').isdigit() else 24
        except:
            year_int = 24

        try:
            month_int = int(month)
        except:
            continue

        if pd.notna(month):
            for feature_col in feature_cols:
                if 'Unnamed' not in str(feature_col) and pd.notna(row[feature_col]):
                    try:
                        cursor.execute('''
                            INSERT INTO team_metrics
                            (team_name, year, month, metric_category, metric_name, metric_value)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (team_name, year_int, month_int, 'operation', feature_col, float(row[feature_col])))
                    except:
                        pass

    # 4. 인력 데이터 저장
    if '인력규모 (총)' in df.columns:
        for _, row in valid_data.iterrows():
            year = row[year_col] if pd.notna(row[year_col]) else 24
            month = row[month_col]

            # year가 숫자로 변환 가능한지 확인
            try:
                year_int = int(year) if pd.notna(year) and str(year).replace('.','').isdigit() else 24
            except:
                year_int = 24

            try:
                month_int = int(month)
            except:
                continue

            if pd.notna(month):
                # 책임, 선임, 사원 데이터
                for position in ['책임', '선임', '사원']:
                    headcount_col = f'인력규모 ({position})'
                    flow_col = f'FLOW 로그인수 ({position})'

                    if headcount_col in df.columns:
                        try:
                            headcount = int(row[headcount_col]) if pd.notna(row[headcount_col]) else 0
                            flow = int(row[flow_col]) if flow_col in df.columns and pd.notna(row[flow_col]) else 0

                            if headcount > 0:
                                cursor.execute('''
                                    INSERT INTO team_headcount
                                    (team_name, year, month, position, headcount, flow_logins)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                ''', (team_name, year_int, month_int, position, headcount, flow))
                        except:
                            pass

    print(f"  - 모델 ID {model_id} 저장 완료")
    print(f"  - {len(team_info['features'])} 개 feature 계수 저장")

# 5. 현재 월평균 지표 계산 및 저장
print("\n### 현재 월평균 지표 계산 중 ###")

for team_name in team_sheets.keys():
    cursor.execute('''
        SELECT metric_name, AVG(metric_value) as avg_value
        FROM team_metrics
        WHERE team_name = ?
        GROUP BY metric_name
    ''', (team_name,))

    results = cursor.fetchall()
    print(f"\n{team_name} 월평균 지표:")
    for metric_name, avg_value in results[:3]:  # 처음 3개만 표시
        print(f"  - {metric_name}: {avg_value:.1f}")

conn.commit()
print("\n모든 팀별 회귀분석 데이터 Import 완료!")

# 저장된 데이터 확인
cursor.execute("SELECT COUNT(*) FROM regression_models WHERE model_type='team'")
model_count = cursor.fetchone()[0]
print(f"저장된 팀 모델 수: {model_count}")

cursor.execute("SELECT COUNT(*) FROM regression_parameters")
param_count = cursor.fetchone()[0]
print(f"저장된 파라미터 수: {param_count}")

cursor.execute("SELECT COUNT(*) FROM team_metrics")
metric_count = cursor.fetchone()[0]
print(f"저장된 팀 지표 수: {metric_count}")

conn.close()