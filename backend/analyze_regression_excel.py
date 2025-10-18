import pandas as pd
import numpy as np
import json

file_path = str(Path(__file__).parent.parent / 'data/화승 회귀분석_250919.xlsx'

print("=" * 80)
print("화승 회귀분석 Excel 파일 상세 분석 리포트")
print("=" * 80)

# 모든 시트 읽기
xl = pd.ExcelFile(file_path)

# 1. 전사 레벨 시트 분석 (R&A 전사, 통기본 전사)
print("\n### 1. 전사 레벨 회귀분석 시트 ###")
for sheet_name in ['R&A 전사', '통기본 전사']:
    print(f"\n## {sheet_name} ##")
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # 데이터 영역과 통계 영역 분리
    print(f"- 전체 크기: {df.shape}")

    # 독립변수 컬럼 확인
    independent_vars = []
    for col in df.columns[:9]:
        if 'Unnamed' not in str(col):
            independent_vars.append(col)

    print(f"- 독립변수: {independent_vars}")

    # 데이터 행 확인 (연도가 있는 행)
    data_rows = df[df['년도'].notna() & (df['년도'] != '년도')]
    if not data_rows.empty:
        # 숫자로 변환 가능한 연도만 필터링
        data_rows = data_rows[pd.to_numeric(data_rows['년도'], errors='coerce').notna()]
        print(f"- 데이터 기간: {data_rows['년도'].min()} ~ {data_rows['년도'].max()}")
        print(f"- 데이터 행 수: {len(data_rows)}")

    # 회귀분석 통계 영역 찾기
    if '요약 출력' in df.columns:
        stats_section = df[df['요약 출력'].notna()]
        if not stats_section.empty:
            print("\n  회귀분석 통계:")
            for idx, row in stats_section.iterrows():
                stat_name = row['요약 출력']
                if pd.notna(stat_name) and stat_name != '요약 출력':
                    # 통계 값 찾기
                    for col in df.columns[11:]:
                        if pd.notna(row[col]):
                            print(f"    - {stat_name}: {row[col]}")
                            break

# 2. 팀별 시트 분석
print("\n### 2. 팀별 업무 지표 및 인력 시트 ###")
team_sheets = ['시스템개발 운영팀', 'SPECIALTY개발팀', 'SL운영팀', '관체사업팀']

for sheet_name in team_sheets:
    print(f"\n## {sheet_name} ##")
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    print(f"- 전체 크기: {df.shape}")

    # 업무 지표 컬럼 확인
    metric_columns = []
    for col in df.columns[2:10]:  # 일반적으로 3번째 컬럼부터 업무 지표
        if 'Unnamed' not in str(col) and 'FLOW' not in str(col) and '인력' not in str(col):
            metric_columns.append(col)

    print(f"- 업무 지표 수: {len(metric_columns)}")
    print(f"- 업무 지표 종류:")
    for i, metric in enumerate(metric_columns[:5], 1):  # 처음 5개만 표시
        print(f"    {i}. {metric}")
    if len(metric_columns) > 5:
        print(f"    ... 외 {len(metric_columns)-5}개")

    # 데이터 기간 확인
    year_col = df.columns[0]  # 첫 번째 컬럼이 연도
    month_col = df.columns[1]  # 두 번째 컬럼이 월

    valid_data = df[df[month_col].notna()]
    if not valid_data.empty:
        # 연도 확인
        years = valid_data[year_col].dropna().unique()
        years = [y for y in years if str(y).replace('.','').isdigit()]
        if years:
            print(f"- 데이터 연도: {years}")

        # 월 범위 확인
        months = valid_data[month_col]
        months = pd.to_numeric(months, errors='coerce').dropna()
        if not months.empty:
            print(f"- 월 범위: {int(months.min())} ~ {int(months.max())}")
            print(f"- 데이터 행 수: {len(valid_data)}")

    # 인력 관련 컬럼 확인
    headcount_cols = [col for col in df.columns if '인력' in str(col) or 'FLOW' in str(col)]
    if headcount_cols:
        print(f"- 인력/FLOW 관련 컬럼: {len(headcount_cols)}개")
        for col in headcount_cols[:3]:
            print(f"    • {col}")

# 3. 회귀분석 모델 정보 추출
print("\n### 3. 회귀분석 모델 요약 ###")

# R&A 전사 시트에서 회귀분석 결과 추출
df_rna = pd.read_excel(file_path, sheet_name='R&A 전사')

print("\n## 전사 모델 (R&A) ##")
# 회귀분석 통계량 찾기
stats_area = df_rna.iloc[:, 10:19]  # 요약 출력 영역
for idx, row in df_rna.iterrows():
    if pd.notna(row['요약 출력']):
        stat_name = str(row['요약 출력'])
        if '다중 상관계수' in stat_name:
            print(f"- R: {row.iloc[11] if pd.notna(row.iloc[11]) else 'N/A'}")
        elif '결정계수' in stat_name:
            print(f"- R²: {row.iloc[11] if pd.notna(row.iloc[11]) else 'N/A'}")
        elif '조정된 결정계수' in stat_name:
            print(f"- Adjusted R²: {row.iloc[11] if pd.notna(row.iloc[11]) else 'N/A'}")
        elif '표준 오차' in stat_name:
            print(f"- 표준오차: {row.iloc[11] if pd.notna(row.iloc[11]) else 'N/A'}")

# 4. 데이터 품질 체크
print("\n### 4. 데이터 품질 및 특이사항 ###")

for sheet_name in xl.sheet_names:
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    # NaN 비율 계산
    nan_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])

    # 빈 열 확인
    empty_cols = [col for col in df.columns if df[col].isna().all()]

    print(f"\n## {sheet_name} ##")
    print(f"- 결측치 비율: {nan_ratio:.1%}")
    print(f"- 빈 컬럼 수: {len(empty_cols)}")

    # 데이터 타입 혼재 확인
    mixed_type_cols = []
    for col in df.columns:
        if 'Unnamed' not in str(col):
            unique_types = df[col].dropna().apply(type).unique()
            if len(unique_types) > 1:
                mixed_type_cols.append(col)

    if mixed_type_cols:
        print(f"- 혼합 데이터 타입 컬럼: {len(mixed_type_cols)}개")

print("\n" + "=" * 80)
print("분석 완료")
print("=" * 80)