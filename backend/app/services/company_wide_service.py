"""
전사 인력 Feature 데이터 처리 서비스
"""
import pandas as pd
import sqlite3
from typing import Dict, Any, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DB_PATH = '/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db'

# 컬럼 매핑: Excel 한글 컬럼 → DB 영문 컬럼
# 공통 컬럼 (외부 지표)
COMMON_COLUMNS = {
    '글로벌 EV시장성장률': 'ev_growth_gl',
    '글로벌 자동차 시장성장률': 'v_growth_gl',
    '국내 자동차 수출액 증가율': 'v_export_kr',
    '국내 자동차부품 수출액 증가율': 'vp_export_kr',
    'GDP성장률': 'gdp_growth_kr',
    '소비자물가상승률': 'cpi_kr',
    '환율변화율_원화기준': 'exchange_rate_change_krw',
    '글로벌물류비지수': 'scm_index_gl',
    '국제유가': 'oil_gl',
    '인건비 증감률': 'labor_cost',
    '정원': 'headcount'
}

# R&A 전용 내부 지표
RNA_COLUMNS = {
    '매출액 증감률': 'revenue',
    '영업이익 증감률': 'profit',
    '가동률 증감률': 'operating_rate',
    '가동일수 증감률': 'operating_date'
}

# tonggibon 전용 내부 지표 (R&D 관련)
TONGGIBON_COLUMNS = {
    '매출액 증가율': 'revenue',
    '영업이익 증가율': 'profit',
    '연구개발비용 증감률': 'operating_rate',  # DB 컬럼 재사용
    '연구개발정부보조금 증감률': 'operating_date'  # DB 컬럼 재사용
}

def get_column_mapping(organization: str) -> dict:
    """조직에 따른 컬럼 매핑 반환"""
    mapping = COMMON_COLUMNS.copy()
    if organization == 'R&A':
        mapping.update(RNA_COLUMNS)
    elif organization == 'tonggibon':
        mapping.update(TONGGIBON_COLUMNS)
    return mapping

class CompanyWideService:
    """전사 인력 데이터 서비스"""

    def validate_excel_file(self, file_path: str, organization: str) -> Dict[str, Any]:
        """
        Excel 파일 검증

        Args:
            file_path: Excel 파일 경로
            organization: 조직 구분 ('R&A' or 'tonggibon')

        Returns:
            검증 결과 딕셔너리
        """
        try:
            # Excel 파일 읽기 (헤더 없이)
            xls = pd.ExcelFile(file_path)

            # 1. 시트 존재 확인
            if 'master' not in xls.sheet_names:
                raise ValueError("필수 시트 'master'를 찾을 수 없습니다")

            # 2. master 시트 읽기 (헤더 없이)
            df_raw = pd.read_excel(file_path, sheet_name='master', header=None)

            # 3. 구조 확인
            if len(df_raw) < 3:
                raise ValueError("데이터가 충분하지 않습니다 (최소 3행 필요)")

            if df_raw.iloc[0, 0] != 'kor' or df_raw.iloc[1, 0] != 'eng':
                raise ValueError("Excel 형식이 올바르지 않습니다 (첫 행: kor, 둘째 행: eng)")

            # 4. 한글 컬럼명 추출 (첫 번째 행)
            korean_columns = df_raw.iloc[0, 1:].tolist()

            # 5. 실제 데이터 추출 (3행부터)
            df = df_raw.iloc[2:].reset_index(drop=True)
            df.columns = ['year'] + korean_columns

            # 6. 조직에 맞는 컬럼 매핑 가져오기
            column_mapping = get_column_mapping(organization)

            # 7. 필수 컬럼 확인
            missing_columns = [col for col in column_mapping.keys() if col not in df.columns]
            if missing_columns:
                raise ValueError(f"필수 컬럼 누락: {missing_columns}")

            # 8. 데이터 타입 변환
            df['year'] = pd.to_numeric(df['year'], errors='coerce')

            for col in column_mapping.keys():
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 8. 유효한 데이터만 필터링 (연도가 있는 행만)
            df = df.dropna(subset=['year'])
            if len(df) == 0:
                raise ValueError("유효한 데이터가 없습니다")

            df['year'] = df['year'].astype(int)

            # 9. 검증 통과
            return {
                'is_valid': True,
                'rows': len(df),
                'years': df['year'].tolist(),
                'organization': organization,
                'data': df,
                'warnings': []
            }

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'organization': organization
            }

    def save_to_database(self, df: pd.DataFrame, organization: str) -> Dict[str, Any]:
        """
        데이터를 company_wide_features 테이블에 저장

        Args:
            df: 검증된 DataFrame
            organization: 조직 구분

        Returns:
            저장 결과
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            saved_count = 0
            updated_count = 0

            for _, row in df.iterrows():
                year = int(row['year'])

                # UPSERT: 같은 organization, year가 있으면 업데이트, 없으면 삽입
                cursor.execute("""
                    SELECT id FROM company_wide_features
                    WHERE organization = ? AND year = ?
                """, (organization, year))

                exists = cursor.fetchone()

                # 데이터 준비
                data = {
                    'organization': organization,
                    'year': year
                }

                # 조직별 컬럼 매핑
                column_mapping = get_column_mapping(organization)
                for kor_col, eng_col in column_mapping.items():
                    value = row[kor_col]
                    # NaN 처리
                    if pd.isna(value):
                        data[eng_col] = None
                    else:
                        if eng_col == 'headcount':
                            data[eng_col] = int(value)
                        else:
                            data[eng_col] = float(value)

                if exists:
                    # UPDATE
                    set_clause = ', '.join([f"{k} = ?" for k in data.keys() if k not in ['organization', 'year']])
                    values = [v for k, v in data.items() if k not in ['organization', 'year']]
                    values.extend([organization, year])

                    cursor.execute(f"""
                        UPDATE company_wide_features
                        SET {set_clause}, updated_at = CURRENT_TIMESTAMP
                        WHERE organization = ? AND year = ?
                    """, values)
                    updated_count += 1
                else:
                    # INSERT
                    columns = list(data.keys())
                    placeholders = ', '.join(['?'] * len(columns))
                    values = list(data.values())

                    cursor.execute(f"""
                        INSERT INTO company_wide_features ({', '.join(columns)})
                        VALUES ({placeholders})
                    """, values)
                    saved_count += 1

            conn.commit()
            conn.close()

            return {
                'success': True,
                'saved_count': saved_count,
                'updated_count': updated_count,
                'total': saved_count + updated_count,
                'organization': organization
            }

        except Exception as e:
            logger.error(f"Database save error: {e}")
            return {
                'success': False,
                'error': str(e),
                'organization': organization
            }

    def get_features_by_organization(self, organization: str) -> List[Dict[str, Any]]:
        """
        조직별 Feature 데이터 조회

        Args:
            organization: 'R&A' or 'tonggibon'

        Returns:
            Feature 데이터 리스트
        """
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                SELECT *
                FROM company_wide_features
                WHERE organization = ?
                ORDER BY year
            """, (organization,))

            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            conn.close()

            result = []
            for row in rows:
                data = dict(zip(columns, row))
                result.append(data)

            return result

        except Exception as e:
            logger.error(f"Query error: {e}")
            return []

    def get_all_features(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        모든 조직의 Feature 데이터 조회

        Returns:
            {'R&A': [...], 'tonggibon': [...]}
        """
        return {
            'R&A': self.get_features_by_organization('R&A'),
            'tonggibon': self.get_features_by_organization('tonggibon')
        }

company_wide_service = CompanyWideService()
