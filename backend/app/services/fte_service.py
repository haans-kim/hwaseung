"""
FTE (Full-Time Equivalent) 분석 서비스
평균 FTE 데이터를 피벗 구조로 저장 (UI 호환)
"""

import pandas as pd
import sqlite3
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

class FTEService:
    def __init__(self):
        # 🔧 FIX: Docker 환경을 위해 환경 변수 사용
        db_path_str = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), '../../hwaseung_RnD.db'))
        self.db_path = Path(db_path_str)

    def validate_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        Excel 파일 검증 (평균FTE 시트)

        Template 구조:
        - 계열사, 부서, 사용자직위, 평균FTE, 인원수, 평균FTE/인원수
        - 각 팀의 직급별 데이터를 피벗하여 저장
        """
        try:
            excel_file = pd.ExcelFile(file_path)

            # "평균FTE" 시트 찾기
            fte_sheet_name = None
            for sheet_name in excel_file.sheet_names:
                if '평균FTE' in sheet_name or '평균' in sheet_name:
                    fte_sheet_name = sheet_name
                    break

            if not fte_sheet_name:
                return {
                    'valid': False,
                    'error': f'"평균FTE" 시트를 찾을 수 없습니다. 사용 가능한 시트: {", ".join(excel_file.sheet_names)}'
                }

            # 데이터 읽기
            df = excel_file.parse(fte_sheet_name)

            # 필수 컬럼 확인
            required_cols = ['계열사', '부서', '사용자직위', '평균FTE', '인원수', '평균FTE/인원수']
            missing_cols = [col for col in required_cols if col not in df.columns]

            if missing_cols:
                return {
                    'valid': False,
                    'error': f'필수 컬럼이 없습니다: {", ".join(missing_cols)}'
                }

            # 데이터 검증
            if df.empty:
                return {
                    'valid': False,
                    'error': '데이터가 비어있습니다'
                }

            # 회사, 팀, 직급 정보 추출
            companies = df['계열사'].unique().tolist()
            teams = df['부서'].unique().tolist()
            positions = df['사용자직위'].unique().tolist()

            return {
                'valid': True,
                'dataframe': df,
                'sheet_name': fte_sheet_name,
                'row_count': len(df),
                'companies': companies,
                'company_count': len(companies),
                'team_count': len(teams),
                'positions': positions,
                'position_count': len(positions)
            }

        except Exception as e:
            return {
                'valid': False,
                'error': f'파일 검증 실패: {str(e)}'
            }

    def save_to_database(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        피벗 구조로 데이터베이스에 저장
        각 팀별로 전체/책임/선임/사원 데이터를 하나의 행으로 피벗
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            saved_count = 0
            errors = []

            # 팀별로 그룹화하여 피벗
            for (company, team), group in df.groupby(['계열사', '부서']):
                try:
                    # 각 직급별 데이터 추출
                    pivot_data = {
                        '팀명': team,
                        '회사': company,
                        'FTE_전체': None,
                        'FTE_책임': None,
                        'FTE_선임': None,
                        'FTE_사원': None,
                        '인원수_전체': None,
                        '인원수_책임': None,
                        '인원수_선임': None,
                        '인원수_사원': None,
                        'FTE_per_인원_전체': None,
                        'FTE_per_인원_책임': None,
                        'FTE_per_인원_선임': None,
                        'FTE_per_인원_사원': None
                    }

                    for _, row in group.iterrows():
                        position = row['사용자직위']
                        avg_fte = row['평균FTE']
                        headcount = row['인원수']
                        avg_per_person = row['평균FTE/인원수']

                        if position == '전체':
                            pivot_data['FTE_전체'] = avg_fte
                            pivot_data['인원수_전체'] = headcount
                            pivot_data['FTE_per_인원_전체'] = avg_per_person
                        elif position == '책임':
                            pivot_data['FTE_책임'] = avg_fte
                            pivot_data['인원수_책임'] = headcount
                            pivot_data['FTE_per_인원_책임'] = avg_per_person
                        elif position == '선임':
                            pivot_data['FTE_선임'] = avg_fte
                            pivot_data['인원수_선임'] = headcount
                            pivot_data['FTE_per_인원_선임'] = avg_per_person
                        elif position == '사원':
                            pivot_data['FTE_사원'] = avg_fte
                            pivot_data['인원수_사원'] = headcount
                            pivot_data['FTE_per_인원_사원'] = avg_per_person

                    # UPSERT
                    cursor.execute("""
                        INSERT INTO fte (
                            팀명, 회사,
                            FTE_전체, FTE_책임, FTE_선임, FTE_사원,
                            인원수_전체, 인원수_책임, 인원수_선임, 인원수_사원,
                            FTE_per_인원_전체, FTE_per_인원_책임, FTE_per_인원_선임, FTE_per_인원_사원
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(회사, 팀명) DO UPDATE SET
                            FTE_전체 = excluded.FTE_전체,
                            FTE_책임 = excluded.FTE_책임,
                            FTE_선임 = excluded.FTE_선임,
                            FTE_사원 = excluded.FTE_사원,
                            인원수_전체 = excluded.인원수_전체,
                            인원수_책임 = excluded.인원수_책임,
                            인원수_선임 = excluded.인원수_선임,
                            인원수_사원 = excluded.인원수_사원,
                            FTE_per_인원_전체 = excluded.FTE_per_인원_전체,
                            FTE_per_인원_책임 = excluded.FTE_per_인원_책임,
                            FTE_per_인원_선임 = excluded.FTE_per_인원_선임,
                            FTE_per_인원_사원 = excluded.FTE_per_인원_사원,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        pivot_data['팀명'], pivot_data['회사'],
                        pivot_data['FTE_전체'], pivot_data['FTE_책임'], pivot_data['FTE_선임'], pivot_data['FTE_사원'],
                        pivot_data['인원수_전체'], pivot_data['인원수_책임'], pivot_data['인원수_선임'], pivot_data['인원수_사원'],
                        pivot_data['FTE_per_인원_전체'], pivot_data['FTE_per_인원_책임'],
                        pivot_data['FTE_per_인원_선임'], pivot_data['FTE_per_인원_사원']
                    ))

                    saved_count += 1

                except Exception as e:
                    errors.append(f'{company} - {team}: {str(e)}')

            conn.commit()
            conn.close()

            return {
                'success': True,
                'saved_count': saved_count,
                'errors': errors if errors else None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_fte_data(self, company: Optional[str] = None, team: Optional[str] = None) -> List[Dict[str, Any]]:
        """FTE 데이터 조회"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM fte WHERE 1=1"
            params = []

            if company:
                query += " AND 회사 = ?"
                params.append(company)

            if team:
                query += " AND 팀명 = ?"
                params.append(team)

            query += " ORDER BY 회사, 팀명"

            cursor.execute(query, params)
            rows = cursor.fetchall()

            result = [dict(row) for row in rows]

            conn.close()
            return result

        except Exception as e:
            print(f"Error getting FTE data: {e}")
            return []

    def delete_fte_data(self, company: Optional[str] = None, team: Optional[str] = None) -> Dict[str, Any]:
        """FTE 데이터 삭제"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            query = "DELETE FROM fte WHERE 1=1"
            params = []

            if company:
                query += " AND 회사 = ?"
                params.append(company)

            if team:
                query += " AND 팀명 = ?"
                params.append(team)

            cursor.execute(query, params)
            deleted_count = cursor.rowcount

            conn.commit()
            conn.close()

            return {
                'success': True,
                'deleted_count': deleted_count
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def get_status(self) -> Dict[str, Any]:
        """FTE 데이터 상태 조회"""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM fte")
            total_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT 회사) FROM fte")
            company_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT 팀명) FROM fte")
            team_count = cursor.fetchone()[0]

            cursor.execute("SELECT 회사, COUNT(*) as count FROM fte GROUP BY 회사")
            company_breakdown = [{'company': row[0], 'count': row[1]} for row in cursor.fetchall()]

            conn.close()

            return {
                'total_count': total_count,
                'company_count': company_count,
                'team_count': team_count,
                'company_breakdown': company_breakdown
            }

        except Exception as e:
            return {
                'error': str(e)
            }

# 싱글톤 인스턴스
fte_service = FTEService()
