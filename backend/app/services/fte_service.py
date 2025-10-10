"""
FTE Service
FTE 분석 데이터 처리 (평균FTE 시트 - 간소화된 구조)
"""

import pandas as pd
import sqlite3
from typing import Dict, Any, Optional, List
from datetime import datetime

class FTEService:
    def __init__(self, db_path: str = "/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db"):
        self.db_path = db_path

    def get_db_connection(self):
        """데이터베이스 연결"""
        return sqlite3.connect(self.db_path)

    def validate_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        FTE Excel 파일 검증 (평균FTE 시트)

        Expected columns: 계열사, 부서, 사용자직위, 평균FTE, 인원수, 평균FTE/인원수
        """
        try:
            # Excel 파일 읽기
            excel_file = pd.ExcelFile(file_path)

            # "평균FTE" 시트 찾기
            fte_sheet_name = None
            for sheet_name in excel_file.sheet_names:
                if '평균FTE' in sheet_name or '평균' in sheet_name:
                    fte_sheet_name = sheet_name
                    break

            if not fte_sheet_name:
                return {
                    "valid": False,
                    "error": "Excel 파일에 '평균FTE' 시트가 없습니다"
                }

            # 시트 읽기
            df = excel_file.parse(fte_sheet_name)

            # 필수 컬럼 검증
            required_cols = ['계열사', '부서', '사용자직위', '평균FTE', '인원수', '평균FTE/인원수']

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return {
                    "valid": False,
                    "error": f"필수 컬럼이 없습니다: {missing_cols}"
                }

            # 데이터 검증
            if len(df) == 0:
                return {
                    "valid": False,
                    "error": "데이터가 없습니다"
                }

            # 통계 정보
            companies = df['계열사'].dropna().unique().tolist()
            teams = df['부서'].dropna().unique().tolist()
            positions = df['사용자직위'].dropna().unique().tolist()

            return {
                "valid": True,
                "dataframe": df,
                "sheet_name": fte_sheet_name,
                "row_count": len(df),
                "companies": companies,
                "teams": teams,
                "positions": positions,
                "company_count": len(companies),
                "team_count": len(teams),
                "position_count": len(positions)
            }

        except Exception as e:
            return {
                "valid": False,
                "error": f"파일 검증 중 오류 발생: {str(e)}"
            }

    def save_to_database(self, df: pd.DataFrame) -> Dict[str, Any]:
        """FTE 데이터를 데이터베이스에 저장"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        saved_count = 0
        errors = []

        try:
            for _, row in df.iterrows():
                # 필수 값 확인
                if pd.isna(row['계열사']) or pd.isna(row['부서']) or pd.isna(row['사용자직위']):
                    continue

                try:
                    cursor.execute("""
                        INSERT INTO fte (
                            company, team, position, avg_fte, headcount, avg_fte_per_person, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(company, team, position)
                        DO UPDATE SET
                            avg_fte = excluded.avg_fte,
                            headcount = excluded.headcount,
                            avg_fte_per_person = excluded.avg_fte_per_person,
                            updated_at = excluded.updated_at
                    """, (
                        row['계열사'],
                        row['부서'],
                        row['사용자직위'],
                        float(row['평균FTE']) if pd.notna(row['평균FTE']) else None,
                        float(row['인원수']) if pd.notna(row['인원수']) else None,
                        float(row['평균FTE/인원수']) if pd.notna(row['평균FTE/인원수']) else None,
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))

                    saved_count += 1

                except Exception as e:
                    errors.append(f"Row {row['계열사']}/{row['부서']}/{row['사용자직위']}: {str(e)}")

            conn.commit()

            return {
                "success": True,
                "saved_count": saved_count,
                "errors": errors if errors else None
            }

        except Exception as e:
            conn.rollback()
            return {
                "success": False,
                "error": f"데이터 저장 실패: {str(e)}"
            }
        finally:
            conn.close()

    def get_fte_data(self, company: Optional[str] = None, team: Optional[str] = None) -> List[Dict[str, Any]]:
        """FTE 데이터 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM fte WHERE 1=1"
        params = []

        if company:
            query += " AND company = ?"
            params.append(company)

        if team:
            query += " AND team = ?"
            params.append(team)

        query += " ORDER BY company, team, position"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        columns = [
            'id', 'company', 'team', 'position', 'avg_fte', 'headcount',
            'avg_fte_per_person', 'created_at', 'updated_at'
        ]

        return [dict(zip(columns, row)) for row in rows]

    def delete_fte_data(self, company: Optional[str] = None, team: Optional[str] = None) -> Dict[str, Any]:
        """FTE 데이터 삭제"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            query = "DELETE FROM fte WHERE 1=1"
            params = []

            if company:
                query += " AND company = ?"
                params.append(company)

            if team:
                query += " AND team = ?"
                params.append(team)

            cursor.execute(query, params)
            deleted_count = cursor.rowcount

            conn.commit()

            return {
                "success": True,
                "deleted_count": deleted_count
            }

        except Exception as e:
            conn.rollback()
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            conn.close()

    def get_status(self) -> Dict[str, Any]:
        """FTE 데이터 상태 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        # 전체 통계
        cursor.execute("SELECT COUNT(*) FROM fte")
        total_count = cursor.fetchone()[0]

        # 회사별 통계
        cursor.execute("""
            SELECT company, COUNT(*) as count, COUNT(DISTINCT team) as team_count, COUNT(DISTINCT position) as position_count
            FROM fte
            GROUP BY company
        """)
        company_stats = {}
        for row in cursor.fetchall():
            company_stats[row[0]] = {
                'record_count': row[1],
                'team_count': row[2],
                'position_count': row[3]
            }

        # 팀 목록
        cursor.execute("SELECT DISTINCT team FROM fte ORDER BY team")
        teams = [row[0] for row in cursor.fetchall()]

        # 직위 목록
        cursor.execute("SELECT DISTINCT position FROM fte ORDER BY position")
        positions = [row[0] for row in cursor.fetchall()]

        conn.close()

        return {
            "total_count": total_count,
            "company_stats": company_stats,
            "teams": teams,
            "positions": positions,
            "team_count": len(teams),
            "position_count": len(positions)
        }

# Singleton instance
fte_service = FTEService()
