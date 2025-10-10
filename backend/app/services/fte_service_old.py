"""
FTE Service
FTE 분석 데이터 처리 (평균FTE 시트만 사용)
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
        FTE Excel 파일 검증 (평균FTE 시트만 사용)

        Expected columns:
        - 회사, 팀명, 기간
        - FTE_전체, FTE_책임, FTE_선임, FTE_사원
        - 인원수_전체, 인원수_책임, 인원수_선임, 인원수_사원
        - FTE_per_인원_전체, FTE_per_인원_책임, FTE_per_인원_선임, FTE_per_인원_사원
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
            required_cols = [
                '회사', '팀명', '기간',
                'FTE_전체', 'FTE_책임', 'FTE_선임', 'FTE_사원',
                '인원수_전체', '인원수_책임', '인원수_선임', '인원수_사원',
                'FTE_per_인원_전체', 'FTE_per_인원_책임', 'FTE_per_인원_선임', 'FTE_per_인원_사원'
            ]

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

            # 회사, 팀명, 기간 값 확인
            if df['회사'].isna().all():
                return {
                    "valid": False,
                    "error": "'회사' 컬럼에 값이 없습니다"
                }

            if df['팀명'].isna().all():
                return {
                    "valid": False,
                    "error": "'팀명' 컬럼에 값이 없습니다"
                }

            if df['기간'].isna().all():
                return {
                    "valid": False,
                    "error": "'기간' 컬럼에 값이 없습니다"
                }

            # 통계 정보
            companies = df['회사'].dropna().unique().tolist()
            teams = df['팀명'].dropna().unique().tolist()
            periods = df['기간'].dropna().unique().tolist()

            return {
                "valid": True,
                "dataframe": df,
                "sheet_name": fte_sheet_name,
                "row_count": len(df),
                "companies": companies,
                "teams": teams,
                "periods": periods,
                "company_count": len(companies),
                "team_count": len(teams),
                "period_count": len(periods)
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
                if pd.isna(row['회사']) or pd.isna(row['팀명']) or pd.isna(row['기간']):
                    continue

                try:
                    cursor.execute("""
                        INSERT INTO fte (
                            회사, 팀명, 기간,
                            FTE_전체, FTE_책임, FTE_선임, FTE_사원,
                            인원수_전체, 인원수_책임, 인원수_선임, 인원수_사원,
                            FTE_per_인원_전체, FTE_per_인원_책임, FTE_per_인원_선임, FTE_per_인원_사원,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(회사, 팀명, 기간)
                        DO UPDATE SET
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
                            updated_at = excluded.updated_at
                    """, (
                        row['회사'],
                        row['팀명'],
                        row['기간'],
                        float(row['FTE_전체']) if pd.notna(row['FTE_전체']) else None,
                        float(row['FTE_책임']) if pd.notna(row['FTE_책임']) else None,
                        float(row['FTE_선임']) if pd.notna(row['FTE_선임']) else None,
                        float(row['FTE_사원']) if pd.notna(row['FTE_사원']) else None,
                        int(row['인원수_전체']) if pd.notna(row['인원수_전체']) else None,
                        int(row['인원수_책임']) if pd.notna(row['인원수_책임']) else None,
                        int(row['인원수_선임']) if pd.notna(row['인원수_선임']) else None,
                        int(row['인원수_사원']) if pd.notna(row['인원수_사원']) else None,
                        float(row['FTE_per_인원_전체']) if pd.notna(row['FTE_per_인원_전체']) else None,
                        float(row['FTE_per_인원_책임']) if pd.notna(row['FTE_per_인원_책임']) else None,
                        float(row['FTE_per_인원_선임']) if pd.notna(row['FTE_per_인원_선임']) else None,
                        float(row['FTE_per_인원_사원']) if pd.notna(row['FTE_per_인원_사원']) else None,
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))

                    saved_count += 1

                except Exception as e:
                    errors.append(f"Row {row['회사']}/{row['팀명']}/{row['기간']}: {str(e)}")

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
            query += " AND 회사 = ?"
            params.append(company)

        if team:
            query += " AND 팀명 = ?"
            params.append(team)

        query += " ORDER BY 회사, 팀명, 기간"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        columns = [
            'id', '팀명', '기간', 'FTE_전체', 'FTE_책임', 'FTE_선임', 'FTE_사원',
            '인원수_전체', '인원수_책임', '인원수_선임', '인원수_사원',
            'FTE_per_인원_전체', 'FTE_per_인원_책임', 'FTE_per_인원_선임', 'FTE_per_인원_사원',
            'created_at', 'updated_at', '회사'
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
                query += " AND 회사 = ?"
                params.append(company)

            if team:
                query += " AND 팀명 = ?"
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
            SELECT 회사, COUNT(*) as count, COUNT(DISTINCT 팀명) as team_count, COUNT(DISTINCT 기간) as period_count
            FROM fte
            GROUP BY 회사
        """)
        company_stats = {}
        for row in cursor.fetchall():
            company_stats[row[0]] = {
                'record_count': row[1],
                'team_count': row[2],
                'period_count': row[3]
            }

        # 팀 목록
        cursor.execute("SELECT DISTINCT 팀명 FROM fte ORDER BY 팀명")
        teams = [row[0] for row in cursor.fetchall()]

        # 기간 목록
        cursor.execute("SELECT DISTINCT 기간 FROM fte ORDER BY 기간")
        periods = [row[0] for row in cursor.fetchall()]

        conn.close()

        return {
            "total_count": total_count,
            "company_stats": company_stats,
            "teams": teams,
            "periods": periods,
            "team_count": len(teams),
            "period_count": len(periods)
        }

# Singleton instance
fte_service = FTEService()
