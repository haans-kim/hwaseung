"""
Team Features Service V2
조직별 Feature 매핑 및 팀 단위 인력 산정 데이터 처리
실제 템플릿 구조에 맞춰 재작성
"""

import pandas as pd
import sqlite3
from typing import Dict, Any, Optional, List
import json
from datetime import datetime

class TeamService:
    def __init__(self, db_path: str = "/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db"):
        self.db_path = db_path

    def get_db_connection(self):
        """데이터베이스 연결"""
        return sqlite3.connect(self.db_path)

    def validate_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        조직인력산정용 Excel 파일 검증

        Sheet 1: feature matching - 조직 계층 정보 (HQ, 본부, 담당, 실, 팀)
        Sheet 2: master - 실제 데이터 (HQ, 팀, 년, 월, 구분, F1-F9, 인력규모)
        """
        try:
            excel_file = pd.ExcelFile(file_path)

            # Sheet 이름 확인
            if len(excel_file.sheet_names) < 2:
                return {
                    "valid": False,
                    "error": "Excel 파일은 최소 2개의 시트가 필요합니다 (feature matching, master)"
                }

            # Sheet 1: feature matching (조직 정보)
            matching_sheet_name = excel_file.sheet_names[0]
            df_matching = excel_file.parse(matching_sheet_name)

            required_org_cols = ['HQ', '본부', '담당/사업단/센터', '실', '팀']
            if not all(col in df_matching.columns for col in required_org_cols):
                return {
                    "valid": False,
                    "error": f"feature matching 시트에 필수 조직 컬럼이 없습니다: {required_org_cols}"
                }

            # Sheet 2: master (실제 데이터)
            master_sheet_name = excel_file.sheet_names[1]
            df_master = excel_file.parse(master_sheet_name)

            required_master_cols = ['HQ', '팀', '년', '월', '구분', '인력규모']
            if not all(col in df_master.columns for col in required_master_cols):
                return {
                    "valid": False,
                    "error": f"master 시트에 필수 컬럼이 없습니다: {required_master_cols}"
                }

            # Feature 컬럼 찾기 (F1, F2, ... F9, F10 등)
            feature_cols = [col for col in df_master.columns if col.startswith('F') and col[1:].isdigit()]
            if len(feature_cols) == 0:
                return {
                    "valid": False,
                    "error": "master 시트에 Feature 컬럼(F1, F2, ...)이 없습니다"
                }

            # 데이터 검증
            if len(df_master) == 0:
                return {
                    "valid": False,
                    "error": "master 시트에 데이터가 없습니다"
                }

            # 통계 정보
            companies = df_master['HQ'].dropna().unique().tolist()
            teams = df_master['팀'].dropna().unique().tolist()
            years = df_master['년'].dropna().unique().tolist()
            months = df_master['월'].dropna().unique().tolist()
            positions = df_master['구분'].dropna().unique().tolist()

            return {
                "valid": True,
                "df_matching": df_matching,
                "df_master": df_master,
                "feature_columns": feature_cols,
                "companies": companies,
                "teams": teams,
                "years": years,
                "months": months,
                "positions": positions,
                "row_count": len(df_master),
                "team_count": len(teams),
                "feature_count": len(feature_cols)
            }

        except Exception as e:
            return {
                "valid": False,
                "error": f"파일 검증 중 오류 발생: {str(e)}"
            }

    def save_to_database(self, df_master: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """
        Master 데이터를 데이터베이스에 저장

        team_features 테이블에 저장:
        - organization (HQ)
        - team (팀)
        - year (년)
        - month (월)
        - position (구분)
        - feature_values (JSON: F1-F9 값들)
        - headcount (인력규모)
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()

        saved_count = 0
        errors = []

        try:
            for _, row in df_master.iterrows():
                # 필수 값 확인
                if pd.isna(row['HQ']) or pd.isna(row['팀']) or pd.isna(row['년']) or pd.isna(row['월']):
                    continue

                organization = row['HQ']
                team = row['팀']
                year = int(row['년'])
                month = int(row['월'])
                position = row.get('구분', '전체')
                headcount = int(row['인력규모']) if pd.notna(row['인력규모']) else None

                # Feature 값들을 JSON으로 구성
                feature_values = {}
                for feature_col in feature_cols:
                    if feature_col in row and pd.notna(row[feature_col]):
                        feature_values[feature_col] = float(row[feature_col])

                feature_values_json = json.dumps(feature_values, ensure_ascii=False)

                try:
                    # UPSERT: INSERT or UPDATE
                    cursor.execute("""
                        INSERT INTO team_features (
                            company, team, year, month, position, feature_values, headcount, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(company, team, year, month, position)
                        DO UPDATE SET
                            feature_values = excluded.feature_values,
                            headcount = excluded.headcount,
                            updated_at = excluded.updated_at
                    """, (
                        organization,
                        team,
                        year,
                        month,
                        position,
                        feature_values_json,
                        headcount,
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))

                    saved_count += 1

                except Exception as e:
                    errors.append(f"Row {organization}/{team}/{year}/{month}/{position}: {str(e)}")

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

    def get_team_features(self, organization: Optional[str] = None) -> List[Dict[str, Any]]:
        """Team Features 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM team_features"
        params = []

        if organization:
            query += " WHERE organization LIKE ?"
            params.append(f"{organization}%")

        query += " ORDER BY organization, year"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        result = []
        for row in rows:
            feature_values = json.loads(row[3]) if row[3] else {}
            result.append({
                'id': row[0],
                'organization': row[1],
                'year': row[2],
                'feature_values': feature_values,
                'headcount': row[4],
                'created_at': row[5],
                'updated_at': row[6]
            })

        return result

    def get_status(self) -> Dict[str, Any]:
        """Team 데이터 상태 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM team_features")
        total_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT organization) FROM team_features")
        org_count = cursor.fetchone()[0]

        conn.close()

        return {
            "total_count": total_count,
            "organization_count": org_count
        }

# Singleton instance
team_service = TeamService()
