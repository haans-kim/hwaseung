"""
Team Features Service
조직별 Feature 매핑 및 팀 단위 인력 산정 데이터 처리
"""

import pandas as pd
import sqlite3
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
import os

class TeamService:
    def __init__(self, db_path: str = "/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db"):
        self.db_path = db_path

    def get_db_connection(self):
        """데이터베이스 연결"""
        return sqlite3.connect(self.db_path)

    def validate_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        조직인력산정용 Excel 파일 검증

        Sheet 1: Feature Mapping (조직, Feature 이름, 설명, 사용여부)
        Sheet 2: Master Data (연도, 조직, [동적 features], 인원)
        """
        try:
            # Excel 파일 읽기
            excel_file = pd.ExcelFile(file_path)

            # Sheet 이름 확인
            if len(excel_file.sheet_names) < 2:
                return {
                    "valid": False,
                    "error": "Excel 파일은 최소 2개의 시트가 필요합니다 (Feature Mapping, Master Data)"
                }

            # Sheet 1: Feature Mapping 검증
            mapping_sheet = excel_file.parse(excel_file.sheet_names[0])
            required_mapping_cols = ['조직', 'Feature 이름', '설명', '사용여부']

            if not all(col in mapping_sheet.columns for col in required_mapping_cols):
                return {
                    "valid": False,
                    "error": f"Feature Mapping 시트에 필수 컬럼이 없습니다: {required_mapping_cols}"
                }

            # Sheet 2: Master Data 검증
            master_sheet = excel_file.parse(excel_file.sheet_names[1])
            required_master_cols = ['연도', '조직', '인원']

            if not all(col in master_sheet.columns for col in required_master_cols):
                return {
                    "valid": False,
                    "error": f"Master Data 시트에 필수 컬럼이 없습니다: {required_master_cols}"
                }

            # Feature 매핑과 Master Data의 조직 일치성 확인
            mapping_orgs = set(mapping_sheet['조직'].unique())
            master_orgs = set(master_sheet['조직'].unique())

            if not master_orgs.issubset(mapping_orgs):
                missing_orgs = master_orgs - mapping_orgs
                return {
                    "valid": False,
                    "error": f"Master Data의 조직이 Feature Mapping에 없습니다: {missing_orgs}"
                }

            # 활성화된 Feature 추출
            active_features = {}
            for org in mapping_orgs:
                org_features = mapping_sheet[
                    (mapping_sheet['조직'] == org) &
                    (mapping_sheet['사용여부'] == 'Y')
                ]['Feature 이름'].tolist()
                active_features[org] = org_features

            # Master Data의 Feature 컬럼 검증
            master_feature_cols = [col for col in master_sheet.columns
                                  if col not in ['연도', '조직', '인원']]

            for org in master_orgs:
                org_data = master_sheet[master_sheet['조직'] == org]
                expected_features = active_features.get(org, [])

                # Master Data에 있는 feature가 모두 매핑에 있는지 확인
                for feature in master_feature_cols:
                    if org_data[feature].notna().any():  # 실제 값이 있는 컬럼만 체크
                        if feature not in expected_features:
                            return {
                                "valid": False,
                                "error": f"조직 '{org}'의 Feature '{feature}'가 Feature Mapping에 활성화되어 있지 않습니다"
                            }

            return {
                "valid": True,
                "mapping_sheet": mapping_sheet,
                "master_sheet": master_sheet,
                "active_features": active_features,
                "organizations": list(mapping_orgs),
                "feature_count": sum(len(v) for v in active_features.values()),
                "data_rows": len(master_sheet)
            }

        except Exception as e:
            return {
                "valid": False,
                "error": f"파일 검증 중 오류 발생: {str(e)}"
            }

    def save_feature_mapping(self, mapping_df: pd.DataFrame) -> Dict[str, Any]:
        """Feature Mapping을 데이터베이스에 저장"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        saved_count = 0
        errors = []

        try:
            for _, row in mapping_df.iterrows():
                organization = row['조직']
                feature_name = row['Feature 이름']
                description = row.get('설명', '')
                is_active = 1 if row.get('사용여부', 'N') == 'Y' else 0

                try:
                    cursor.execute("""
                        INSERT INTO team_feature_mapping
                        (organization, feature_name, description, is_active, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(organization, feature_name)
                        DO UPDATE SET
                            description = excluded.description,
                            is_active = excluded.is_active,
                            updated_at = excluded.updated_at
                    """, (organization, feature_name, description, is_active, datetime.now()))

                    saved_count += 1

                except Exception as e:
                    errors.append(f"Row {organization}/{feature_name}: {str(e)}")

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
                "error": f"Feature mapping 저장 실패: {str(e)}"
            }
        finally:
            conn.close()

    def save_team_features(self, master_df: pd.DataFrame, active_features: Dict[str, List[str]]) -> Dict[str, Any]:
        """Master Data를 데이터베이스에 저장"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        saved_count = 0
        errors = []

        try:
            for _, row in master_df.iterrows():
                organization = row['조직']
                year = int(row['연도'])
                headcount = int(row['인원']) if pd.notna(row['인원']) else None

                # 해당 조직의 활성 Feature만 추출하여 JSON 구성
                org_features = active_features.get(organization, [])
                feature_values = {}

                for feature in org_features:
                    if feature in row and pd.notna(row[feature]):
                        feature_values[feature] = float(row[feature])

                feature_values_json = json.dumps(feature_values, ensure_ascii=False)

                try:
                    cursor.execute("""
                        INSERT INTO team_features
                        (organization, year, feature_values, headcount, updated_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(organization, year)
                        DO UPDATE SET
                            feature_values = excluded.feature_values,
                            headcount = excluded.headcount,
                            updated_at = excluded.updated_at
                    """, (organization, year, feature_values_json, headcount, datetime.now()))

                    saved_count += 1

                except Exception as e:
                    errors.append(f"Row {organization}/{year}: {str(e)}")

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
                "error": f"Team features 저장 실패: {str(e)}"
            }
        finally:
            conn.close()

    def get_feature_mapping(self, organization: Optional[str] = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """Feature Mapping 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM team_feature_mapping"
        params = []

        conditions = []
        if organization:
            conditions.append("organization = ?")
            params.append(organization)
        if active_only:
            conditions.append("is_active = 1")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY organization, feature_name"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        columns = ['id', 'organization', 'feature_name', 'description', 'is_active', 'created_at', 'updated_at']
        return [dict(zip(columns, row)) for row in rows]

    def get_team_features(self, organization: Optional[str] = None) -> List[Dict[str, Any]]:
        """Team Features 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM team_features"
        params = []

        if organization:
            query += " WHERE organization = ?"
            params.append(organization)

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

    def delete_team_data(self, organization: str) -> Dict[str, Any]:
        """특정 조직의 Team 데이터 삭제"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute("DELETE FROM team_feature_mapping WHERE organization = ?", (organization,))
            mapping_deleted = cursor.rowcount

            cursor.execute("DELETE FROM team_features WHERE organization = ?", (organization,))
            features_deleted = cursor.rowcount

            conn.commit()

            return {
                "success": True,
                "mapping_deleted": mapping_deleted,
                "features_deleted": features_deleted
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
        """Team 데이터 상태 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        # 조직별 통계
        cursor.execute("""
            SELECT organization, COUNT(*) as feature_count
            FROM team_feature_mapping
            WHERE is_active = 1
            GROUP BY organization
        """)
        mapping_stats = {row[0]: row[1] for row in cursor.fetchall()}

        cursor.execute("""
            SELECT organization, COUNT(*) as year_count, MIN(year) as min_year, MAX(year) as max_year
            FROM team_features
            GROUP BY organization
        """)
        feature_stats = {}
        for row in cursor.fetchall():
            feature_stats[row[0]] = {
                'year_count': row[1],
                'year_range': f"{row[2]}-{row[3]}"
            }

        conn.close()

        return {
            "mapping_stats": mapping_stats,
            "feature_stats": feature_stats,
            "organizations": list(set(list(mapping_stats.keys()) + list(feature_stats.keys())))
        }

# Singleton instance
team_service = TeamService()
