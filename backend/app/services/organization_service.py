"""
Organization Service
조직도 데이터 처리 (계층 구조: 회사 → 본부 → 담당/사업단/센터 → 실 → 팀)
"""

import pandas as pd
import sqlite3
from typing import Dict, Any, List
from datetime import datetime

class OrganizationService:
    def __init__(self, db_path: str = "/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db"):
        self.db_path = db_path

    def get_db_connection(self):
        """데이터베이스 연결"""
        return sqlite3.connect(self.db_path)

    def validate_excel_file(self, file_path: str) -> Dict[str, Any]:
        """
        조직도 Excel 파일 검증

        Expected columns:
        - 회사 (Company/HQ)
        - 본부 (Division)
        - 담당/사업단/센터 (Department/Business Unit/Center)
        - 실 (Office)
        - 팀 (Team)
        - 비고 (Note) - optional
        """
        try:
            # Excel 파일 읽기 (첫 번째 시트)
            df = pd.read_excel(file_path)

            # 컬럼명 정규화 (공백 제거)
            df.columns = df.columns.str.strip()

            # 필수 컬럼 확인
            required_cols = ['회사', '본부', '담당/사업단/센터', '실', '팀']

            # 다양한 컬럼명 변형 지원
            column_mapping = {
                '회사': ['회사', 'HQ', '회사명'],
                '본부': ['본부', '사업본부'],
                '담당/사업단/센터': ['담당/사업단/센터', '담당_사업단_센터', '담당', '사업단', '센터'],
                '실': ['실', '실명'],
                '팀': ['팀', '팀명']
            }

            # 실제 컬럼명 매핑
            actual_columns = {}
            for required_col in required_cols:
                found = False
                for possible_name in column_mapping.get(required_col, [required_col]):
                    if possible_name in df.columns:
                        actual_columns[required_col] = possible_name
                        found = True
                        break

                if not found:
                    return {
                        "valid": False,
                        "error": f"필수 컬럼을 찾을 수 없습니다: {required_col}"
                    }

            # 컬럼명 표준화
            rename_dict = {v: k for k, v in actual_columns.items()}
            df = df.rename(columns=rename_dict)

            # 비고 컬럼이 없으면 추가
            if '비고' not in df.columns:
                df['비고'] = None

            # 데이터 검증
            if len(df) == 0:
                return {
                    "valid": False,
                    "error": "데이터가 없습니다"
                }

            # 회사명 확인
            companies = df['회사'].dropna().unique().tolist()
            if len(companies) == 0:
                return {
                    "valid": False,
                    "error": "'회사' 컬럼에 값이 없습니다"
                }

            # 팀명 확인
            teams = df['팀'].dropna().unique().tolist()
            if len(teams) == 0:
                return {
                    "valid": False,
                    "error": "'팀' 컬럼에 값이 없습니다"
                }

            # 통계 정보
            divisions = df['본부'].dropna().unique().tolist()
            departments = df['담당/사업단/센터'].dropna().unique().tolist()
            offices = df['실'].dropna().unique().tolist()

            return {
                "valid": True,
                "dataframe": df[['회사', '본부', '담당/사업단/센터', '실', '팀', '비고']],
                "row_count": len(df),
                "companies": companies,
                "company_count": len(companies),
                "division_count": len(divisions),
                "department_count": len(departments),
                "office_count": len(offices),
                "team_count": len(teams)
            }

        except Exception as e:
            return {
                "valid": False,
                "error": f"파일 검증 중 오류 발생: {str(e)}"
            }

    def save_to_database(self, df: pd.DataFrame, replace_all: bool = True) -> Dict[str, Any]:
        """
        조직도 데이터를 데이터베이스에 저장

        Args:
            df: 조직도 DataFrame
            replace_all: True면 기존 데이터 삭제 후 저장, False면 UPSERT
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            # 기존 데이터 삭제 (replace_all=True인 경우)
            if replace_all:
                cursor.execute("DELETE FROM organization")
                deleted_count = cursor.rowcount
            else:
                deleted_count = 0

            saved_count = 0
            errors = []

            for _, row in df.iterrows():
                try:
                    # 담당/사업단/센터는 담당_사업단_센터로 변환
                    dept_value = row.get('담당/사업단/센터', None)

                    cursor.execute("""
                        INSERT INTO organization (
                            회사, 본부, 담당_사업단_센터, 실, 팀, 비고, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('회사', None),
                        row.get('본부', None),
                        dept_value,
                        row.get('실', None),
                        row.get('팀', None),
                        row.get('비고', None),
                        datetime.now().isoformat(),
                        datetime.now().isoformat()
                    ))

                    saved_count += 1

                except Exception as e:
                    team_name = row.get('팀', 'Unknown')
                    errors.append(f"Row {team_name}: {str(e)}")

            conn.commit()

            return {
                "success": True,
                "deleted_count": deleted_count if replace_all else 0,
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

    def get_organization_data(self, company: str = None) -> List[Dict[str, Any]]:
        """조직도 데이터 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM organization"
        params = []

        if company:
            query += " WHERE 회사 = ?"
            params.append(company)

        query += " ORDER BY 회사, 본부, 담당_사업단_센터, 실, 팀"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        columns = ['id', '회사', '본부', '담당_사업단_센터', '실', '팀', '비고', 'created_at', 'updated_at']
        return [dict(zip(columns, row)) for row in rows]

    def get_hierarchy_tree(self, company: str = None) -> Dict[str, Any]:
        """조직도 계층 구조를 트리 형태로 반환"""
        data = self.get_organization_data(company)

        # 회사별로 그룹화
        tree = {}

        for row in data:
            company_name = row['회사'] or '미분류'
            division = row['본부'] or '미분류'
            department = row['담당_사업단_센터'] or '미분류'
            office = row['실'] or '미분류'
            team = row['팀'] or '미분류'

            # 계층 구조 생성
            if company_name not in tree:
                tree[company_name] = {}

            if division not in tree[company_name]:
                tree[company_name][division] = {}

            if department not in tree[company_name][division]:
                tree[company_name][division][department] = {}

            if office not in tree[company_name][division][department]:
                tree[company_name][division][department][office] = []

            tree[company_name][division][department][office].append(team)

        return tree

    def get_status(self) -> Dict[str, Any]:
        """조직도 데이터 상태 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        # 전체 통계
        cursor.execute("SELECT COUNT(*) FROM organization")
        total_count = cursor.fetchone()[0]

        # 회사별 통계
        cursor.execute("""
            SELECT
                회사,
                COUNT(DISTINCT 본부) as division_count,
                COUNT(DISTINCT 담당_사업단_센터) as department_count,
                COUNT(DISTINCT 실) as office_count,
                COUNT(DISTINCT 팀) as team_count
            FROM organization
            WHERE 회사 IS NOT NULL
            GROUP BY 회사
        """)

        company_stats = {}
        for row in cursor.fetchall():
            company_stats[row[0]] = {
                'division_count': row[1],
                'department_count': row[2],
                'office_count': row[3],
                'team_count': row[4]
            }

        # 회사 목록
        cursor.execute("SELECT DISTINCT 회사 FROM organization WHERE 회사 IS NOT NULL ORDER BY 회사")
        companies = [row[0] for row in cursor.fetchall()]

        conn.close()

        return {
            "total_count": total_count,
            "company_stats": company_stats,
            "companies": companies,
            "company_count": len(companies)
        }

# Singleton instance
organization_service = OrganizationService()
