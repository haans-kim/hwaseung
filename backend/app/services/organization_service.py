"""
Organization Service
조직도 데이터 처리 (계층 구조: 회사 → 본부 → 담당/사업단/센터 → 실 → 팀)
"""

import pandas as pd
import sqlite3
import os
from typing import Dict, Any, List
from datetime import datetime

class OrganizationService:
    def __init__(self, db_path: str = None):
        # 🔧 FIX: Docker 환경을 위해 환경 변수 사용
        if db_path is None:
            db_path = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), '../../hwaseung_RnD.db'))
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
        - F1~F9 (Feature 정의) - optional
        """
        try:
            # Excel 파일 읽기 (첫 번째 시트 - Feature Matching)
            df = pd.read_excel(file_path, sheet_name=0)

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

            # Feature 컬럼 추출 (F1~F9)
            feature_cols = [col for col in df.columns if col.startswith('F') and len(col) == 2 and col[1].isdigit()]

            # 조직 데이터만 추출
            org_cols = ['회사', '본부', '담당/사업단/센터', '실', '팀', '비고']
            org_df = df[org_cols].copy()

            # Feature 정의가 있는 데이터 추출
            feature_df = None
            if feature_cols:
                # Feature 정의가 하나라도 있는 행만 추출
                has_features = df[feature_cols].notna().any(axis=1)
                if has_features.any():
                    feature_df = df[has_features][['회사', '팀'] + feature_cols].copy()

            return {
                "valid": True,
                "dataframe": org_df,
                "feature_dataframe": feature_df,
                "feature_columns": feature_cols,
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

    def save_feature_definitions(self, feature_df: pd.DataFrame, replace_all: bool = True) -> Dict[str, Any]:
        """
        Feature 정의를 데이터베이스에 저장

        Args:
            feature_df: Feature 정의 DataFrame (회사, 팀, F1~F9 컬럼 포함)
            replace_all: True면 기존 Feature 정의 삭제 후 저장
        """
        if feature_df is None or len(feature_df) == 0:
            return {
                "success": True,
                "deleted_count": 0,
                "saved_count": 0,
                "message": "No feature definitions to save"
            }

        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            deleted_count = 0

            # replace_all=True인 경우 기존 데이터 삭제
            if replace_all:
                cursor.execute("DELETE FROM team_feature_definitions")
                deleted_count = cursor.rowcount

            saved_count = 0
            errors = []

            # Feature 컬럼 추출
            feature_cols = [col for col in feature_df.columns if col.startswith('F') and len(col) == 2 and col[1].isdigit()]

            for _, row in feature_df.iterrows():
                company = row.get('회사', None)
                team = row.get('팀', None)

                if not company or not team:
                    continue

                # 각 Feature별로 저장
                for feature_col in feature_cols:
                    feature_name = row.get(feature_col, None)

                    # Feature 정의가 있는 경우만 저장
                    if pd.notna(feature_name) and str(feature_name).strip():
                        try:
                            cursor.execute("""
                                INSERT OR REPLACE INTO team_feature_definitions (
                                    company, team, feature_number, feature_name,
                                    created_at, updated_at
                                ) VALUES (?, ?, ?, ?, ?, ?)
                            """, (
                                company,
                                team,
                                feature_col,
                                str(feature_name).strip(),
                                datetime.now().isoformat(),
                                datetime.now().isoformat()
                            ))
                            saved_count += 1
                        except Exception as e:
                            errors.append(f"Team {team}, {feature_col}: {str(e)}")

            conn.commit()

            return {
                "success": True,
                "deleted_count": deleted_count,
                "saved_count": saved_count,
                "errors": errors if errors else None
            }

        except Exception as e:
            conn.rollback()
            return {
                "success": False,
                "error": f"Feature 정의 저장 실패: {str(e)}"
            }
        finally:
            conn.close()

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

    def get_feature_definitions(self, team: str = None, company: str = None) -> List[Dict[str, Any]]:
        """Feature 정의 조회"""
        conn = self.get_db_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM team_feature_definitions"
        params = []
        conditions = []

        if team:
            conditions.append("team = ?")
            params.append(team)

        if company:
            conditions.append("company = ?")
            params.append(company)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY company, team, feature_number"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        columns = ['id', 'company', 'team', 'feature_number', 'feature_name', 'created_at', 'updated_at']
        return [dict(zip(columns, row)) for row in rows]

    def get_analysis_ready_teams(self) -> List[Dict[str, Any]]:
        """
        분석가능팀 조회
        조건: Feature 정의가 있고 회귀모델이 있는 팀
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()

        query = """
            SELECT DISTINCT
                tfd.company,
                tfd.team,
                COUNT(DISTINCT tfd.feature_number) as feature_count,
                COUNT(DISTINCT rm.model_type) as model_count
            FROM team_feature_definitions tfd
            INNER JOIN regression_models rm ON tfd.team = rm.org_name
            GROUP BY tfd.company, tfd.team
            HAVING feature_count > 0 AND model_count > 0
            ORDER BY tfd.company, tfd.team
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        teams = []
        for row in rows:
            teams.append({
                'company': row[0],
                'team': row[1],
                'feature_count': row[2],
                'model_count': row[3]
            })

        return teams

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

    def process_master_sheet(self, file_path: str, replace_all: bool = True) -> Dict[str, Any]:
        """
        Master 시트 데이터를 처리하여 team_metrics와 team_headcount 테이블에 저장

        Master 시트 컬럼:
        - HQ (회사)
        - 팀
        - 년, 월
        - 구분 (전체, 선임, 책임, 사원)
        - F1~F9 (Feature 값)
        - 인력규모
        """
        try:
            # Master 시트 읽기
            master_df = pd.read_excel(file_path, sheet_name='master')

            if master_df is None or len(master_df) == 0:
                return {
                    "success": True,
                    "message": "No master data to process",
                    "metrics_saved": 0,
                    "headcount_saved": 0
                }

            conn = self.get_db_connection()
            cursor = conn.cursor()

            try:
                # replace_all=True인 경우 기존 데이터 삭제
                metrics_deleted = 0
                headcount_deleted = 0

                if replace_all:
                    cursor.execute("DELETE FROM team_metrics")
                    metrics_deleted = cursor.rowcount
                    cursor.execute("DELETE FROM team_headcount")
                    headcount_deleted = cursor.rowcount

                metrics_saved = 0
                headcount_saved = 0
                errors = []

                # Feature 컬럼 추출 (F1~F9)
                feature_cols = [col for col in master_df.columns if col.startswith('F') and len(col) == 2 and col[1].isdigit()]

                for _, row in master_df.iterrows():
                    team_name = row.get('팀', None)
                    year = row.get('년', None)
                    month = row.get('월', None)
                    position = row.get('구분', None)
                    headcount = row.get('인력규모', None)

                    if not team_name or pd.isna(year) or pd.isna(month) or not position:
                        continue

                    # year를 2자리로 변환 (2025 -> 25)
                    if year >= 2000:
                        year = year - 2000

                    # 구분 매핑 (전체 -> 총합)
                    if position == '전체':
                        position = '총합'

                    # team_headcount에 인력규모 저장
                    if pd.notna(headcount):
                        try:
                            cursor.execute("""
                                INSERT OR REPLACE INTO team_headcount (
                                    team_name, year, month, position, headcount
                                ) VALUES (?, ?, ?, ?, ?)
                            """, (
                                team_name,
                                int(year),
                                int(month),
                                position,
                                int(headcount)
                            ))
                            headcount_saved += 1
                        except Exception as e:
                            errors.append(f"Headcount save error - Team: {team_name}, Year: {year}, Month: {month}, Position: {position}: {str(e)}")

                    # team_metrics에 Feature 값들 저장
                    for feature_col in feature_cols:
                        feature_value = row.get(feature_col, None)

                        if pd.notna(feature_value):
                            try:
                                cursor.execute("""
                                    INSERT OR REPLACE INTO team_metrics (
                                        team_name, year, month, metric_name, metric_value
                                    ) VALUES (?, ?, ?, ?, ?)
                                """, (
                                    team_name,
                                    int(year),
                                    int(month),
                                    feature_col,
                                    float(feature_value)
                                ))
                                metrics_saved += 1
                            except Exception as e:
                                errors.append(f"Metrics save error - Team: {team_name}, Feature: {feature_col}: {str(e)}")

                conn.commit()

                return {
                    "success": True,
                    "metrics_deleted": metrics_deleted,
                    "metrics_saved": metrics_saved,
                    "headcount_deleted": headcount_deleted,
                    "headcount_saved": headcount_saved,
                    "errors": errors if errors else None
                }

            except Exception as e:
                conn.rollback()
                return {
                    "success": False,
                    "error": f"Master 데이터 저장 실패: {str(e)}"
                }
            finally:
                conn.close()

        except Exception as e:
            return {
                "success": False,
                "error": f"Master 시트 처리 실패: {str(e)}"
            }

    def calculate_team_predictions(self, replace_all: bool = True) -> Dict[str, Any]:
        """
        회귀 모델을 사용하여 팀별 예측값 계산 및 저장

        조건: Feature 정의 + 회귀모델 + team_metrics + team_headcount 모두 있는 팀
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            # replace_all=True인 경우 기존 예측값 삭제
            deleted_count = 0
            if replace_all:
                cursor.execute("DELETE FROM team_predictions")
                deleted_count = cursor.rowcount

            # 분석가능팀 조회 (Feature 정의 + 회귀모델)
            analysis_ready_teams = self.get_analysis_ready_teams()

            if not analysis_ready_teams:
                return {
                    "success": True,
                    "message": "No teams available for prediction",
                    "deleted_count": deleted_count,
                    "predictions_saved": 0
                }

            predictions_saved = 0
            errors = []
            team_results = []

            for team_info in analysis_ready_teams:
                team_name = team_info['team']

                try:
                    # 팀 메트릭 평균값 조회
                    cursor.execute("""
                        SELECT metric_name, AVG(metric_value) as avg_value
                        FROM team_metrics
                        WHERE team_name = ?
                        GROUP BY metric_name
                    """, (team_name,))

                    metrics = {row[0]: row[1] for row in cursor.fetchall()}

                    if not metrics:
                        errors.append(f"Team {team_name}: No metrics data found")
                        continue

                    # 4개 모델 타입별 예측 (총, 책임, 선임, 사원)
                    model_types = ['총', '책임', '선임', '사원']
                    position_map = {'총': '총합', '책임': '책임', '선임': '선임', '사원': '사원'}

                    for model_type in model_types:
                        # 모델 조회
                        cursor.execute("""
                            SELECT id FROM regression_models
                            WHERE org_name = ? AND model_type = ?
                            LIMIT 1
                        """, (team_name, model_type))

                        model_result = cursor.fetchone()
                        if not model_result:
                            errors.append(f"Team {team_name}, {model_type}: No model found")
                            continue

                        model_id = model_result[0]

                        # 회귀 계수 조회
                        cursor.execute("""
                            SELECT parameter_name, coefficient
                            FROM regression_parameters
                            WHERE model_id = ?
                        """, (model_id,))

                        parameters = cursor.fetchall()

                        # 예측값 계산
                        prediction = 0
                        for param_name, coefficient in parameters:
                            if param_name == 'intercept':
                                prediction += coefficient
                            elif param_name in metrics:
                                prediction += coefficient * metrics[param_name]

                        predicted_headcount = max(0, round(prediction))

                        # 현재 인원 조회
                        cursor.execute("""
                            SELECT headcount FROM team_headcount
                            WHERE team_name = ? AND year = 25 AND month = 8 AND position = ?
                            LIMIT 1
                        """, (team_name, position_map[model_type]))

                        current_result = cursor.fetchone()
                        current_headcount = int(current_result[0]) if current_result else 0

                        # 변화량 및 변화율 계산
                        change = predicted_headcount - current_headcount
                        change_percent = (change / current_headcount * 100) if current_headcount > 0 else 0

                        # 분류 결정
                        if change > 0:
                            category = '충원필요'
                        elif change < 0:
                            category = '감원검토'
                        else:
                            category = '적정'

                        # DB에 저장
                        cursor.execute("""
                            INSERT OR REPLACE INTO team_predictions
                            (team_name, position, current_headcount, predicted_headcount, change, change_percent, category)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (team_name, model_type, current_headcount, predicted_headcount, change, change_percent, category))

                        predictions_saved += 1

                except Exception as e:
                    errors.append(f"Team {team_name} prediction error: {str(e)}")
                    continue

            conn.commit()

            return {
                "success": True,
                "deleted_count": deleted_count,
                "predictions_saved": predictions_saved,
                "teams_analyzed": len(analysis_ready_teams),
                "errors": errors if errors else None
            }

        except Exception as e:
            conn.rollback()
            return {
                "success": False,
                "error": f"Prediction calculation failed: {str(e)}"
            }
        finally:
            conn.close()

# Singleton instance
organization_service = OrganizationService()
