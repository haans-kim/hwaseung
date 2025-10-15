"""
Team Features Service V2
조직별 Feature 매핑 및 팀 단위 인력 산정 데이터 처리
실제 템플릿 구조에 맞춰 재작성
"""

import pandas as pd
import sqlite3
import os
import numpy as np
from typing import Dict, Any, Optional, List
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
import logging

class TeamService:
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

    def save_to_database(self, df_master: pd.DataFrame, df_matching: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        """
        Master 데이터와 Feature 정의를 데이터베이스에 저장

        team_features 테이블에 저장:
        - organization (HQ)
        - team (팀)
        - year (년)
        - month (월)
        - position (구분)
        - feature_values (JSON: F1-F9 값들)
        - headcount (인력규모)

        team_feature_definitions 테이블에 저장:
        - company (HQ)
        - team (팀)
        - feature_number (F1, F2, ...)
        - feature_name (실제 메트릭 이름)
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()

        saved_count = 0
        feature_def_count = 0
        errors = []

        try:
            # 1. Feature Definitions 저장
            for _, row in df_matching.iterrows():
                company = row.get('HQ')
                team = row.get('팀')

                if pd.isna(company) or pd.isna(team):
                    continue

                # 기존 feature definitions 삭제
                cursor.execute("""
                    DELETE FROM team_feature_definitions
                    WHERE company = ? AND team = ?
                """, (company, team))

                # 새로운 feature definitions 저장
                for col in feature_cols:
                    feature_name = row.get(col)
                    if pd.notna(feature_name):
                        cursor.execute("""
                            INSERT INTO team_feature_definitions
                            (company, team, feature_number, feature_name)
                            VALUES (?, ?, ?, ?)
                        """, (company, team, col, feature_name))
                        feature_def_count += 1

            conn.commit()
            logging.info(f"✅ Saved {feature_def_count} feature definitions")

        except Exception as e:
            conn.rollback()
            error_msg = f"Error saving feature definitions: {str(e)}"
            errors.append(error_msg)
            logging.error(error_msg)

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
                    # team_features 테이블에 저장
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

                    # team_headcount 테이블에도 저장 (OrganizationSimulation 페이지용)
                    # year를 2자리로 변환 (2025 -> 25)
                    year_short = year % 100

                    # position 변환 ('전체' -> '총합')
                    position_for_headcount = '총합' if position == '전체' else position

                    # 먼저 기존 데이터 삭제
                    cursor.execute("""
                        DELETE FROM team_headcount
                        WHERE team_name = ? AND year = ? AND month = ? AND position = ?
                    """, (team, year_short, month, position_for_headcount))

                    # 새 데이터 삽입
                    cursor.execute("""
                        INSERT INTO team_headcount (
                            team_name, year, month, position, headcount
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        team,
                        year_short,
                        month,
                        position_for_headcount,
                        headcount
                    ))

                    saved_count += 1

                except Exception as e:
                    errors.append(f"Row {organization}/{team}/{year}/{month}/{position}: {str(e)}")

            conn.commit()

            return {
                "success": True,
                "saved_count": saved_count,
                "feature_def_count": feature_def_count,
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

    def train_regression_models(self) -> Dict[str, Any]:
        """
        업로드된 team_features 데이터로 팀별 회귀 모델 학습
        각 팀의 각 직급별로 모델 생성
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            # 팀 목록 조회
            cursor.execute("SELECT DISTINCT company, team FROM team_features")
            teams = cursor.fetchall()

            models_trained = 0
            errors = []

            for company, team in teams:
                try:
                    # 팀 데이터 조회
                    cursor.execute("""
                        SELECT position, feature_values, headcount
                        FROM team_features
                        WHERE company = ? AND team = ?
                    """, (company, team))

                    team_data = cursor.fetchall()
                    if not team_data:
                        continue

                    # 직급별로 모델 학습
                    positions = {'전체': '총', '책임': '책임', '선임': '선임', '사원': '사원'}

                    for position_kr, model_type in positions.items():
                        # 해당 직급 데이터 필터링
                        position_data = [row for row in team_data if row[0] == position_kr]

                        if len(position_data) < 2:  # 최소 2개 데이터 필요
                            continue

                        # Feature 값 추출
                        X_list = []
                        y_list = []

                        for _, feature_json, headcount in position_data:
                            if headcount is None:
                                continue

                            features = json.loads(feature_json) if feature_json else {}
                            # F1~F9 순서로 feature 값 추출
                            feature_values = [features.get(f'F{i}', 0) for i in range(1, 10)]
                            X_list.append(feature_values)
                            y_list.append(headcount)

                        if len(X_list) < 2:
                            continue

                        X = np.array(X_list)
                        y = np.array(y_list)

                        # 선형 회귀 모델 학습
                        model = LinearRegression()
                        model.fit(X, y)

                        # 모델 저장 (기존 모델 삭제 후 새로 저장)
                        cursor.execute("""
                            DELETE FROM regression_models
                            WHERE org_name = ? AND model_type = ?
                        """, (team, model_type))

                        cursor.execute("""
                            INSERT INTO regression_models (org_name, model_type)
                            VALUES (?, ?)
                        """, (team, model_type))

                        model_id = cursor.lastrowid

                        # Feature 정의 조회 (F번호 → 실제 메트릭 이름)
                        cursor.execute("""
                            SELECT feature_number, feature_name
                            FROM team_feature_definitions
                            WHERE company = ? AND team = ?
                            ORDER BY feature_number
                        """, (company, team))

                        feature_map = {row[0]: row[1] for row in cursor.fetchall()}

                        # 계수 저장 (intercept)
                        cursor.execute("""
                            INSERT INTO regression_parameters (model_id, parameter_name, coefficient)
                            VALUES (?, 'intercept', ?)
                        """, (model_id, float(model.intercept_)))

                        # 계수 저장 (F1~F9를 실제 메트릭 이름으로)
                        for i, coef in enumerate(model.coef_, 1):
                            feature_number = f'F{i}'
                            # 실제 feature 이름 사용 (없으면 F번호 사용)
                            param_name = feature_map.get(feature_number, feature_number)

                            cursor.execute("""
                                INSERT INTO regression_parameters (model_id, parameter_name, coefficient)
                                VALUES (?, ?, ?)
                            """, (model_id, param_name, float(coef)))

                        models_trained += 1
                        logging.info(f"✅ Trained model for {team} - {model_type}")

                except Exception as e:
                    error_msg = f"Error training model for {team}: {str(e)}"
                    errors.append(error_msg)
                    logging.error(error_msg)
                    continue

            conn.commit()

            return {
                "success": True,
                "models_trained": models_trained,
                "teams_processed": len(teams),
                "errors": errors if errors else None
            }

        except Exception as e:
            conn.rollback()
            return {
                "success": False,
                "error": f"Model training failed: {str(e)}"
            }
        finally:
            conn.close()

    def calculate_predictions_from_features(self) -> Dict[str, Any]:
        """
        team_features 데이터와 학습된 모델을 사용하여 team_predictions 계산
        가장 최근 데이터(최신 년월)를 사용
        """
        conn = self.get_db_connection()
        cursor = conn.cursor()

        try:
            # 기존 예측값 삭제
            cursor.execute("DELETE FROM team_predictions")
            deleted_count = cursor.rowcount

            # 팀 목록 조회
            cursor.execute("SELECT DISTINCT company, team FROM team_features")
            teams = cursor.fetchall()

            predictions_saved = 0
            errors = []

            for company, team in teams:
                try:
                    # 팀의 가장 최근 데이터 조회 (최신 년월)
                    cursor.execute("""
                        SELECT MAX(year || '-' || printf('%02d', month))
                        FROM team_features
                        WHERE company = ? AND team = ?
                    """, (company, team))

                    latest_date = cursor.fetchone()[0]
                    if not latest_date:
                        continue

                    latest_year, latest_month = latest_date.split('-')

                    # 직급별 예측 계산
                    positions = {'전체': '총', '책임': '책임', '선임': '선임', '사원': '사원'}
                    position_display = {'전체': '총합', '책임': '책임', '선임': '선임', '사원': '사원'}

                    for position_kr, model_type in positions.items():
                        # 모델 조회
                        cursor.execute("""
                            SELECT id FROM regression_models
                            WHERE org_name = ? AND model_type = ?
                            LIMIT 1
                        """, (team, model_type))

                        model_result = cursor.fetchone()
                        if not model_result:
                            errors.append(f"Team {team}, {model_type}: No model found")
                            continue

                        model_id = model_result[0]

                        # 회귀 계수 조회
                        cursor.execute("""
                            SELECT parameter_name, coefficient
                            FROM regression_parameters
                            WHERE model_id = ?
                        """, (model_id,))

                        parameters = {row[0]: row[1] for row in cursor.fetchall()}

                        # 최신 feature 값 조회
                        cursor.execute("""
                            SELECT feature_values, headcount
                            FROM team_features
                            WHERE company = ? AND team = ?
                              AND year = ? AND month = ?
                              AND position = ?
                            LIMIT 1
                        """, (company, team, int(latest_year), int(latest_month), position_kr))

                        feature_result = cursor.fetchone()
                        if not feature_result:
                            continue

                        feature_json, current_headcount = feature_result
                        features = json.loads(feature_json) if feature_json else {}

                        # 예측값 계산
                        prediction = parameters.get('intercept', 0)
                        for i in range(1, 10):
                            feature_name = f'F{i}'
                            if feature_name in parameters:
                                prediction += parameters[feature_name] * features.get(feature_name, 0)

                        predicted_headcount = max(0, round(prediction))
                        current_headcount = int(current_headcount) if current_headcount else 0

                        # 변화량 및 변화율 계산
                        change = predicted_headcount - current_headcount
                        change_percent = (change / current_headcount * 100) if current_headcount > 0 else 0

                        # 카테고리 결정
                        if change_percent > 10:
                            category = '충원필요'
                        elif change_percent < -10:
                            category = '감원검토'
                        else:
                            category = '적정'

                        # team_predictions에 저장
                        cursor.execute("""
                            INSERT INTO team_predictions
                            (team_name, position, current_headcount, predicted_headcount,
                             change, change_percent, category)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (team, position_display[position_kr], current_headcount,
                              predicted_headcount, change, change_percent, category))

                        predictions_saved += 1

                except Exception as e:
                    error_msg = f"Error calculating prediction for {team}: {str(e)}"
                    errors.append(error_msg)
                    logging.error(error_msg)
                    continue

            conn.commit()

            return {
                "success": True,
                "deleted_count": deleted_count,
                "predictions_saved": predictions_saved,
                "teams_processed": len(teams),
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
team_service = TeamService()
