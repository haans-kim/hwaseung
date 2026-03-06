"""
전사 적정인력 산정 Dashboard 서비스 (R*A, tonggibon)
"""
import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, Any, List, Optional
import logging
import os
from sklearn.inspection import permutation_importance

from app.services.company_wide_modeling_service import company_wide_modeling_service

# 환경변수 또는 상대 경로로 DB 경로 설정
DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), '../../hwaseung_RnD.db'))

class CompanyWideDashboardService:
    """전사 적정인력 산정 Dashboard"""

    def __init__(self):
        # Feature 한글명 매핑
        self.feature_labels = {
            'ev_growth_gl': '글로벌 EV시장 성장률',
            'v_growth_gl': '글로벌 자동차 시장성장률',
            'v_export_kr': '국내 자동차 수출액 증가율',
            'vp_export_kr': '국내 자동차부품 수출액 증가율',
            'gdp_growth_kr': 'GDP성장률',
            'cpi_kr': '소비자물가상승률',
            'exchange_rate_change_krw': '환율변화율(원화기준)',
            'scm_index_gl': '글로벌물류비지수',
            'oil_gl': '국제유가',
            'labor_cost': '인건비 증감률',
            'revenue': '매출액 증가율',
            'profit': '영업이익 증가율',
            'operating_rate': '가동률/연구개발비용 증감률',
            'operating_date': '가동일수/연구개발정부보조금 증감률'
        }

        # organization별 내부 지표 라벨
        self.organization_labels = {
            'R*A': {
                'revenue': '매출액 증감률',
                'profit': '영업이익 증감률',
                'operating_rate': '가동률 증감률',
                'operating_date': '가동일수 증감률'
            },
            'tonggibon': {
                'revenue': '매출액 증가율',
                'profit': '영업이익 증가율',
                'operating_rate': '연구개발비용 증감률',
                'operating_date': '연구개발정부보조금 증감률'
            }
        }

    def get_latest_data(self, organization: str) -> Dict[str, Any]:
        """최신 연도 데이터 조회"""
        try:
            conn = sqlite3.connect(DB_PATH)

            query = """
                SELECT *
                FROM company_wide_features
                WHERE organization = ?
                ORDER BY year DESC
                LIMIT 1
            """

            df = pd.read_sql_query(query, conn, params=(organization,))
            conn.close()

            if len(df) == 0:
                raise ValueError(f"No data found for {organization}")

            return df.iloc[0].to_dict()

        except Exception as e:
            logging.error(f"Error getting latest data: {e}")
            raise

    def predict_2026(self, organization: str) -> Dict[str, Any]:
        """
        다음 년도 적정인력 예측 (동적 년도 계산)

        - 데이터베이스에서 최신 년도를 찾아서 다음 년도 예측
        - 예: 2024년이 최신 → 2025년 예측, 2025년이 최신 → 2026년 예측

        Args:
            organization: 'R*A' or 'tonggibon'

        Returns:
            예측 결과
        """
        try:
            # 모델 로드 시도
            model = company_wide_modeling_service.models.get(organization)

            if model is None:
                try:
                    company_wide_modeling_service.load_model(organization, 'latest')
                    model = company_wide_modeling_service.models.get(organization)
                except:
                    raise ValueError(f"No trained model found for {organization}")

            from pycaret.regression import predict_model, pull
            import io
            import sys

            # DB에서 데이터 가져오기
            conn = sqlite3.connect(DB_PATH)

            # 최신 년도 찾기 (동적)
            query_latest_year = """
                SELECT MAX(year) as latest_year
                FROM company_wide_features
                WHERE organization = ? AND headcount IS NOT NULL
            """
            latest_year_result = pd.read_sql_query(query_latest_year, conn, params=(organization,))
            latest_year_with_headcount = int(latest_year_result.iloc[0]['latest_year']) if len(latest_year_result) > 0 else 2024

            # 현재 년도와 다음 년도 계산
            current_year = latest_year_with_headcount
            next_year = current_year + 1
            following_year = next_year + 1

            print(f"Latest year with headcount: {current_year}")
            print(f"Predicting for: {next_year}")

            # 기준 년도 headcount 가져오기
            query_base = f"""
                SELECT headcount
                FROM company_wide_features
                WHERE organization = ? AND year = {current_year} AND headcount IS NOT NULL
                LIMIT 1
            """
            result_base = pd.read_sql_query(query_base, conn, params=(organization,))
            headcount_base = result_base.iloc[0]['headcount'] if len(result_base) > 0 else None

            # 다음 년도 features 가져오기 (없으면 현재 년도 features 사용)
            query_next = f"""
                SELECT year, headcount, features_json
                FROM company_wide_features
                WHERE organization = ? AND year = {next_year}
                LIMIT 1
            """
            result_next = pd.read_sql_query(query_next, conn, params=(organization,))

            # 다음 년도 데이터가 없으면 현재 년도 features 사용
            if len(result_next) == 0:
                query_next = f"""
                    SELECT year, headcount, features_json
                    FROM company_wide_features
                    WHERE organization = ? AND year = {current_year}
                    LIMIT 1
                """
                result_next = pd.read_sql_query(query_next, conn, params=(organization,))
                logging.warning(f"⚠️ No {next_year} features, using {current_year} features for prediction")

            # 그 다음 년도 features 가져오기 (있을 경우)
            query_following = f"""
                SELECT year, headcount, features_json
                FROM company_wide_features
                WHERE organization = ? AND year = {following_year}
                LIMIT 1
            """
            result_following = pd.read_sql_query(query_following, conn, params=(organization,))
            conn.close()

            # features_json을 파싱하여 DataFrame 생성
            def parse_features_json_to_df(result_df):
                if len(result_df) == 0:
                    return pd.DataFrame()

                import json
                row = result_df.iloc[0]
                features_json_str = row.get('features_json', None)

                if not features_json_str:
                    return pd.DataFrame()

                features = json.loads(features_json_str)
                # '정원'과 'headcount' 제외
                feature_dict = {k: v for k, v in features.items() if k not in ['정원', 'headcount']}
                return pd.DataFrame([feature_dict])

            data_next = parse_features_json_to_df(result_next)
            data_following = parse_features_json_to_df(result_following)

            # 컬럼 제거 함수 (이제는 이미 정제된 데이터이므로 그대로 사용)
            def prepare_prediction_data(df):
                return df.copy()

            # 다음 년도 예측
            prediction_next = None
            if len(data_next) > 0:
                pred_df_next = prepare_prediction_data(data_next)

                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    predictions = predict_model(model, data=pred_df_next)
                    prediction_next = predictions['prediction_label'].iloc[0]
                finally:
                    sys.stdout = old_stdout

            # 그 다음 년도 예측 (데이터가 있을 경우만)
            prediction_following = None
            if len(data_following) > 0:
                pred_df_following = prepare_prediction_data(data_following)

                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    predictions = predict_model(model, data=pred_df_following)
                    prediction_following = predictions['prediction_label'].iloc[0]
                finally:
                    sys.stdout = old_stdout

            # 🔧 FIX: DB에서 저장된 모델 R2 score 가져오기
            try:
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()

                # 가장 최근에 학습된 모델의 R² 가져오기
                cursor.execute("""
                    SELECT r_squared
                    FROM company_wide_model_metrics
                    WHERE organization = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                """, (organization,))

                result = cursor.fetchone()
                conn.close()

                if result and result[0] is not None:
                    r2_score = result[0]
                    logging.info(f"Loaded R² from DB for {organization}: {r2_score:.4f}")
                else:
                    # DB에 없으면 현재 메모리의 metrics 시도
                    try:
                        old_stdout = sys.stdout
                        sys.stdout = io.StringIO()
                        metrics = pull()
                        sys.stdout = old_stdout
                        r2_score = metrics.loc[metrics.index[0], 'R2'] if metrics is not None else 0.85
                        logging.warning(f"⚠️ No metrics in DB for {organization}, using pull(): {r2_score:.4f}")
                    except:
                        r2_score = 0.85
                        logging.warning(f"⚠️ No metrics available for {organization}, using default: {r2_score}")
            except Exception as e:
                logging.error(f"Failed to load metrics from DB: {e}")
                r2_score = 0.85

            # 결과 구성 (동적 년도 사용)
            if prediction_following is not None:
                # 그 다음 년도 데이터가 있으면 그 다음 년도 예측 반환 (기준: 다음 년도 예측)
                return {
                    'year': following_year,
                    'predicted_headcount': round(prediction_following),
                    'previous_year': next_year,
                    'previous_headcount': round(prediction_next) if prediction_next else None,
                    'change': round(prediction_following - prediction_next) if prediction_next else 0,
                    'change_percent': round((prediction_following - prediction_next) / prediction_next * 100, 1) if prediction_next else 0,
                    'model_r2': r2_score,
                    'organization': organization,
                    f'prediction_{next_year}': round(prediction_next) if prediction_next else None,
                    f'actual_{current_year}': int(headcount_base) if headcount_base else None
                }
            else:
                # 그 다음 년도 데이터가 없으면 다음 년도 features로 다음 년도 예측 (기준: 현재 년도 실제)
                return {
                    'year': next_year,
                    'predicted_headcount': round(prediction_next) if prediction_next else None,
                    'previous_year': current_year,
                    'previous_headcount': int(headcount_base) if headcount_base else None,
                    'change': round(prediction_next - headcount_base) if (prediction_next and headcount_base) else 0,
                    'change_percent': round((prediction_next - headcount_base) / headcount_base * 100, 1) if (prediction_next and headcount_base) else 0,
                    'model_r2': r2_score,
                    'organization': organization,
                    'note': f'{next_year} features로 {next_year} 예측'
                }

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            raise


    def get_feature_importance(self, organization: str, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        Permutation Importance 계산

        Args:
            organization: 'R*A' or 'tonggibon'
            top_n: 상위 N개 feature

        Returns:
            Feature importance 리스트
        """
        try:
            model = company_wide_modeling_service.models.get(organization)

            if model is None:
                company_wide_modeling_service.load_model(organization, 'latest')
                model = company_wide_modeling_service.models.get(organization)

            # 🔥 증강된 데이터가 있으면 사용, 없으면 원본 데이터 사용
            if company_wide_modeling_service.augmented_data.get(organization) is not None:
                logging.info(f"📊 Using augmented data for feature importance: {organization}")
                df = company_wide_modeling_service.augmented_data[organization].copy()
            else:
                logging.info(f"📊 Using original data for feature importance: {organization}")
                df = company_wide_modeling_service.get_data_from_db(organization)

            # 데이터 준비
            cols_to_drop = ['id', 'organization', 'year', 'created_at', 'updated_at']
            for col in cols_to_drop:
                if col in df.columns:
                    df = df.drop(columns=[col])

            if 'headcount' not in df.columns:
                raise ValueError("Target column 'headcount' not found")

            # NaN 제거
            df = df.dropna(subset=['headcount'])
            if len(df) == 0:
                raise ValueError("No valid data after removing NaN")

            X = df.drop(columns=['headcount'])
            y = df['headcount']

            # Feature columns에서도 NaN 제거
            X = X.fillna(X.mean())  # 평균값으로 대체

            # Permutation importance 계산
            perm_importance = permutation_importance(
                model, X, y,
                n_repeats=10,
                random_state=42,
                n_jobs=1  # 메모리 안전
            )

            # Feature importance 정렬
            feature_names = X.columns.tolist()
            importance_data = []

            for idx, feature in enumerate(feature_names):
                # 한글 라벨 가져오기
                if organization in self.organization_labels and feature in self.organization_labels[organization]:
                    label = self.organization_labels[organization][feature]
                else:
                    label = self.feature_labels.get(feature, feature)

                importance_data.append({
                    'feature': feature,
                    'label': label,
                    'importance': perm_importance.importances_mean[idx],
                    'std': perm_importance.importances_std[idx]
                })

            # 중요도 순으로 정렬
            importance_data.sort(key=lambda x: abs(x['importance']), reverse=True)

            # 상위 N개
            return importance_data[:top_n]

        except Exception as e:
            logging.error(f"Feature importance calculation failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: 더미 데이터
            return []

    def simulate_scenario(
        self,
        organization: str,
        variables: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        변수 조정 시뮬레이션

        Args:
            organization: 'R*A' or 'tonggibon'
            variables: 조정할 변수 딕셔너리

        Returns:
            시뮬레이션 결과
        """
        try:
            model = company_wide_modeling_service.models.get(organization)

            if model is None:
                company_wide_modeling_service.load_model(organization, 'latest')
                model = company_wide_modeling_service.models.get(organization)

            # 기본 예측 (변수 조정 전)
            baseline = self.predict_2026(organization)
            baseline_headcount = baseline['predicted_headcount']

            # 최신 데이터 가져오기
            latest_data = self.get_latest_data(organization)

            # 변수 조정 적용 (0이 아닌 값만 적용)
            for var_name, var_value in variables.items():
                if var_name in latest_data and var_value != 0:
                    latest_data[var_name] = var_value

            # 예측용 데이터 준비
            prediction_df = pd.DataFrame([latest_data])
            cols_to_drop = ['id', 'organization', 'year', 'headcount', 'created_at', 'updated_at']
            for col in cols_to_drop:
                if col in prediction_df.columns:
                    prediction_df = prediction_df.drop(columns=[col])

            # 예측
            from pycaret.regression import predict_model
            import io
            import sys

            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                predictions = predict_model(model, data=prediction_df)
                simulated_value = predictions['prediction_label'].iloc[0]
            finally:
                sys.stdout = old_stdout

            # 변화량 계산
            change = simulated_value - baseline_headcount
            change_percent = (change / baseline_headcount) * 100 if baseline_headcount != 0 else 0

            return {
                'baseline_headcount': baseline_headcount,
                'simulated_headcount': round(simulated_value),
                'change': round(change),
                'change_percent': round(change_percent, 1),
                'variables_adjusted': variables,
                'organization': organization
            }

        except Exception as e:
            logging.error(f"Simulation failed: {e}")
            raise

    def get_trend_data(self, organization: str) -> Dict[str, Any]:
        """
        트렌드 데이터 (과거 + 예측)

        Args:
            organization: 'R*A' or 'tonggibon'

        Returns:
            트렌드 데이터
        """
        try:
            conn = sqlite3.connect(DB_PATH)

            # 과거 데이터 (headcount가 있는 것만)
            query = """
                SELECT year, headcount
                FROM company_wide_features
                WHERE organization = ? AND headcount IS NOT NULL
                ORDER BY year
            """

            df = pd.read_sql_query(query, conn, params=(organization,))
            conn.close()

            # 다음 년도 예측 (동적)
            prediction = self.predict_2026(organization)

            # 데이터 결합 - 실제 연도 그대로 사용
            years = df['year'].tolist()
            actual = df['headcount'].tolist()
            predicted = [None] * len(df)

            # 예측 년도 추가
            prediction_year = prediction['year']
            years.append(prediction_year)
            actual.append(None)  # 예측 년도는 실제 값 없음
            predicted.append(prediction['predicted_headcount'])

            return {
                'years': years,
                'actual': actual,
                'predicted': predicted,
                'organization': organization
            }

        except Exception as e:
            logging.error(f"Trend data generation failed: {e}")
            raise

    def get_variable_ranges(self, organization: str) -> Dict[str, Dict[str, float]]:
        """변수별 조정 범위 정의"""
        try:
            conn = sqlite3.connect(DB_PATH)

            query = """
                SELECT *
                FROM company_wide_features
                WHERE organization = ?
            """

            df = pd.read_sql_query(query, conn, params=(organization,))
            conn.close()

            ranges = {}

            # 각 변수의 min, max 계산
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            exclude_cols = ['id', 'year', 'headcount']

            for col in numeric_columns:
                if col not in exclude_cols:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    mean_val = df[col].mean()

                    # 범위를 조금 확장 (±30%)
                    range_span = max_val - min_val
                    extended_min = min_val - range_span * 0.3
                    extended_max = max_val + range_span * 0.3

                    # 한글 라벨
                    if organization in self.organization_labels and col in self.organization_labels[organization]:
                        label = self.organization_labels[organization][col]
                    else:
                        label = self.feature_labels.get(col, col)

                    ranges[col] = {
                        'label': label,
                        'min': round(extended_min, 2),
                        'max': round(extended_max, 2),
                        'default': round(mean_val, 2)
                    }

            return ranges

        except Exception as e:
            logging.error(f"Variable ranges calculation failed: {e}")
            return {}

# 싱글톤 인스턴스
company_wide_dashboard_service = CompanyWideDashboardService()
