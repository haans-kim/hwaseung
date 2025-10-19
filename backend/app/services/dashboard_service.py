import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.services.data_service import data_service

class DashboardService:
    def __init__(self):
        self.scenario_templates = {
            "base": {
                "name": "기본 시나리오",
                "description": "현재 경제 상황 기준",
                "variables": {
                    "oil_gl": -13.7,
                    "exchange_rate_change_krw": 4.2,
                    "vp_export_kr": -0.14,
                    "cpi_kr": 1.8,
                    "v_export_kr": 20.0
                }
            },
            "optimistic": {
                "name": "낙관적 시나리오",
                "description": "고유가 + 수출 증가",
                "variables": {
                    "oil_gl": 30.0,
                    "exchange_rate_change_krw": 10.0,
                    "vp_export_kr": 20.0,
                    "cpi_kr": 5.0,
                    "v_export_kr": 25.0
                }
            },
            "moderate": {
                "name": "중립적 시나리오",
                "description": "안정적 경제 성장",
                "variables": {
                    "oil_gl": 0.0,
                    "exchange_rate_change_krw": 2.0,
                    "vp_export_kr": 5.0,
                    "cpi_kr": 3.0,
                    "v_export_kr": 10.0
                }
            },
            "pessimistic": {
                "name": "비관적 시나리오",
                "description": "저유가 + 수출 감소",
                "variables": {
                    "oil_gl": -30.0,
                    "exchange_rate_change_krw": -10.0,
                    "vp_export_kr": -20.0,
                    "cpi_kr": 0.5,
                    "v_export_kr": -15.0
                }
            }
        }

        # 🔧 FIX: Lazy loading - 모델이 학습된 후에 호출되도록 변경
        self._variable_definitions_cache = None

    @property
    def variable_definitions(self) -> Dict[str, Dict[str, Any]]:
        """변수 정의 - 최초 접근 시에만 빌드 (Lazy Loading)"""
        if self._variable_definitions_cache is None:
            self._variable_definitions_cache = self._build_variable_definitions()
        return self._variable_definitions_cache

    def _build_variable_definitions(self) -> Dict[str, Dict[str, Any]]:
        """기반 Importance와 실제 데이터를 기반으로 변수 정의 동적 생성"""
        try:
            # 1. Feature importance 가져오기
            top_features = self._get_top_features()
            
            # 2. 2025년 실제 데이터 가져오기
            actual_values = self._get_2025_actual_data()
            
            # 3. 변수 정의 생성
            variable_defs = {}
            
            # 변수별 메타 정보 매핑 (상위 10개 모두 포함)
            feature_meta = {
                'operating_income': {
                    'name': '영업이익 증가율',
                    'description': '영업이익 증가율 (%)',
                    'min_value': -30.0, 'max_value': 50.0, 'unit': '%'
                },
                'exchange_rate_change_krw': {
                    'name': '환율 변동률',
                    'description': '원달러 환율 변동률 (%)',
                    'min_value': -15.0, 'max_value': 20.0, 'unit': '%'
                },
                'oil_gl': {
                    'name': '글로벌 유가',
                    'description': '국제 유가 변동률 (%)',
                    'min_value': -50.0, 'max_value': 50.0, 'unit': '%'
                },
                'gdp_growth_kr': {
                    'name': 'GDP 성장률',
                    'description': '한국 GDP 성장률 (%)',
                    'min_value': -5.0, 'max_value': 8.0, 'unit': '%'
                },
                'scm_index_gl': {
                    'name': '공급망 지수',
                    'description': '글로벌 공급망 지수',
                    'min_value': 500, 'max_value': 2000, 'unit': ''
                },
                'ev_growth_gl': {
                    'name': '글로벌 EV시장 성장률',
                    'description': '글로벌 전기차 시장 성장률 (%)',
                    'min_value': -15.0, 'max_value': 25.0, 'unit': '%'
                },
                'vp_export_kr': {
                    'name': '자동차부품 수출 증가율',
                    'description': '한국 자동차부품 수출액 증가율 (%)',
                    'min_value': -30.0, 'max_value': 30.0, 'unit': '%'
                },
                'v_growth_gl': {
                    'name': '글로벌 자동차시장 성장률',
                    'description': '글로벌 자동차 시장 성장률 (%)',
                    'min_value': -20.0, 'max_value': 30.0, 'unit': '%'
                },
                'v_export_kr': {
                    'name': '자동차 수출 증가율',
                    'description': '한국 자동차 수출액 증가율 (%)',
                    'min_value': -25.0, 'max_value': 25.0, 'unit': '%'
                },
                'revenue': {
                    'name': '매출액 증가율',
                    'description': '기업 매출액 증가율 (%)',
                    'min_value': -30.0, 'max_value': 50.0, 'unit': '%'
                },
                'cpi_kr': {
                    'name': '소비자물가지수',
                    'description': '한국 소비자물가지수 증가율 (%)',
                    'min_value': -2.0, 'max_value': 8.0, 'unit': '%'
                },
                'production_capa': {
                    'name': '생산 능력',
                    'description': '생산 능력 지수',
                    'min_value': 0.5, 'max_value': 2.0, 'unit': ''
                },
                'labor_costs': {
                    'name': '인건비 증가율',
                    'description': '총 인건비 증가율 (%)',
                    'min_value': -20.0, 'max_value': 30.0, 'unit': '%'
                },
                'rnd_cost': {
                    'name': 'R&D 투자 증가율',
                    'description': '연구개발비 증가율 (%)',
                    'min_value': -30.0, 'max_value': 50.0, 'unit': '%'
                }
            }
            
            # 상위 feature들에 대해 변수 정의 생성
            for feature in top_features:
                if feature in feature_meta and feature in actual_values:
                    meta = feature_meta[feature]
                    variable_defs[feature] = {
                        'name': meta['name'],
                        'description': meta['description'],
                        'min_value': meta['min_value'],
                        'max_value': meta['max_value'],
                        'unit': meta['unit'],
                        'current_value': actual_values[feature]  # 실제 2025년 데이터
                    }
            
            print(f"✅ Built {len(variable_defs)} variable definitions from top features: {list(variable_defs.keys())}")
            return variable_defs
            
        except Exception as e:
            print(f"⚠️ Failed to build variable definitions: {e}")
            # 오류 발생 시 빈 딕셔너리 반환 - 하드코딩 금지
            return {}
    
    def _get_top_features(self) -> List[str]:
        """현재 모델에서 SHAP Feature importance 기반 상위 5개 변수 반환"""
        try:
            from app.services.analysis_service import analysis_service
            from app.services.modeling_service import modeling_service
            
            # 모델 확인
            if modeling_service.current_model is None:
                raise ValueError("No model loaded")
            
            # SHAP Feature Importance 가져오기 (차트와 동일한 소스)
            feature_importance_result = analysis_service.get_feature_importance(
                model=modeling_service.current_model,
                method='shap',
                top_n=10
            )
            
            if 'feature_importance' not in feature_importance_result:
                raise ValueError("Failed to get feature importance")
            
            # 상위 10개 feature 추출 (더 많은 변수 조정 가능)
            top_features = [
                item['feature']
                for item in feature_importance_result['feature_importance'][:10]
            ]

            print(f"✅ Top {len(top_features)} features by importance: {top_features}")
            return top_features
            
        except Exception as e:
            print(f"⚠️ Failed to calculate SHAP feature importance: {e}")
            # 오류 발생 시 빈 리스트 반환하여 변수 정의를 생성하지 않음
            return []
    
    def _get_2025_actual_data(self) -> Dict[str, float]:
        """실제 2025년 데이터에서 값들 추출"""
        try:
            from app.services.data_service import data_service
            
            if data_service.current_data is None:
                raise ValueError("No data available")
                
            df = data_service.current_data
            year_2025_data = df[df['eng'] == 2025]
            
            if len(year_2025_data) == 0:
                raise ValueError("No 2025 data found")
            
            row = year_2025_data.iloc[0]
            result = {}
            
            # 모든 feature 컴럼에 대해 값 추출
            for col in df.columns:
                if col not in ['headcount', 'eng']:
                    value = pd.to_numeric(row[col], errors='coerce')
                    if pd.notna(value):
                        result[col] = float(value)
                    else:
                        result[col] = 0.0
            
            print(f"✅ Extracted 2025 actual data for {len(result)} features")
            return result
            
        except Exception as e:
            print(f"⚠️ Failed to get 2025 actual data: {e}")
            return {}
    
    def _prepare_model_input(self, variables: Dict[str, float]) -> pd.DataFrame:
        """실제 데이터 구조에 맞는 모델 입력 준비"""
        try:
            from app.services.data_service import data_service
            
            # 실제 데이터의 컬럼 구조 사용 (headcount 제외)
            if data_service.current_data is None:
                raise ValueError("No data available")
                
            all_columns = list(data_service.current_data.columns)
            feature_columns = [col for col in all_columns if col not in ['headcount', 'eng']]
            
            print(f"✅ Using actual data columns: {feature_columns}")
            
            # 실제 데이터에서 최신값 가져오기
            df = data_service.current_data.copy()
            latest_row = df.iloc[-1]  # 최신 데이터 (2024년)
            
            input_data = {}
            
            for col in feature_columns:
                if col in variables:
                    # 사용자가 조정한 변수 사용
                    input_data[col] = variables[col]
                    print(f"  ✅ Using user input for {col}: {variables[col]}")
                else:
                    # 최신 데이터값 사용
                    value = pd.to_numeric(latest_row[col], errors='coerce')
                    if pd.notna(value):
                        input_data[col] = value
                    else:
                        # 기본값
                        input_data[col] = 0.0
            
            result_df = pd.DataFrame([input_data], columns=feature_columns)
            print(f"✅ Model input shape: {result_df.shape}")
            print(f"   Adjustable variables: {[k for k in variables.keys() if k in feature_columns]}")
            return result_df
            
        except Exception as e:
            print(f"❌ Error in _prepare_model_input: {e}")
            raise e
    
    def _predict_performance_trend(self) -> float:
        """과거 성과 인상률 데이터를 기반으로 다음 년도 성과 인상률 예측

        데이터 구조:
        - 과거 데이터: 각 연도의 경제지표 + 다음 해 임금인상률
        - 최신 년도: 경제지표만 있음 (다음 년도 임금인상률이 예측 대상)

        성과 인상률 트렌드는 과거 임금인상률을 기반으로 다음 년도 예측
        """
        try:
            from app.services.data_service import data_service
            from sklearn.linear_model import LinearRegression
            
            if data_service.current_data is None:
                # 데이터가 없는 경우 에러
                raise ValueError("No data available for performance trend prediction")
            
            # master_data(원본)가 있으면 사용, 없으면 current_data 사용
            if hasattr(data_service, 'master_data') and data_service.master_data is not None:
                df = data_service.master_data.copy()
            else:
                df = data_service.current_data.copy()
            
            # 성과 인상률 관련 컬럼 찾기
            performance_columns = [
                'wage_increase_mi_sbl',  # SBL Merit Increase (성과급)
                'wage_increase_mi_group',  # 그룹 성과급
                'merit_increase',  # 성과급
                'performance_rate',  # 성과 인상률
            ]
            
            # 사용 가능한 컬럼 찾기
            available_col = None
            for col in performance_columns:
                if col in df.columns:
                    available_col = col
                    break
            
            if not available_col:
                # 성과 인상률 컬럼이 없는 경우, 총 인상률에서 추정
                if 'wage_increase_total_sbl' in df.columns and 'wage_increase_baseup_sbl' in df.columns:
                    # 총 인상률 - Base-up = 성과 인상률
                    df['estimated_performance'] = df['wage_increase_total_sbl'] - df['wage_increase_baseup_sbl']
                    available_col = 'estimated_performance'
                elif 'wage_increase_total_sbl' in df.columns and 'wage_increase_bu_sbl' in df.columns:
                    df['estimated_performance'] = df['wage_increase_total_sbl'] - df['wage_increase_bu_sbl']
                    available_col = 'estimated_performance'
                elif 'wage_increase_total_group' in df.columns and 'wage_increase_bu_group' in df.columns:
                    df['estimated_performance'] = df['wage_increase_total_group'] - df['wage_increase_bu_group']
                    available_col = 'estimated_performance'
                else:
                    # 추정할 수 없는 경우 에러
                    raise ValueError("Cannot estimate performance rate from available data")
            
            # 연도와 성과 인상률 데이터 준비
            if 'year' in df.columns:
                year_col = 'year'
            elif 'Year' in df.columns:
                year_col = 'Year'
            elif 'eng' in df.columns:
                # eng 컬럼이 연도 데이터인 경우
                year_col = 'eng'
            else:
                # 연도 컬럼이 없으면 실제 데이터 시작 연도부터 인덱스 사용
                df['year'] = range(2021, 2021 + len(df))
                year_col = 'year'
            
            # 데이터 정리
            trend_data = df[[year_col, available_col]].copy()
            trend_data.columns = ['year', 'performance_rate']
            
            # 수치형으로 변환
            trend_data['year'] = pd.to_numeric(trend_data['year'], errors='coerce')
            trend_data['performance_rate'] = pd.to_numeric(trend_data['performance_rate'], errors='coerce')
            trend_data = trend_data.dropna()
            
            # 데이터가 퍼센트로 저장되어 있는지 확인 (2.0 이상이면 퍼센트로 간주)
            if len(trend_data) > 0 and trend_data['performance_rate'].mean() > 0.5:
                print(f"⚠️ Data appears to be in percentage format (mean: {trend_data['performance_rate'].mean():.2f})")
                # 퍼센트를 비율로 변환 (2.0% -> 0.02)
                trend_data['performance_rate'] = trend_data['performance_rate'] / 100
                print(f"   Converted to ratio format (new mean: {trend_data['performance_rate'].mean():.4f})")
            
            # 2025년 데이터 제외 (타겟이 없는 예측 대상 데이터)
            # 성과 인상률이 실제로 존재하는 데이터만 사용
            trend_data = trend_data[trend_data['performance_rate'].notna()]
            
            # 2025년 이후 데이터 제외 (미래 예측 대상)
            trend_data = trend_data[trend_data['year'] < 2025]
            
            if len(trend_data) < 3:
                # 데이터가 너무 적으면 에러
                raise ValueError("Insufficient data for trend analysis")
            
            # 실제 데이터 기간 사용 (2021-2025)
            trend_data = trend_data.sort_values('year').tail(10)
            
            # 선형회귀 모델 학습
            X = trend_data[['year']].values
            y = trend_data['performance_rate'].values
            
            # 디버깅: 실제 데이터 값 출력
            print(f"📊 Performance rate data for regression:")
            for i, row in trend_data.iterrows():
                print(f"   Year {int(row['year'])}: {row['performance_rate']:.4f} ({row['performance_rate']*100:.2f}%)")
            
            # 평균값 계산 (단순 평균도 참고)
            mean_performance = y.mean()
            print(f"   Average performance rate: {mean_performance:.4f} ({mean_performance*100:.2f}%)")
            
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            
            # 회귀 계수 출력
            print(f"   Regression coefficient (slope): {lr_model.coef_[0]:.6f}")
            print(f"   Regression intercept: {lr_model.intercept_:.6f}")

            # 동적으로 예측 년도 계산 (최신 년도 + 1)
            latest_year = int(trend_data[year_col].max())
            next_year = latest_year + 1
            prediction_year_arr = np.array([[next_year]])
            predicted_performance = lr_model.predict(prediction_year_arr)[0]

            print(f"   Raw prediction for {next_year}: {predicted_performance:.4f} ({predicted_performance*100:.2f}%)")

            print(f"📊 Final Performance rate prediction for {next_year}: {predicted_performance:.3f} ({predicted_performance*100:.1f}%)")
            print(f"   Based on {len(trend_data)} years of data from column '{available_col}'")
            
            return float(predicted_performance)
            
        except Exception as e:
            print(f"⚠️ Error predicting performance trend: {e}")
            # 오류 시 에러 발생
            raise
    
    def _predict_headcount_next_year(self) -> Dict[str, Any]:
        """선형 회귀를 사용하여 다음 년도 headcount 예측"""
        try:
            from app.services.data_service import data_service
            from sklearn.linear_model import LinearRegression
            
            if data_service.current_data is None:
                raise ValueError("No data available for headcount prediction")
            
            # master_data가 있으면 사용, 없으면 current_data 사용
            if hasattr(data_service, 'master_data') and data_service.master_data is not None:
                df = data_service.master_data.copy()
            else:
                df = data_service.current_data.copy()
            
            # headcount와 year 컬럼 확인
            if 'headcount' not in df.columns:
                raise ValueError("No headcount data available")
            
            # 연도 컬럼 찾기
            if 'year' in df.columns:
                year_col = 'year'
            elif 'eng' in df.columns:
                year_col = 'eng'
            else:
                # 연도 컬럼이 없으면 실제 데이터 시작 연도부터 인덱스로 연도 생성
                df['year'] = range(2021, 2021 + len(df))
                year_col = 'year'
            
            # 데이터 정리
            headcount_data = df[[year_col, 'headcount']].copy()
            headcount_data.columns = ['year', 'headcount']
            
            # 수치형으로 변환 및 결측값 제거
            headcount_data['year'] = pd.to_numeric(headcount_data['year'], errors='coerce')
            headcount_data['headcount'] = pd.to_numeric(headcount_data['headcount'], errors='coerce')
            headcount_data = headcount_data.dropna()
            
            if len(headcount_data) < 2:
                raise ValueError("Insufficient headcount data for prediction")
            
            print(f"📊 Headcount data for prediction:")
            for _, row in headcount_data.iterrows():
                print(f"   Year {int(row['year'])}: {int(row['headcount'])} people")
            
            # 안정적인 예측을 위해 최근 트렌드 중심으로 계산
            recent_data = headcount_data.tail(3)  # 최근 3년 데이터만 사용
            
            if len(recent_data) >= 2:
                # 최근 데이터로 선형회귀
                X_recent = recent_data[['year']].values
                y_recent = recent_data['headcount'].values
                
                lr_model = LinearRegression()
                lr_model.fit(X_recent, y_recent)
                
                print(f"   Recent trend coefficient (slope): {lr_model.coef_[0]:.2f}")
                print(f"   Recent trend intercept: {lr_model.intercept_:.2f}")
                
                # 동적으로 다음 년도 예측 계산
                latest_year = int(recent_data['year'].max())
                next_year = latest_year + 1
                prediction_year_arr = np.array([[next_year]])
                base_prediction = lr_model.predict(prediction_year_arr)[0]
                print(f"   Predicting for year: {next_year}")
                
                # 보수적 조정: 급격한 변화 방지
                latest_headcount = recent_data.iloc[-1]['headcount']
                
                # 최대 ±20% 변동 제한
                max_change = latest_headcount * 0.2
                min_prediction = latest_headcount - max_change
                max_prediction = latest_headcount + max_change
                
                predicted_headcount = np.clip(base_prediction, min_prediction, max_prediction)
                predicted_headcount = max(400, round(predicted_headcount))  # 최소 400명 보장
            else:
                # 데이터가 부족한 경우 최근 값 기준으로 보수적 예측
                latest_headcount = headcount_data.iloc[-1]['headcount']
                predicted_headcount = round(latest_headcount * 1.02)  # 2% 성장 가정
                lr_model = None
            
            # 예측 년도 계산
            latest_year = int(headcount_data['year'].max()) if len(headcount_data) > 0 else 2024
            next_year = latest_year + 1

            print(f"📊 Headcount prediction for {next_year}: {predicted_headcount} people")

            # 성장률 계산 (최근년도 대비)
            if len(headcount_data) > 0:
                latest_headcount = headcount_data.iloc[-1]['headcount']
                growth_rate = (predicted_headcount - latest_headcount) / latest_headcount
                print(f"   Growth vs latest year: {growth_rate*100:.1f}%")
                print(f"📊 Final headcount prediction for {next_year}: {predicted_headcount} people")
            else:
                growth_rate = 0
            
            return {
                "predicted_headcount": int(predicted_headcount),
                "growth_rate": float(growth_rate),
                "historical_data": headcount_data.to_dict('records'),
                "model_info": {
                    "slope": float(lr_model.coef_[0]) if lr_model else 0.0,
                    "intercept": float(lr_model.intercept_) if lr_model else 0.0,
                    "data_points": len(headcount_data),
                    "recent_data_points": len(recent_data) if 'recent_data' in locals() else len(headcount_data)
                }
            }
            
        except Exception as e:
            print(f"⚠️ Error predicting headcount: {e}")
            # 기본값 반환
            return {
                "predicted_headcount": 620,  # 현실적인 기본값 (현재 600명 + 약간 성장)
                "growth_rate": 0.03,  # 3% 성장 가정
                "historical_data": [],
                "model_info": {"error": str(e)}
            }

    def predict_wage_increase(self, model, input_data: Dict[str, float], confidence_level: float = 0.95) -> Dict[str, Any]:
        """2025년 임금인상률 예측

        Args:
            model: 학습된 모델
            input_data: 예측에 사용할 2025년 경제지표 데이터
            confidence_level: 신뢰구간 수준
            
        Returns:
            예측 년도의 임금인상률 예측 결과
        """

        try:
            # ModelingService에서 최신 년도 데이터 확인
            from app.services.modeling_service import modeling_service
            from app.services.data_service import data_service

            # 예측 년도 동적 계산: 데이터의 마지막 년도 + 1
            prediction_year = None
            if data_service.current_data is not None and 'year' in data_service.current_data.columns:
                latest_year = int(data_service.current_data['year'].max())
                prediction_year = latest_year + 1
                print(f"📊 Prediction year calculated: {prediction_year} (based on latest data year: {latest_year})")

            # input_data가 없고 modeling_service에 최신 년도 데이터가 있으면 사용
            if not input_data and hasattr(modeling_service, 'prediction_data') and modeling_service.prediction_data is not None:
                # 최신 년도 데이터 사용
                print(f"📊 Using latest year data from modeling service for {prediction_year} prediction")
                model_input = modeling_service.prediction_data.iloc[[0]]  # 첫 번째 행만 사용
                
                # 데이터 누수 방지: 임금 관련 컬럼 모두 제거
                wage_columns_to_remove = [
                    'wage_increase_total_sbl', 'wage_increase_mi_sbl', 'wage_increase_bu_sbl',
                    'wage_increase_baseup_sbl', 'Base-up 인상률', '성과인상률', '임금인상률',
                    'wage_increase_total_group', 'wage_increase_mi_group', 'wage_increase_bu_group'
                ]
                model_input = model_input.drop(columns=wage_columns_to_remove, errors='ignore')
            else:
                # 입력 데이터 준비
                model_input = self._prepare_model_input(input_data)
            
            # PyCaret의 predict_model 사용
            try:
                from pycaret.regression import predict_model
                predictions_df = predict_model(model, data=model_input)
                # 'prediction_label' 컬럼에서 예측값 추출
                if 'prediction_label' in predictions_df.columns:
                    prediction = predictions_df['prediction_label'].iloc[0]
                elif 'Label' in predictions_df.columns:
                    prediction = predictions_df['Label'].iloc[0]
                else:
                    # 마지막 컬럼이 예측값일 가능성이 높음
                    prediction = predictions_df.iloc[0, -1]
            except Exception as e:
                logging.warning(f"PyCaret predict_model failed, using direct prediction: {e}")
                # 폴백: 직접 예측 시도
                prediction = model.predict(model_input)[0]
            
            # PyCaret 모델의 예측값 그대로 사용
            predicted_headcount = round(float(prediction))
            
            # 현재 headcount 대비 성장률 계산
            from app.services.data_service import data_service
            growth_rate = 0.0
            
            if data_service.current_data is not None and 'headcount' in data_service.current_data.columns:
                current_headcount_data = data_service.current_data['headcount'].dropna()
                if len(current_headcount_data) > 0:
                    latest_headcount = current_headcount_data.iloc[-1]
                    growth_rate = (predicted_headcount - latest_headcount) / latest_headcount
                    print(f"📊 PyCaret model prediction: {predicted_headcount} people")
                    print(f"📊 Growth vs latest year ({latest_headcount}): {growth_rate*100:.1f}%")
            
            # headcount 예측 정보 구성
            headcount_prediction = {
                "predicted_headcount": int(predicted_headcount),
                "growth_rate": float(growth_rate),
                "historical_data": [],
                "model_info": {
                    "model_type": "PyCaret ML Model",
                    "features_used": len(model_input.columns)
                }
            }
            
            # PyCaret 모델의 원본 예측값 사용
            # 예측 년도 출력 (동적)
            print(f"🔍 Final ML model prediction: {predicted_headcount} people for {prediction_year if prediction_year else 'next year'}")
            
            # headcount 예측의 신뢰구간 계산
            try:
                from pycaret.regression import get_config
                X_train = get_config('X_train')
                y_train = get_config('y_train')
                
                if X_train is not None and y_train is not None:
                    train_predictions = model.predict(X_train)
                    residuals = y_train - train_predictions
                    residual_std = np.std(residuals)
                    
                    from scipy import stats
                    alpha = 1 - confidence_level
                    z_score = stats.norm.ppf(1 - alpha/2)
                    margin_error = z_score * residual_std
                    
                    confidence_interval = [
                        int(max(0, predicted_headcount - margin_error)),
                        int(predicted_headcount + margin_error)
                    ]
                else:
                    # PyCaret config가 없으면 간단한 신뢰구간 (±10%)
                    confidence_interval = [
                        int(predicted_headcount * 0.9),
                        int(predicted_headcount * 1.1)
                    ]
            except:
                confidence_interval = [
                    int(predicted_headcount * 0.9),
                    int(predicted_headcount * 1.1)
                ]
            
            return {
                "message": "Headcount prediction completed",
                "prediction": predicted_headcount,  # headcount 예측값
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level,
                "input_variables": input_data,
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "model_type": type(model).__name__,
                "headcount_prediction": headcount_prediction  # 상세 정보
            }
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_scenario_templates(self) -> List[Dict[str, Any]]:
        """시나리오 템플릿 목록 반환"""
        
        templates = []
        for key, template in self.scenario_templates.items():
            templates.append({
                "id": key,
                "name": template["name"],
                "description": template["description"],
                "variables": template["variables"]
            })
        
        return templates
    
    def get_available_variables(self) -> Dict[str, Any]:
        """사용 가능한 변수 목록과 정의 반환"""
        
        variables = []
        current_values = {}
        
        for key, definition in self.variable_definitions.items():
            variables.append({
                "name": key,
                "display_name": definition["name"],
                "description": definition["description"],
                "min_value": definition["min_value"],
                "max_value": definition["max_value"],
                "unit": definition["unit"],
                "current_value": definition["current_value"]
            })
            current_values[key] = definition["current_value"]
        
        return {
            "variables": variables,
            "current_values": current_values
        }
    
    def get_economic_indicators(self) -> Dict[str, Any]:
        """주요 경제 지표 반환"""
        
        # 실제 데이터나 외부 API에서 가져올 수 있도록 확장 가능
        return {
            "indicators": {
                "current_inflation": 2.5,
                "current_gdp_growth": 2.8,
                "current_unemployment": 3.2,
                "current_wage_growth": 3.5,
                "last_year_wage_growth": 3.8,
                "industry_average": 3.2,
                "public_sector_average": 2.9
            },
            "last_updated": datetime.now().strftime("%Y-%m-%d")
        }
    
    def perform_scenario_analysis(self, model, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """여러 시나리오에 대한 예측 수행"""
        
        results = []
        
        for scenario in scenarios:
            try:
                prediction_result = self.predict_wage_increase(
                    model,
                    scenario["variables"]
                )
                
                results.append({
                    "scenario_name": scenario.get("scenario_name", "Custom"),
                    "prediction": prediction_result["prediction"],
                    "confidence_interval": prediction_result["confidence_interval"],
                    "variables": scenario["variables"]
                })
                
            except Exception as e:
                logging.error(f"Scenario analysis failed for {scenario.get('scenario_name')}: {str(e)}")
                results.append({
                    "scenario_name": scenario.get("scenario_name", "Custom"),
                    "error": str(e)
                })
        
        return results
    
    def perform_sensitivity_analysis(self, model, base_variables: Dict[str, float], 
                                    target_variable: str, value_range: List[float]) -> Dict[str, Any]:
        """민감도 분석 수행"""
        
        results = []
        
        for value in value_range:
            test_variables = base_variables.copy()
            test_variables[target_variable] = value
            
            try:
                prediction_result = self.predict_wage_increase(model, test_variables)
                results.append({
                    "variable_value": value,
                    "prediction": prediction_result["prediction"]
                })
            except Exception as e:
                logging.error(f"Sensitivity analysis failed for {target_variable}={value}: {str(e)}")
                results.append({
                    "variable_value": value,
                    "error": str(e)
                })
        
        return {
            "target_variable": target_variable,
            "base_value": base_variables.get(target_variable),
            "results": results
        }
    
    def get_trend_data(self) -> Dict[str, Any]:
        """트렌드 데이터 반환"""
        
        try:
            # 원본 master_data 파일 로드
            import pickle
            import os
            
            master_data_path = os.path.join(os.path.dirname(__file__), '../../data/master_data.pkl')
            
            if os.path.exists(master_data_path):
                with open(master_data_path, 'rb') as f:
                    data = pickle.load(f)
                    # data가 dict인 경우 DataFrame으로 변환
                    if isinstance(data, dict):
                        if 'data' in data:
                            df = data['data']
                        else:
                            df = pd.DataFrame(data)
                    else:
                        df = data
                print(f"✅ Loaded original master_data from {master_data_path}")
            elif data_service.current_data is not None:
                df = data_service.current_data.copy()
                print("⚠️ Using current_data (may contain augmented data)")
            else:
                df = None
            
            if df is not None:
                # 타겟 컬럼 찾기 (headcount 데이터)
                target_col = 'headcount'
                if target_col not in df.columns:
                    # 다른 가능한 headcount 컬럼들 시도
                    for col in ['정원', 'employee_count', 'total_headcount']:
                        if col in df.columns:
                            target_col = col
                            break
                
                # year 또는 eng 컬럼 찾기
                year_col = 'year' if 'year' in df.columns else 'eng' if 'eng' in df.columns else None
                
                if target_col in df.columns and year_col:
                    # 원본 데이터만 사용 (master_data는 이미 원본)
                    yearly_data = df.groupby(year_col)[target_col].first().dropna()
                    
                    # 과거 데이터 포맷팅
                    # Headcount 연도별 데이터
                    historical_data = []
                    
                    for year, value in yearly_data.items():
                        if pd.notna(value):
                            # headcount는 절대값이므로 그대로 사용
                            display_value = int(float(value))
                            input_year = int(year)  # 입력 데이터 연도
                            prediction_year = input_year + 1  # 예측 대상 연도
                            
                            # 2021년 데이터 → 2022년 예측, 2022년 데이터 → 2023년 예측 ...
                            data_point = {
                                "year": prediction_year,  # 예측 대상 연도로 표시
                                "value": display_value,
                                "type": "historical",
                                "input_year": input_year  # 참조용
                            }
                            
                            historical_data.append(data_point)
                    
                    # 다음 년도 예측 데이터 추가 (모델이 있는 경우)
                    # 동적으로 예측 년도 계산
                    latest_year = int(df[year_col].max())
                    next_year = latest_year + 1

                    # 이미 다음 년도 데이터가 있는지 확인
                    has_next_year = any(d.get('year') == next_year for d in historical_data)

                    from app.services.modeling_service import modeling_service
                    if modeling_service.current_model and not has_next_year:
                        try:
                            # 최신 년도 데이터로 PyCaret 모델 예측 수행
                            # 최신 년도 행(마지막 행)의 실제 데이터 사용
                            year_latest_data = df[df[year_col] == latest_year]
                            
                            if len(year_latest_data) > 0:
                                # 최신 년도 데이터에서 feature 값들 추출
                                row_latest = year_latest_data.iloc[0]
                                feature_columns = [col for col in df.columns if col not in ['headcount', year_col]]

                                model_input = {}
                                for col in feature_columns:
                                    value = pd.to_numeric(row_latest[col], errors='coerce')
                                    if pd.notna(value):
                                        model_input[col] = value
                                    else:
                                        model_input[col] = 0.0

                                print(f"✅ Using {latest_year} data for prediction: {list(model_input.keys())[:5]}...")
                            else:
                                # 최신 년도 데이터가 없으면 기본값 사용
                                model_input = {
                                    'operating_income': 5.2,
                                    'ev_growth_gl': 8.5,
                                    'exchange_rate_change_krw': 2.3,
                                    'labor_costs': 4.8,
                                    'v_growth_gl': 7.2
                                }
                                print(f"⚠️ No {latest_year} data found, using default values")

                            # PyCaret 모델로 다음 년도 headcount 예측
                            prediction_result = self.predict_wage_increase(
                                modeling_service.current_model,
                                model_input,
                                confidence_level=0.95
                            )
                            
                            # 예측값 검증
                            pred_value = prediction_result["prediction"]
                            base_up = prediction_result.get("base_up_rate", 0)
                            perf = prediction_result.get("performance_rate", 0)
                            
                            # headcount 예측값은 절대값이므로 정상 범위 체크 수정
                            # 예측값이 0보다 작거나 너무 큰 경우만 비정상으로 처리
                            if pred_value < 0 or pred_value > 10000:
                                print(f"⚠️ Abnormal headcount prediction value: {pred_value}")
                                raise ValueError(f"Abnormal headcount prediction: {pred_value}")
                            
                            # headcount 예측 결과 추가 (절대값으로 사용)
                            prediction_data = {
                                "year": next_year,
                                "value": int(round(pred_value)),  # headcount는 절대값
                                "type": "prediction",
                                "input_year": latest_year  # 최신 년도 데이터로 예측
                            }
                            historical_data.append(prediction_data)

                            print(f"✅ Added {next_year} headcount prediction: {prediction_data['value']}명 (from {latest_year} data)")
                        except Exception as e:
                            print(f"⚠️ Could not generate prediction: {e}")
                            # 오류 시 ML 예측값을 추가하지 않음
                            pass
                    
                    # 예측 년도 계산 (마지막 데이터 년도 + 1)
                    chart_prediction_year = latest_year + 1 if 'latest_year' in locals() else 2025

                    return {
                        "message": "Trend data retrieved successfully",
                        "trend_data": historical_data,
                        "baseup_data": baseup_data if 'baseup_data' in locals() else [],
                        "chart_config": {
                            "title": f"인원 수 추이 및 {chart_prediction_year}년 예측",
                            "y_axis_label": "인원 수 (명)",
                            "x_axis_label": "연도"
                        }
                    }
            
            # 데이터가 없으면 빈 배열 반환
            return {
                "message": "No trend data available",
                "trend_data": [],
                "chart_config": {
                    "title": "인원 수 추이",
                    "y_axis_label": "인원 수 (명)",
                    "x_axis_label": "연도"
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to get trend data: {str(e)}")
            return {
                "message": f"Error: {str(e)}",
                "trend_data": [],
                "chart_config": {}
            }

# 싱글톤 인스턴스
dashboard_service = DashboardService()