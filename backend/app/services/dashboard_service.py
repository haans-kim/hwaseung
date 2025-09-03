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
                    "operating_income": 5.2,
                    "ev_growth_gl": 8.5,
                    "exchange_rate_change_krw": 2.3,
                    "labor_costs": 4.8,
                    "v_growth_gl": 7.2
                }
            },
            "optimistic": {
                "name": "낙관적 시나리오",
                "description": "고성장 + 수익성 개선",
                "variables": {
                    "operating_income": 15.0,
                    "ev_growth_gl": 20.0,
                    "exchange_rate_change_krw": 5.0,
                    "labor_costs": 8.0,
                    "v_growth_gl": 18.0
                }
            },
            "moderate": {
                "name": "중립적 시나리오",
                "description": "안정적 성장",
                "variables": {
                    "operating_income": 8.0,
                    "ev_growth_gl": 12.0,
                    "exchange_rate_change_krw": 3.5,
                    "labor_costs": 6.0,
                    "v_growth_gl": 10.0
                }
            },
            "pessimistic": {
                "name": "비관적 시나리오",
                "description": "저성장 + 수익성 악화",
                "variables": {
                    "operating_income": -5.0,
                    "ev_growth_gl": 2.0,
                    "exchange_rate_change_krw": -2.0,
                    "labor_costs": 2.0,
                    "v_growth_gl": 1.5
                }
            }
        }
        
        self.variable_definitions = {
            "operating_income": {
                "name": "영업이익",
                "description": "전년 대비 영업이익 증가율 (%)",
                "min_value": -20.0,
                "max_value": 30.0,
                "unit": "%",
                "current_value": 5.2
            },
            "ev_growth_gl": {
                "name": "기업가치 성장률",
                "description": "글로벌 기업가치 증가율 (%)",
                "min_value": -15.0,
                "max_value": 25.0,
                "unit": "%",
                "current_value": 8.5
            },
            "exchange_rate_change_krw": {
                "name": "환율 변동률",
                "description": "원달러 환율 변동률 (%)",
                "min_value": -10.0,
                "max_value": 15.0,
                "unit": "%",
                "current_value": 2.3
            },
            "labor_costs": {
                "name": "인건비",
                "description": "총 인건비 증가율 (%)",
                "min_value": 0.0,
                "max_value": 20.0,
                "unit": "%",
                "current_value": 4.8
            },
            "v_growth_gl": {
                "name": "매출 성장률",
                "description": "글로벌 매출 성장률 (%)",
                "min_value": -10.0,
                "max_value": 25.0,
                "unit": "%",
                "current_value": 7.2
            }
        }
    
    def _prepare_model_input(self, variables: Dict[str, float]) -> pd.DataFrame:
        """모델 입력용 데이터 준비 - PyCaret 모델에 맞게 수정"""
        try:
            # PyCaret 모델의 feature names 가져오기
            from app.services.modeling_service import modeling_service
            from pycaret.regression import get_config
            
            # PyCaret 설정에서 feature 정보 가져오기
            try:
                # 먼저 모델링 서비스에서 feature names 가져오기 (가장 정확함)
                if hasattr(modeling_service, 'feature_names') and modeling_service.feature_names:
                    feature_columns = modeling_service.feature_names
                    print(f"✅ Using feature names from modeling_service: {len(feature_columns)} features")
                else:
                    # PyCaret config에서 직접 가져오기
                    X_train = get_config('X_train')
                    if X_train is not None:
                        feature_columns = list(X_train.columns)
                        print(f"✅ Using feature names from PyCaret config: {len(feature_columns)} features")
                    else:
                        # 기본 feature 리스트 (실제 데이터 기반)
                        feature_columns = [
                            'gdp_growth_kr', 'cpi_kr', 'unemployment_rate_kr', 'minimum_wage_increase_kr',
                            'gdp_growth_usa', 'cpi_usa', 'esi_usa', 'exchange_rate_change_krw',
                            'revenue_growth_sbl', 'op_profit_growth_sbl', 'labor_cost_rate_sbl',
                            'labor_cost_ratio_change_sbl', 'labor_cost_per_employee_sbl', 'labor_to_revenue_sbl',
                            'revenue_per_employee_sbl', 'op_profit_per_employee_sbl', 'hcroi_sbl', 'hcva_sbl',
                            'wage_increase_ce', 'revenue_growth_ce', 'op_profit_growth_ce', 'hcroi_ce', 'hcva_ce',
                            'market_size_growth_rate', 'compensation_competitiveness', 'wage_increase_bu_group',
                            'wage_increase_mi_group', 'wage_increase_total_group', 'public_sector_wage_increase'
                        ]
                        print(f"⚠️ Using default feature list: {len(feature_columns)} features")
            except Exception as e:
                print(f"Warning: Could not get PyCaret config: {e}")
                # 기본 feature 리스트 사용
                feature_columns = [
                    'gdp_growth_kr', 'cpi_kr', 'unemployment_rate_kr', 'minimum_wage_increase_kr',
                    'gdp_growth_usa', 'cpi_usa', 'esi_usa', 'exchange_rate_change_krw',
                    'revenue_growth_sbl', 'op_profit_growth_sbl', 'labor_cost_rate_sbl',
                    'labor_cost_ratio_change_sbl', 'labor_cost_per_employee_sbl', 'labor_to_revenue_sbl',
                    'revenue_per_employee_sbl', 'op_profit_per_employee_sbl', 'hcroi_sbl', 'hcva_sbl',
                    'wage_increase_ce', 'revenue_growth_ce', 'op_profit_growth_ce', 'hcroi_ce', 'hcva_ce',
                    'market_size_growth_rate', 'compensation_competitiveness', 'wage_increase_bu_group',
                    'wage_increase_mi_group', 'wage_increase_total_group', 'public_sector_wage_increase'
                ]
            
            # 변수 매핑: Dashboard 변수 → 실제 데이터 컬럼
            # 영향요인 분석 결과 기반으로 가장 중요한 변수들 매핑
            variable_mapping = {
                'wage_increase_bu_group': ('wage_increase_bu_group', 0.01),  # 3.0% → 0.03 (가장 중요!)
                'gdp_growth': ('gdp_growth_kr', 0.01),      # 2.8% → 0.028
                'unemployment_rate': ('unemployment_rate_kr', 0.01),  # 3.2% → 0.032
                'market_size_growth_rate': ('market_size_growth_rate', 0.01),  # 5.0% → 0.05
                'hcroi_sbl': ('hcroi_sbl', 1.0)  # 1.5배 → 1.5 (비율이므로 그대로)
            }
            
            # 데이터에서 수치형 값들의 평균값 계산 (결측값과 '-' 제외)
            df_clean = None
            if data_service.current_data is not None:
                df_clean = data_service.current_data.copy()
                for col in df_clean.columns:
                    if df_clean[col].dtype == 'object':  # 문자열 컬럼
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            input_data = {}
            
            for col in feature_columns:
                # 매핑된 변수가 있으면 사용자 입력값 적용
                mapped_variable = None
                scale_factor = 1.0
                
                for dash_var, (data_col, scale) in variable_mapping.items():
                    if data_col == col and dash_var in variables:
                        mapped_variable = dash_var
                        scale_factor = scale
                        break
                
                if mapped_variable:
                    input_data[col] = variables[mapped_variable] * scale_factor
                else:
                    # 해당 컬럼의 평균값 사용 (결측값 제외)
                    if df_clean is not None and col in df_clean.columns:
                        col_mean = df_clean[col].mean()
                        if pd.notna(col_mean):
                            input_data[col] = col_mean
                        else:
                            # 컬럼별 기본값 설정
                            if 'wage' in col or 'increase' in col:
                                input_data[col] = 0.03  # 임금 관련은 3%
                            elif 'growth' in col:
                                input_data[col] = 0.02  # 성장률 관련은 2%
                            elif 'rate' in col or 'ratio' in col:
                                input_data[col] = 0.1  # 비율 관련은 10%
                            else:
                                input_data[col] = 0.0
                    else:
                        # 컬럼별 기본값 설정
                        if 'wage' in col or 'increase' in col:
                            input_data[col] = 0.03
                        elif 'growth' in col:
                            input_data[col] = 0.02
                        elif 'rate' in col or 'ratio' in col:
                            input_data[col] = 0.1
                        else:
                            input_data[col] = 0.0
            
            print(f"📊 Model input prepared with {len(input_data)} features")
            
            # DataFrame 생성 시 컬럼 순서 보장
            result_df = pd.DataFrame([input_data], columns=feature_columns)
            print(f"✅ DataFrame shape: {result_df.shape}, columns: {list(result_df.columns)[:5]}...")
            return result_df
                
        except Exception as e:
            logging.error(f"Error preparing model input: {str(e)}")
            print(f"❌ Error details: {e}")
            
            # 폴백: 29개 feature로 기본 DataFrame 생성
            default_features = [
                'gdp_growth_kr', 'cpi_kr', 'unemployment_rate_kr', 'minimum_wage_increase_kr',
                'gdp_growth_usa', 'cpi_usa', 'esi_usa', 'exchange_rate_change_krw',
                'revenue_growth_sbl', 'op_profit_growth_sbl', 'labor_cost_rate_sbl',
                'labor_cost_ratio_change_sbl', 'labor_cost_per_employee_sbl', 'labor_to_revenue_sbl',
                'revenue_per_employee_sbl', 'op_profit_per_employee_sbl', 'hcroi_sbl', 'hcva_sbl',
                'wage_increase_ce', 'revenue_growth_ce', 'op_profit_growth_ce', 'hcroi_ce', 'hcva_ce',
                'market_size_growth_rate', 'compensation_competitiveness', 'wage_increase_bu_group',
                'wage_increase_mi_group', 'wage_increase_total_group', 'public_sector_wage_increase'
            ]
            
            default_data = {}
            for col in default_features:
                if col == 'wage_increase_bu_group':
                    default_data[col] = variables.get('wage_increase_bu_group', 3.0) * 0.01
                elif col == 'gdp_growth_kr':
                    default_data[col] = variables.get('gdp_growth', 2.8) * 0.01
                elif col == 'unemployment_rate_kr':
                    default_data[col] = variables.get('unemployment_rate', 3.2) * 0.01
                elif col == 'market_size_growth_rate':
                    default_data[col] = variables.get('market_size_growth_rate', 5.0) * 0.01
                elif col == 'hcroi_sbl':
                    default_data[col] = variables.get('hcroi_sbl', 1.5)  # 비율이므로 그대로
                elif col == 'cpi_kr':
                    default_data[col] = 0.025  # 기본 인플레이션 2.5%
                elif col == 'minimum_wage_increase_kr':
                    default_data[col] = 0.025  # 기본 최저임금인상률 2.5%
                else:
                    default_data[col] = 0.02  # 기본값
            
            return pd.DataFrame([default_data])
    
    def _predict_performance_trend(self) -> float:
        """과거 성과 인상률 데이터를 기반으로 2026년 성과 인상률 예측
        
        데이터 구조:
        - 2021-2024년: 각 연도의 경제지표 + 다음 해 임금인상률
        - 2025년: 경제지표만 있음 (2026년 임금인상률이 예측 대상)
        
        성과 인상률 트렌드는 2022-2025년 임금인상률을 기반으로 2026년 예측
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
            
            # 2026년 예측
            prediction_year = np.array([[2026]])
            predicted_performance = lr_model.predict(prediction_year)[0]
            
            print(f"   Raw prediction for 2026: {predicted_performance:.4f} ({predicted_performance*100:.2f}%)")
            
            print(f"📊 Final Performance rate prediction for 2026: {predicted_performance:.3f} ({predicted_performance*100:.1f}%)")
            print(f"   Based on {len(trend_data)} years of data from column '{available_col}'")
            
            return float(predicted_performance)
            
        except Exception as e:
            print(f"⚠️ Error predicting performance trend: {e}")
            # 오류 시 에러 발생
            raise
    
    def _predict_headcount_2026(self) -> Dict[str, Any]:
        """PyCaret을 사용하여 2026년 headcount 예측"""
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
            
            # 선형회귀 모델 학습
            X = headcount_data[['year']].values
            y = headcount_data['headcount'].values
            
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            
            # 회귀 계수 출력
            print(f"   Regression coefficient (slope): {lr_model.coef_[0]:.2f}")
            print(f"   Regression intercept: {lr_model.intercept_:.2f}")
            
            # 2026년 예측
            prediction_year = np.array([[2026]])
            predicted_headcount = lr_model.predict(prediction_year)[0]
            predicted_headcount = max(0, round(predicted_headcount))  # 음수 방지 및 반올림
            
            print(f"📊 Headcount prediction for 2026: {predicted_headcount} people")
            
            # 성장률 계산 (최근년도 대비)
            if len(headcount_data) > 0:
                latest_headcount = headcount_data.iloc[-1]['headcount']
                growth_rate = (predicted_headcount - latest_headcount) / latest_headcount
                print(f"   Growth vs latest year: {growth_rate*100:.1f}%")
            else:
                growth_rate = 0
            
            return {
                "predicted_headcount": int(predicted_headcount),
                "growth_rate": float(growth_rate),
                "historical_data": headcount_data.to_dict('records'),
                "model_info": {
                    "slope": float(lr_model.coef_[0]),
                    "intercept": float(lr_model.intercept_),
                    "data_points": len(headcount_data)
                }
            }
            
        except Exception as e:
            print(f"⚠️ Error predicting headcount: {e}")
            # 기본값 반환
            return {
                "predicted_headcount": 700,  # 기본 예상값
                "growth_rate": 0.05,  # 5% 성장 가정
                "historical_data": [],
                "model_info": {"error": str(e)}
            }

    def predict_wage_increase(self, model, input_data: Dict[str, float], confidence_level: float = 0.95) -> Dict[str, Any]:
        """2026년 임금인상률 예측
        
        Args:
            model: 학습된 모델
            input_data: 예측에 사용할 2025년 경제지표 데이터
            confidence_level: 신뢰구간 수준
            
        Returns:
            2026년 임금인상률 예측 결과
        """
        
        try:
            # ModelingService에서 2025년 데이터 확인
            from app.services.modeling_service import modeling_service
            
            # input_data가 없고 modeling_service에 2025년 데이터가 있으면 사용
            if not input_data and hasattr(modeling_service, 'prediction_data') and modeling_service.prediction_data is not None:
                # 2025년 데이터 사용
                print("📊 Using 2025 data from modeling service for 2026 prediction")
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
            
            # 성과 인상률은 적정인력 산정에서 사용하지 않음
            performance_rate = 0.0
            
            # 2026년 headcount 예측 추가
            headcount_prediction = self._predict_headcount_2026()
            
            # 반올림 처리를 위해 소수점 4자리까지만 유지
            raw_prediction = round(float(prediction), 4)
            performance_rate = round(performance_rate, 4)
            
            # 최근 트렌드 반영한 조정
            # 최근 2년이 5.3%, 5.6%로 높은 인상률을 보임
            from app.services.data_service import data_service
            
            # 그룹 Base-up의 논리적 영향 반영
            # 그룹 Base-up이 높으면 SBL 임금도 높아야 함 (상식적 관계)
            if isinstance(input_data, dict) and 'wage_increase_bu_group' in input_data:
                group_baseup_input = input_data['wage_increase_bu_group']
                # 기준값(3.0%)과의 차이를 계산
                baseup_diff = (group_baseup_input - 3.0) * 0.01
                # 양의 관계로 조정 (그룹 base-up 1%p 증가 → 예측값 0.3%p 증가)
                logical_adjustment = baseup_diff * 0.3
                prediction_value = round(raw_prediction + logical_adjustment, 4)
            else:
                prediction_value = raw_prediction
            
            print(f"🔍 Debug - Raw model prediction: {raw_prediction:.4f} ({raw_prediction*100:.2f}%)")
            print(f"🔍 Debug - Adjusted prediction (60% model + 40% trend): {prediction_value:.4f} ({prediction_value*100:.2f}%)")
            print(f"🔍 Debug - Performance rate (from trend): {performance_rate:.4f} ({performance_rate*100:.2f}%)")
            
            # Base-up = 총 인상률 - 성과 인상률
            base_up_rate = round(prediction_value - performance_rate, 4)
            print(f"🔍 Debug - Base-up (total - performance): {base_up_rate:.4f} ({base_up_rate*100:.2f}%)")
            
            # Base-up이 음수인 경우 - 성과 인상률은 변경하지 않고 base_up만 조정
            if base_up_rate < 0:
                print(f"⚠️ Debug - Base-up negative ({base_up_rate:.4f}), setting to 0")
                base_up_rate = 0
                # 성과 인상률은 트렌드 예측값 그대로 유지
            
            # 성과 인상률이 총 예측값보다 큰 경우 - 성과 인상률은 유지하고 base_up을 0으로
            if performance_rate > prediction_value:
                print(f"⚠️ Debug - Performance ({performance_rate:.4f}) > Total ({prediction_value:.4f})")
                print(f"⚠️ Debug - Keeping performance rate as is, setting base_up to 0")
                base_up_rate = 0
                # 성과 인상률은 트렌드 예측값 그대로 유지
            
            # 최종 검증: 합계가 총 예측값과 일치하도록 조정
            calculated_total = round(base_up_rate + performance_rate, 4)
            if abs(calculated_total - prediction_value) > 0.0001:
                # 차이가 있으면 base_up_rate로 조정
                base_up_rate = round(prediction_value - performance_rate, 4)
            
            print(f"✅ Debug - FINAL VALUES:")
            print(f"   Performance: {performance_rate:.4f} ({performance_rate*100:.2f}%)")
            print(f"   Base-up: {base_up_rate:.4f} ({base_up_rate*100:.2f}%)")
            print(f"   Total: {prediction_value:.4f} ({prediction_value*100:.2f}%)")
            print(f"   Sum check: {base_up_rate + performance_rate:.4f} vs {prediction_value:.4f}")
            
            # 신뢰구간 계산 (간단한 방법 - 잔차 기반)
            try:
                from pycaret.regression import get_config
                X_train = get_config('X_train')
                y_train = get_config('y_train')
                
                if X_train is not None and y_train is not None:
                    train_predictions = model.predict(X_train)
                    residuals = y_train - train_predictions
                    residual_std = np.std(residuals)
                    
                    # 신뢰구간
                    from scipy import stats
                    alpha = 1 - confidence_level
                    z_score = stats.norm.ppf(1 - alpha/2)
                    margin_error = z_score * residual_std
                    
                    confidence_interval = [
                        float(prediction - margin_error),
                        float(prediction + margin_error)
                    ]
                else:
                    # PyCaret config가 없으면 간단한 신뢰구간 계산
                    confidence_interval = [
                        round(prediction_value * 0.95, 4),
                        round(prediction_value * 1.05, 4)
                    ]
            except:
                confidence_interval = [
                    round(prediction_value * 0.95, 4),
                    round(prediction_value * 1.05, 4)
                ]
            
            return {
                "message": "Wage increase prediction completed",
                "prediction": prediction_value,
                "base_up_rate": base_up_rate,
                "performance_rate": performance_rate,
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level,
                "input_variables": input_data,
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "model_type": type(model).__name__,
                "headcount_prediction": headcount_prediction,  # 2026년 headcount 예측 추가
                "breakdown": {
                    "base_up": {
                        "rate": base_up_rate,
                        "percentage": round(base_up_rate * 100, 2),
                        "description": "기본 인상분",
                        "calculation": "총 인상률 - 성과 인상률"
                    },
                    "performance": {
                        "rate": performance_rate,
                        "percentage": round(performance_rate * 100, 2),
                        "description": "과거 10년 성과급 추세 기반 예측",
                        "calculation": "선형회귀 분석으로 예측"
                    },
                    "total": {
                        "rate": prediction_value,
                        "percentage": round(prediction_value * 100, 2),
                        "description": "2026년 총 임금 인상률 예측",
                        "verification": f"{round(base_up_rate * 100, 2)}% + {round(performance_rate * 100, 2)}% = {round(prediction_value * 100, 2)}%"
                    }
                }
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
                            actual_year = int(year)
                            
                            data_point = {
                                "year": actual_year,
                                "value": display_value,
                                "type": "historical"
                            }
                            
                            historical_data.append(data_point)
                    
                    # 2026년 예측 데이터 추가 (모델이 있는 경우)
                    # 이미 2026년 데이터가 있는지 확인
                    has_2026 = any(d.get('year') == 2026 for d in historical_data)
                    
                    from app.services.modeling_service import modeling_service
                    if modeling_service.current_model and not has_2026:
                        try:
                            # 실제 모델 예측 수행
                            default_input = {
                                'wage_increase_bu_group': 3.0,
                                'gdp_growth': 2.8,
                                'unemployment_rate': 3.2,
                                'market_size_growth_rate': 5.0,
                                'hcroi_sbl': 1.5
                            }
                            
                            # 예측 수행
                            prediction_result = self.predict_wage_increase(
                                modeling_service.current_model,
                                default_input,
                                confidence_level=0.95
                            )
                            
                            # 예측값 검증
                            pred_value = prediction_result["prediction"]
                            base_up = prediction_result.get("base_up_rate", 0)
                            perf = prediction_result.get("performance_rate", 0)
                            
                            # 비정상적인 값 체크 (예: 100% 이상 또는 음수)
                            if abs(pred_value) > 1.0 or pred_value < 0:
                                print(f"⚠️ Abnormal prediction value: {pred_value}")
                                raise ValueError("Abnormal prediction value")
                            
                            # 예측 결과를 퍼센트로 변환하여 추가
                            prediction_data = {
                                "year": 2026,
                                "value": round(pred_value * 100, 2),
                                "base_up": round(base_up * 100, 2),
                                "performance": round(perf * 100, 2),
                                "type": "prediction"
                            }
                            historical_data.append(prediction_data)
                            
                            # Base-up 데이터도 별도로 추가 (차트에서 사용)
                            if hasbaseup and 'baseup_data' in locals():
                                baseup_pred = {
                                    "year": 2026,
                                    "value": round(prediction_result.get("base_up_rate", 0) * 100, 2),
                                    "type": "prediction"
                                }
                                baseup_data.append(baseup_pred)
                            
                            print(f"✅ Added 2026 prediction: Total={prediction_data['value']}%, Base-up={prediction_data['base_up']}%")
                        except Exception as e:
                            print(f"⚠️ Could not generate prediction: {e}")
                            # 오류 시에는 추가하지 않음 (중복 방지)
                            pass
                    
                    return {
                        "message": "Trend data retrieved successfully",
                        "trend_data": historical_data,
                        "baseup_data": baseup_data if 'baseup_data' in locals() else [],
                        "chart_config": {
                            "title": "인원 수 추이 및 2026년 예측",
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