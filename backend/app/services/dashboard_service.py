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
                    "inflation_rate": 2.5,
                    "gdp_growth": 2.8,
                    "unemployment_rate": 3.2,
                    "productivity_growth": 2.0,
                    "exchange_rate_volatility": 1.0
                }
            },
            "optimistic": {
                "name": "낙관적 시나리오",
                "description": "경제 호황 상황",
                "variables": {
                    "inflation_rate": 2.0,
                    "gdp_growth": 4.5,
                    "unemployment_rate": 2.5,
                    "productivity_growth": 3.5,
                    "exchange_rate_volatility": 0.8
                }
            },
            "moderate": {
                "name": "중립적 시나리오",
                "description": "안정적 성장",
                "variables": {
                    "inflation_rate": 2.5,
                    "gdp_growth": 3.0,
                    "unemployment_rate": 3.0,
                    "productivity_growth": 2.5,
                    "exchange_rate_volatility": 1.0
                }
            },
            "pessimistic": {
                "name": "비관적 시나리오",
                "description": "경제 침체 상황",
                "variables": {
                    "inflation_rate": 3.5,
                    "gdp_growth": -1.5,
                    "unemployment_rate": 6.5,
                    "productivity_growth": -0.5,
                    "exchange_rate_volatility": 1.8
                }
            }
        }
        
        self.variable_definitions = {
            "inflation_rate": {
                "name": "인플레이션율",
                "description": "소비자물가지수 상승률 (%)",
                "min_value": -2.0,
                "max_value": 8.0,
                "unit": "%",
                "current_value": 2.5
            },
            "gdp_growth": {
                "name": "GDP 성장률",
                "description": "실질 GDP 전년 대비 성장률 (%)",
                "min_value": -5.0,
                "max_value": 8.0,
                "unit": "%",
                "current_value": 2.8
            },
            "unemployment_rate": {
                "name": "실업률",
                "description": "경제활동인구 대비 실업자 비율 (%)",
                "min_value": 1.0,
                "max_value": 10.0,
                "unit": "%",
                "current_value": 3.2
            },
            "productivity_growth": {
                "name": "생산성 증가율",
                "description": "노동생산성 전년 대비 증가율 (%)",
                "min_value": -3.0,
                "max_value": 6.0,
                "unit": "%",
                "current_value": 2.0
            },
            "exchange_rate_volatility": {
                "name": "환율 변동성",
                "description": "환율 변동성 지수 (기준=1.0)",
                "min_value": 0.5,
                "max_value": 2.5,
                "unit": "지수",
                "current_value": 1.0
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
            variable_mapping = {
                'gdp_growth': ('gdp_growth_kr', 0.01),      # 2.8% → 0.028
                'inflation_rate': ('cpi_kr', 0.01),        # 2.5% → 0.025  
                'unemployment_rate': ('unemployment_rate_kr', 0.01),  # 3.2% → 0.032
                'productivity_growth': ('minimum_wage_increase_kr', 0.01),  # 2.0% → 0.02
                'exchange_rate_volatility': ('exchange_rate_change_krw', 0.01)  # 1.0 → 0.01
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
                if col == 'gdp_growth_kr':
                    default_data[col] = variables.get('gdp_growth', 2.8) * 0.01
                elif col == 'cpi_kr':
                    default_data[col] = variables.get('inflation_rate', 2.5) * 0.01
                elif col == 'unemployment_rate_kr':
                    default_data[col] = variables.get('unemployment_rate', 3.2) * 0.01
                elif col == 'minimum_wage_increase_kr':
                    default_data[col] = variables.get('productivity_growth', 2.0) * 0.01
                elif col == 'exchange_rate_change_krw':
                    default_data[col] = variables.get('exchange_rate_volatility', 1.0) * 0.01
                else:
                    default_data[col] = 0.02  # 기본값
            
            return pd.DataFrame([default_data])
    
    def predict_wage_increase(self, model, input_data: Dict[str, float], confidence_level: float = 0.95) -> Dict[str, Any]:
        """임금인상률 예측"""
        
        try:
            # 입력 데이터 준비
            model_input = self._prepare_model_input(input_data)
            
            # 예측 수행
            prediction = model.predict(model_input)[0]
            
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
                    # 기본값
                    confidence_interval = [
                        float(prediction * 0.85),
                        float(prediction * 1.15)
                    ]
            except:
                confidence_interval = [
                    float(prediction * 0.85),
                    float(prediction * 1.15)
                ]
            
            return {
                "message": "Wage increase prediction completed",
                "prediction": float(prediction),
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level,
                "input_variables": input_data,
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "model_type": type(model).__name__
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
            if data_service.current_data is not None and 'year' in data_service.current_data.columns:
                df = data_service.current_data.copy()
                
                # 연도별 임금인상률 데이터
                if 'wage_increase_total_sbl' in df.columns:
                    trend_data = df[['year', 'wage_increase_total_sbl']].dropna()
                    
                    # 수치형으로 변환
                    trend_data['year'] = pd.to_numeric(trend_data['year'], errors='coerce')
                    trend_data['wage_increase_total_sbl'] = pd.to_numeric(trend_data['wage_increase_total_sbl'], errors='coerce')
                    trend_data = trend_data.dropna()
                    
                    return {
                        "years": trend_data['year'].tolist(),
                        "values": trend_data['wage_increase_total_sbl'].tolist(),
                        "label": "임금인상률 (%)",
                        "available": True
                    }
            
            # 기본 데이터 (예시)
            return {
                "years": list(range(2015, 2024)),
                "values": [2.8, 3.2, 3.5, 3.8, 4.2, 3.9, 3.6, 3.3, 3.5],
                "label": "임금인상률 (%)",
                "available": False
            }
            
        except Exception as e:
            logging.error(f"Failed to get trend data: {str(e)}")
            return {
                "years": [],
                "values": [],
                "label": "임금인상률 (%)",
                "available": False,
                "error": str(e)
            }

# 싱글톤 인스턴스
dashboard_service = DashboardService()