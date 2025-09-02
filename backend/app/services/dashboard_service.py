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
    
    def _predict_performance_trend(self) -> float:
        """과거 성과 인상률 데이터를 기반으로 2026년 성과 인상률 예측
        
        데이터 구조:
        - 2015-2024년: 각 연도의 경제지표 + 다음 해 임금인상률
        - 2025년: 경제지표만 있음 (2026년 임금인상률이 예측 대상)
        
        성과 인상률 트렌드는 2016-2025년 임금인상률을 기반으로 2026년 예측
        """
        try:
            from app.services.data_service import data_service
            from sklearn.linear_model import LinearRegression
            
            if data_service.current_data is None:
                # 데이터가 없는 경우 기본값 반환
                return 0.02  # 2% 기본값
            
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
                    # 추정할 수 없는 경우 기본값
                    return 0.02
            
            # 연도와 성과 인상률 데이터 준비
            if 'year' in df.columns:
                year_col = 'year'
            elif 'Year' in df.columns:
                year_col = 'Year'
            elif 'eng' in df.columns:
                # eng 컬럼이 연도 데이터인 경우
                year_col = 'eng'
            else:
                # 연도 컬럼이 없으면 인덱스 사용
                df['year'] = range(2016, 2016 + len(df))
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
                # 데이터가 너무 적으면 기본값
                return 0.02
            
            # 최근 10년 데이터만 사용 (2015-2024)
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
            # 오류 시 기본값 반환
            return 0.02  # 2% 기본값
    
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
            
            # 과거 10개년 성과 인상률 데이터를 기반으로 선형회귀 예측
            performance_rate = self._predict_performance_trend()
            
            # 반올림 처리를 위해 소수점 4자리까지만 유지
            prediction_value = round(float(prediction), 4)
            performance_rate = round(performance_rate, 4)
            
            print(f"🔍 Debug - Total prediction: {prediction_value:.4f} ({prediction_value*100:.2f}%)")
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
                    # 기본값
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
                # 타겟 컬럼 찾기
                target_col = 'wage_increase_total_sbl'
                if target_col not in df.columns:
                    # 다른 가능한 컬럼들 시도
                    for col in ['wage_increase_rate', 'target', 'wage_increase']:
                        if col in df.columns:
                            target_col = col
                            break
                
                # year 또는 eng 컬럼 찾기
                year_col = 'year' if 'year' in df.columns else 'eng' if 'eng' in df.columns else None
                
                if target_col in df.columns and year_col:
                    # 원본 데이터만 사용 (master_data는 이미 원본)
                    yearly_data = df.groupby(year_col)[target_col].first().dropna()
                    
                    # 과거 데이터 포맷팅
                    # 엑셀 구조: 2015년 feature → 2016년 임금인상률
                    # 따라서 year + 1로 표시
                    historical_data = []
                    
                    # Base-up과 Performance 컬럼 찾기
                    baseup_col = None
                    performance_col = None
                    for col in df.columns:
                        if 'wage_increase_bu' in col.lower() or 'base_up' in col.lower():
                            baseup_col = col
                        if 'wage_increase_mi' in col.lower() or 'performance' in col.lower():
                            performance_col = col
                    
                    for year, value in yearly_data.items():
                        if pd.notna(value):
                            # value가 이미 퍼센트인지 확인 (1보다 작으면 비율, 크면 퍼센트)
                            display_value = float(value) if value > 1 else float(value * 100)
                            # 실제 적용 연도는 feature 연도 + 1
                            actual_year = int(year) + 1
                            # 2025년 데이터는 제외 (2026년 예측값이므로)
                            if actual_year <= 2025:
                                data_point = {
                                    "year": actual_year,
                                    "value": display_value,
                                    "type": "historical"
                                }
                                
                                # Base-up과 Performance 데이터 추가 (있는 경우)
                                if baseup_col and year in df[year_col].values:
                                    baseup_value = df[df[year_col] == year][baseup_col].iloc[0]
                                    if pd.notna(baseup_value):
                                        data_point["base_up"] = float(baseup_value) if baseup_value > 1 else float(baseup_value * 100)
                                
                                if performance_col and year in df[year_col].values:
                                    perf_value = df[df[year_col] == year][performance_col].iloc[0]
                                    if pd.notna(perf_value):
                                        data_point["performance"] = float(perf_value) if perf_value > 1 else float(perf_value * 100)
                                
                                historical_data.append(data_point)
                    
                    # 2026년 예측 데이터 추가 (모델이 있는 경우)
                    from app.services.modeling_service import modeling_service
                    if modeling_service.current_model:
                        try:
                            # 간단한 예측값 추가 (실제 예측 로직은 나중에)
                            last_value = historical_data[-1]["value"] if historical_data else 3.5
                            prediction_data = {
                                "year": 2026,
                                "value": last_value * 1.05,  # 임시 예측값
                                "confidence_lower": last_value * 0.95,
                                "confidence_upper": last_value * 1.15,
                                "type": "prediction"
                            }
                            historical_data.append(prediction_data)
                        except:
                            pass
                    
                    return {
                        "message": "Trend data retrieved successfully",
                        "trend_data": historical_data,
                        "chart_config": {
                            "title": "임금인상률 추이 및 2026년 예측",
                            "y_axis_label": "임금인상률 (%)",
                            "x_axis_label": "연도"
                        }
                    }
            
            # 기본 데이터 반환
            return {
                "message": "Using default trend data",
                "trend_data": [],
                "chart_config": {
                    "title": "임금인상률 추이",
                    "y_axis_label": "임금인상률 (%)",
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