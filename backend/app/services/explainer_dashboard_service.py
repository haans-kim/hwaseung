"""
ExplainerDashboard 서비스
"""
import os
import pickle
import logging
import threading
import time
from typing import Optional, Dict, Any
from pathlib import Path

import pandas as pd
import numpy as np
from explainerdashboard import ClassifierExplainer, RegressionExplainer, ExplainerDashboard

logger = logging.getLogger(__name__)


class ExplainerDashboardService:
    """ExplainerDashboard 생성 및 관리"""
    
    def __init__(self):
        self.dashboard: Optional[ExplainerDashboard] = None
        self.dashboard_thread: Optional[threading.Thread] = None
        self.dashboard_port: int = 8050
        self.is_running: bool = False
        self.dashboard_url: Optional[str] = None
        
    def create_dashboard(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                        feature_names: list, model_name: str = "Wage Increase Model") -> Dict[str, Any]:
        """ExplainerDashboard 생성"""
        try:
            logger.info("Creating ExplainerDashboard...")
            
            # 모델이 이미 있으면 종료
            if self.is_running:
                self.stop_dashboard()
            
            # X_test에 feature names 설정
            if isinstance(X_test, pd.DataFrame):
                X_test.columns = feature_names
            else:
                X_test = pd.DataFrame(X_test, columns=feature_names)
            
            # 인덱스를 연도와 증강 번호로 설정
            # 예: "2015_원본", "2015_증강1", ... "2016_원본", "2016_증강1", ...
            if len(X_test) > 0:
                # 연도별로 몇 개의 데이터가 있는지 추정 (예: 10개씩)
                samples_per_year = 10
                years = []
                for i in range(len(X_test)):
                    year_offset = i // samples_per_year
                    sample_num = i % samples_per_year
                    base_year = 2015 + year_offset
                    if sample_num == 0:
                        years.append(f"{base_year}년_원본")
                    else:
                        years.append(f"{base_year}년_증강{sample_num}")
                X_test.index = years
            
            # Feature descriptions 생성 (dictionary 형태)
            feature_descriptions = {
                'gdp_growth_kr': '한국 GDP 성장률',
                'gdp_growth_usa': '미국 GDP 성장률',
                'cpi_kr': '한국 소비자물가지수',
                'cpi_usa': '미국 소비자물가지수',
                'unemployment_rate_kr': '한국 실업률',
                'unemployment_rate_usa': '미국 실업률',
                'exchange_rate_change_krw': '원화 환율 변동률',
                'minimum_wage_increase_kr': '한국 최저임금인상률',
                'public_sector_wage_increase': '공공부문 임금인상률',
                'private_sector_wage_increase': '민간부문 임금인상률',
                'wage_increase_bu_group': 'BU그룹 임금인상률',
                'wage_increase_mi_group': 'MI그룹 임금인상률',
                'wage_increase_total_group': '그룹 전체 임금인상률',
                'wage_increase_ce': 'CE 임금인상률',
                'market_size_growth_rate': '시장규모 성장률',
                'labor_cost_rate_sbl': 'SBL 인건비율',
                'labor_cost_per_employee_sbl': 'SBL 인당인건비',
                'labor_to_revenue_sbl': 'SBL 매출대비인건비',
                'hcva_sbl': 'SBL 인력부가가치',
                'hcva_ce': 'CE 인력부가가치',
                'hcroi_sbl': 'SBL 인력투자수익률',
                'hcroi_ce': 'CE 인력투자수익률',
                'esi_usa': '미국 ESI 지수',
                'vix_usa': '미국 VIX 지수'
            }
            
            # 실제 feature names에 맞춰 descriptions 필터링
            descriptions_dict = {feat: feature_descriptions.get(feat, feat) for feat in feature_names}
            
            # Explainer 생성 (회귀 모델)
            explainer = RegressionExplainer(
                model, 
                X_test, 
                y_test,
                model_output='raw',  # 원본 출력값 사용
                units='%',  # 단위 설정
                descriptions=descriptions_dict,  # feature 설명 (dict)
                target='임금인상률'  # 타겟 변수명
            )
            
            # 대시보드 생성
            self.dashboard = ExplainerDashboard(
                explainer,
                title="임금인상률 예측 모델 분석",
                description="2025년 임금인상률 예측 모델의 상세 분석 대시보드",
                simple=False,  # 전체 기능 사용
                hide_poweredby=True,  # Powered by 숨김
                
                # 포트 설정
                port=self.dashboard_port,
                
                # 모드 설정
                mode='dash',  # inline, external, jupyterlab 중 선택
                
                # 기타 설정
                bootstrap=True
            )
            
            # 별도 스레드에서 대시보드 실행
            self.dashboard_thread = threading.Thread(
                target=self._run_dashboard,
                daemon=True
            )
            self.dashboard_thread.start()
            
            # 대시보드가 시작될 때까지 대기
            time.sleep(3)
            
            self.is_running = True
            self.dashboard_url = f"http://localhost:{self.dashboard_port}"
            
            logger.info(f"ExplainerDashboard started at {self.dashboard_url}")
            
            return {
                "success": True,
                "url": self.dashboard_url,
                "port": self.dashboard_port,
                "message": "ExplainerDashboard가 성공적으로 생성되었습니다."
            }
            
        except Exception as e:
            logger.error(f"Failed to create ExplainerDashboard: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "message": "ExplainerDashboard 생성 중 오류가 발생했습니다."
            }
    
    def _run_dashboard(self):
        """대시보드 실행 (별도 스레드)"""
        try:
            self.dashboard.run(use_waitress=True)
        except Exception as e:
            logger.error(f"Dashboard runtime error: {str(e)}")
            self.is_running = False
    
    def stop_dashboard(self):
        """대시보드 중지"""
        try:
            if self.dashboard:
                # Dash 서버 종료
                if hasattr(self.dashboard, 'app') and hasattr(self.dashboard.app, 'server'):
                    func = self.dashboard.app.server.shutdown
                    func()
                
                self.dashboard = None
                self.is_running = False
                self.dashboard_url = None
                
                logger.info("ExplainerDashboard stopped")
                
        except Exception as e:
            logger.error(f"Failed to stop dashboard: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """대시보드 상태 확인"""
        return {
            "is_running": self.is_running,
            "url": self.dashboard_url,
            "port": self.dashboard_port if self.is_running else None
        }
    
    def save_explainer(self, filepath: str):
        """Explainer 저장"""
        try:
            if self.dashboard and hasattr(self.dashboard, 'explainer'):
                self.dashboard.explainer.dump(filepath)
                logger.info(f"Explainer saved to {filepath}")
                return True
        except Exception as e:
            logger.error(f"Failed to save explainer: {str(e)}")
        return False
    
    def load_explainer(self, filepath: str) -> bool:
        """저장된 Explainer 로드"""
        try:
            from explainerdashboard import ExplainerDashboard
            
            # 저장된 explainer 로드
            dashboard = ExplainerDashboard.from_file(filepath)
            
            # 대시보드 실행
            self.dashboard = dashboard
            self.dashboard_thread = threading.Thread(
                target=self._run_dashboard,
                daemon=True
            )
            self.dashboard_thread.start()
            
            time.sleep(3)
            
            self.is_running = True
            self.dashboard_url = f"http://localhost:{self.dashboard_port}"
            
            logger.info(f"Explainer loaded and dashboard started at {self.dashboard_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load explainer: {str(e)}")
            return False


# 싱글톤 인스턴스
explainer_dashboard_service = ExplainerDashboardService()