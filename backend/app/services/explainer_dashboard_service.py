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
from app.services.data_service import data_service

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
            
            # data_service에서 한글 컬럼명 가져오기
            feature_descriptions = data_service.get_display_names(feature_names)
            
            # 한글 → 영문 매핑 생성
            korean_to_english = {}
            english_to_korean = {}
            korean_feature_names = []
            for feat in feature_names:
                korean_name = feature_descriptions.get(feat, feat)
                korean_feature_names.append(korean_name)
                korean_to_english[korean_name] = feat
                english_to_korean[feat] = korean_name
            
            # 모델 래퍼 클래스 정의 (한글 컬럼명 → 영문 컬럼명 변환)
            class ModelWrapper:
                def __init__(self, original_model, korean_to_english):
                    self.model = original_model
                    self.korean_to_english = korean_to_english
                    
                def predict(self, X):
                    # 한글 컬럼명을 영문으로 변환
                    if isinstance(X, pd.DataFrame):
                        X_english = X.rename(columns=self.korean_to_english)
                    else:
                        # numpy array인 경우 DataFrame으로 변환
                        X_english = pd.DataFrame(X, columns=list(self.korean_to_english.keys()))
                        X_english = X_english.rename(columns=self.korean_to_english)
                    return self.model.predict(X_english)
                
                def __getattr__(self, name):
                    # 다른 속성들은 원본 모델로 전달
                    return getattr(self.model, name)
            
            # 래핑된 모델 생성
            wrapped_model = ModelWrapper(model, korean_to_english)
            
            # X_test에 한글 feature names 설정
            if isinstance(X_test, pd.DataFrame):
                X_test.columns = korean_feature_names
            else:
                X_test = pd.DataFrame(X_test, columns=korean_feature_names)
            
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
            
            # Explainer 생성 (회귀 모델) - 기본 파라미터만 사용
            explainer = RegressionExplainer(
                wrapped_model,  # 래핑된 모델 사용 
                X_test, 
                y_test,
                units='%'  # 단위 설정
            )
            
            # 대시보드 생성 - 기본 설정만 사용
            self.dashboard = ExplainerDashboard(
                explainer,
                title="임금인상률 예측 모델 분석",
                description="2026년 임금인상률 예측 모델의 상세 분석 대시보드",
                port=self.dashboard_port,
                mode='dash'
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