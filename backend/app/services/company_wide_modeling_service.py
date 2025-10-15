"""
전사 적정인력 산정 모델링 서비스 (R&A, tonggibon)
PyCaret 기반 회귀 모델링
"""
import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, Any, List, Optional, Tuple
import warnings
import io
import sys
import gc  # 🚨 메모리 관리
import logging
import os
from pathlib import Path

# PyCaret import
try:
    from pycaret.regression import (
        setup, compare_models, create_model, tune_model,
        finalize_model, predict_model, save_model, load_model,
        pull, get_config
    )
    PYCARET_AVAILABLE = True
except ImportError:
    PYCARET_AVAILABLE = False
    logging.warning("PyCaret not available")

from app.services.augmentation_service import augmentation_service

# 환경변수 또는 상대 경로로 DB 경로 설정
DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), '../../hwaseung_RnD.db'))

class CompanyWideModelingService:
    """전사 적정인력 산정 모델링 (R&A, tonggibon)"""

    def __init__(self):
        # Organization별 모델 및 실험 관리
        self.models = {
            'R&A': None,
            'tonggibon': None
        }
        self.experiments = {
            'R&A': None,
            'tonggibon': None
        }
        self.is_setup_complete = {
            'R&A': False,
            'tonggibon': False
        }
        self.feature_names = {
            'R&A': None,
            'tonggibon': None
        }
        self.model_results = {
            'R&A': None,
            'tonggibon': None
        }
        # 증강된 데이터 저장 (선택적)
        self.augmented_data = {
            'R&A': None,
            'tonggibon': None
        }
        self.augmentation_info = {
            'R&A': None,
            'tonggibon': None
        }

        # 모델 저장 경로
        self.models_dir = Path(__file__).parent.parent.parent / 'models'
        self.models_dir.mkdir(exist_ok=True)

        # 데이터 크기에 따른 모델 선택
        self.small_data_models = ['lr', 'ridge', 'lasso', 'en', 'dt']
        self.medium_data_models = ['lr', 'ridge', 'lasso', 'en', 'dt', 'rf', 'gbr']
        self.large_data_models = ['lr', 'ridge', 'lasso', 'en', 'dt', 'rf', 'gbr', 'xgboost', 'lightgbm']

        # 🚨 메모리 안전: 최대 데이터 크기 제한
        self.MAX_ROWS_FOR_MODELING = 10000
        self.MAX_FEATURES = 50

    def check_pycaret_availability(self) -> bool:
        """PyCaret 사용 가능 여부 확인"""
        return PYCARET_AVAILABLE

    def get_data_from_db(self, organization: str) -> pd.DataFrame:
        """
        DB에서 organization별 데이터 로드

        Args:
            organization: 'R&A' or 'tonggibon'

        Returns:
            DataFrame with features and target (headcount)
        """
        if organization not in ['R&A', 'tonggibon']:
            raise ValueError(f"Invalid organization: {organization}")

        try:
            conn = sqlite3.connect(DB_PATH)

            query = """
                SELECT *
                FROM company_wide_features
                WHERE organization = ?
                ORDER BY year
            """

            df = pd.read_sql_query(query, conn, params=(organization,))
            conn.close()

            if len(df) == 0:
                raise ValueError(f"No data found for organization: {organization}")

            logging.info(f"✅ Loaded {len(df)} rows for {organization}")
            return df

        except Exception as e:
            logging.error(f"Error loading data for {organization}: {e}")
            raise

    def prepare_data(self, organization: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        모델링을 위한 데이터 준비 (메모리 안전)

        Args:
            organization: 'R&A' or 'tonggibon'

        Returns:
            준비된 DataFrame과 정보 딕셔너리
        """
        # DB에서 데이터 로드
        df = self.get_data_from_db(organization)

        # 🚨 메모리 폭발 방지: 데이터 크기 체크
        if len(df) > self.MAX_ROWS_FOR_MODELING:
            logging.warning(f"⚠️ Dataset too large ({len(df)} rows). Sampling {self.MAX_ROWS_FOR_MODELING} rows.")
            df = df.sample(n=self.MAX_ROWS_FOR_MODELING, random_state=42)

        info = {
            'original_shape': df.shape,
            'organization': organization,
            'target_column': 'headcount',
            'dropped_columns': [],
            'sampled': len(df) > self.MAX_ROWS_FOR_MODELING
        }

        # 불필요한 컬럼 제거
        columns_to_drop = ['id', 'organization', 'created_at', 'updated_at']
        for col in columns_to_drop:
            if col in df.columns:
                df = df.drop(columns=[col])
                info['dropped_columns'].append(col)

        # year 컬럼 제거 (시계열이 아닌 회귀 문제)
        if 'year' in df.columns:
            df = df.drop(columns=['year'])
            info['dropped_columns'].append('year')
            logging.info("📊 Removed year column (regression problem, not time series)")

        # headcount (target) 결측값 제거
        if 'headcount' not in df.columns:
            raise ValueError("Target column 'headcount' not found")

        initial_rows = len(df)
        df = df.dropna(subset=['headcount'])
        removed_rows = initial_rows - len(df)

        if removed_rows > 0:
            info['removed_target_missing'] = removed_rows
            logging.info(f"📊 Removed {removed_rows} rows with missing headcount")

        if len(df) < 3:
            raise ValueError(f"Insufficient data: only {len(df)} rows with valid headcount")

        # 숫자형 변환
        for col in df.columns:
            if col != 'headcount':
                df[col] = pd.to_numeric(df[col], errors='coerce')

        df['headcount'] = pd.to_numeric(df['headcount'], errors='coerce')
        df = df.dropna(subset=['headcount'])

        # 최종 정보
        info['final_shape'] = df.shape
        info['feature_count'] = len(df.columns) - 1
        info['data_rows'] = len(df)

        logging.info(f"✅ Data prepared: {info['final_shape']}, {info['feature_count']} features")

        return df, info

    def augment_data(
        self,
        organization: str,
        target_size: int = 200,
        method: str = 'auto'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        데이터 증강

        Args:
            organization: 'R&A' or 'tonggibon'
            target_size: 목표 데이터 크기 (기본 200)
            method: 증강 방법 ('auto', 'noise', 'mixup')

        Returns:
            증강된 DataFrame과 증강 정보
        """
        df, prep_info = self.prepare_data(organization)

        logging.info(f"🔄 Augmenting data for {organization}: {len(df)} → {target_size}")

        # augmentation_service 사용
        augmented_df, aug_info = augmentation_service.smart_augment(
            df=df,
            target_column='headcount',
            year_column=None,  # year는 이미 제거됨
            target_size=target_size,
            method=method
        )

        info = {
            **prep_info,
            **aug_info,
            'augmentation_successful': True
        }

        logging.info(f"✅ Augmentation complete: {len(augmented_df)} rows")

        return augmented_df, info

    def augment_and_store(
        self,
        organization: str,
        target_size: int = 200,
        method: str = 'auto'
    ) -> Dict[str, Any]:
        """
        데이터 증강 후 저장 (setup 전 선택적으로 실행)

        Args:
            organization: 'R&A' or 'tonggibon'
            target_size: 목표 데이터 크기
            method: 증강 방법

        Returns:
            증강 결과 정보
        """
        augmented_df, info = self.augment_data(organization, target_size, method)

        # 증강된 데이터 저장
        self.augmented_data[organization] = augmented_df
        self.augmentation_info[organization] = info

        return {
            'message': f'Data augmentation completed for {organization}',
            'organization': organization,
            'original_size': info.get('original_shape', (0,))[0],
            'augmented_size': len(augmented_df),
            'feature_count': info.get('feature_count', 0),
            'method': method
        }

    def setup_pycaret_environment(
        self,
        organization: str,
        session_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        PyCaret 환경 설정 (증강 여부 무관)

        증강된 데이터가 있으면 사용, 없으면 원본 데이터 사용

        Args:
            organization: 'R&A' or 'tonggibon'
            session_id: 재현성을 위한 시드값

        Returns:
            설정 정보
        """
        if not self.check_pycaret_availability():
            raise RuntimeError("PyCaret is not available")

        # 🔧 FIX: PyCaret 전역 상태 문제 완화
        # 재setup이면 현재 organization만 정리
        if self.is_setup_complete.get(organization, False):
            logging.info(f"🔄 Re-setup detected for {organization}, clearing previous state")
            if self.experiments.get(organization) is not None:
                del self.experiments[organization]
            if self.model_results.get(organization) is not None:
                if 'best_models' in self.model_results[organization]:
                    del self.model_results[organization]['best_models']
                del self.model_results[organization]

            self.experiments[organization] = None
            self.model_results[organization] = None
            gc.collect()

        # ⚠️ 참고: PyCaret은 전역 상태를 사용하므로, 두 organization을 동시에 사용할 수 없습니다.
        # compare/train 시 자동으로 재setup하도록 구현되어 있습니다.

        # 증강된 데이터가 있으면 사용, 없으면 원본 데이터 준비
        if self.augmented_data.get(organization) is not None:
            logging.info(f"📊 Using augmented data for {organization}")
            ml_data = self.augmented_data[organization]
            data_info = self.augmentation_info[organization]
        else:
            logging.info(f"📊 Using original data for {organization}")
            ml_data, data_info = self.prepare_data(organization)

        # Session ID 설정 (organization별 고정값)
        if session_id is None:
            session_id = 42 if organization == 'R&A' else 43

        # 출력 억제
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            # 🚨 메모리 정리 (setup 전)
            gc.collect()

            # 데이터 크기에 따른 설정 (메모리 최소화)
            data_size = len(ml_data)

            # 🚨 메모리 에러 방지: 모든 케이스에서 최소 설정
            train_size = 0.9  # 테스트 셋 최소화
            fold = 2  # 항상 2-fold
            normalize = True  # 정규화는 유지 (성능)
            transformation = False  # 변환 비활성화 (메모리 절약)

            # PyCaret setup (메모리 안전 모드)
            exp = setup(
                data=ml_data,
                target='headcount',
                session_id=session_id,
                train_size=train_size,
                html=False,
                verbose=False,

                # 🚨 메모리 안전 설정
                use_gpu=False,
                n_jobs=1,  # 병렬 처리 비활성화

                # 전처리
                imputation_type='simple',
                numeric_imputation='mean',
                normalize=normalize,
                normalize_method='zscore' if normalize else None,
                transformation=transformation,
                transformation_method='yeo-johnson' if transformation else None,
                remove_outliers=False,
                remove_multicollinearity=True,
                multicollinearity_threshold=0.9,

                fold=2  # 항상 2-fold (메모리 최소화)
            )

            self.experiments[organization] = exp
            self.is_setup_complete[organization] = True

            # Feature names 저장
            X_train = get_config('X_train')
            if X_train is not None:
                self.feature_names[organization] = list(X_train.columns)
                logging.info(f"📊 Stored {len(self.feature_names[organization])} feature names")

        except Exception as e:
            logging.error(f"PyCaret setup failed: {e}")
            raise
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return {
            'message': f'PyCaret environment setup completed for {organization}',
            'organization': organization,
            'data_info': data_info,
            'data_size': data_size,
            'train_size': train_size,
            'fold': fold,
            'features': self.feature_names[organization]
        }

    def compare_models(self, organization: str, n_select: int = 3) -> Dict[str, Any]:
        """
        모델 비교

        Args:
            organization: 'R&A' or 'tonggibon'
            n_select: 선택할 최상위 모델 수

        Returns:
            비교 결과
        """
        # 🔧 FIX: PyCaret 전역 상태 문제 해결 - 항상 재setup
        # PyCaret은 전역 상태를 사용하므로, 매번 해당 organization으로 재setup 필요
        logging.info(f"🔄 Re-setting up PyCaret for {organization} before compare")
        self.setup_pycaret_environment(organization)

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            # 🚨 메모리 정리 (비교 전)
            gc.collect()

            # 데이터 크기에 따른 모델 선택 (메모리 안전)
            X_train = get_config('X_train')
            data_size = len(X_train) if X_train is not None else 100

            # 🚨 메모리 안전: 모델 수 최소화 (메모리 에러 방지)
            if data_size < 30:
                models_to_use = ['lr']  # 1개만
                n_select = 1
                safe_fold = 2
            elif data_size < 100:
                models_to_use = ['lr', 'ridge']  # 2개만 (가벼운 모델만)
                n_select = min(1, n_select)  # 1개만 선택
                safe_fold = 2
            elif data_size < 500:
                models_to_use = ['lr', 'ridge', 'lasso']  # 3개만 (선형 모델만)
                n_select = min(1, n_select)  # 1개만 선택
                safe_fold = 2
            else:
                models_to_use = ['lr', 'ridge', 'rf']  # 3개까지만 (트리 모델 1개만)
                n_select = min(1, n_select)  # 1개만 선택
                safe_fold = 2

            # 모델 비교 (메모리 안전)
            best_models = compare_models(
                include=models_to_use,
                sort='R2',
                n_select=min(n_select, len(models_to_use)),
                verbose=False,
                fold=safe_fold,
                errors='ignore'
            )

            if not isinstance(best_models, list):
                best_models = [best_models]

            comparison_results = pull()

            self.model_results[organization] = {
                'best_models': best_models,
                'comparison_df': comparison_results,
                'recommended_model': best_models[0] if best_models else None
            }

        except Exception as e:
            logging.error(f"Model comparison failed: {e}")
            # Fallback to linear regression
            linear_model = create_model('lr', verbose=False)
            self.model_results[organization] = {
                'best_models': [linear_model],
                'comparison_df': None,
                'recommended_model': linear_model,
                'fallback_used': True
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

            # 🚨 메모리 정리 (비교 후)
            gc.collect()

        # 비교 결과를 딕셔너리로 변환
        comparison_df = self.model_results[organization]['comparison_df']
        comparison_data = comparison_df.to_dict(orient='records') if comparison_df is not None else []

        return {
            'message': f'Model comparison completed for {organization}',
            'organization': organization,
            'models_compared': len(models_to_use),
            'best_model_count': len(self.model_results[organization]['best_models']),
            'recommended_model_type': type(self.model_results[organization]['recommended_model']).__name__,
            'comparison_data': comparison_data
        }

    def train_model(self, organization: str, model_name: str) -> Dict[str, Any]:
        """
        특정 모델 학습

        Args:
            organization: 'R&A' or 'tonggibon'
            model_name: 모델 이름 (예: 'rf', 'gbr', 'lr')

        Returns:
            학습 결과
        """
        # 🔧 FIX: PyCaret 전역 상태 문제 해결 - 항상 재setup
        # PyCaret은 전역 상태를 사용하므로, 매번 해당 organization으로 재setup 필요
        logging.info(f"🔄 Re-setting up PyCaret for {organization} before training")
        self.setup_pycaret_environment(organization)

        old_stdout = sys.stdout
        old_stderr = sys.stderr

        try:
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            # 모델 생성
            models_without_random_state = ['lr', 'ridge', 'lasso', 'en', 'lar', 'llar', 'omp', 'br', 'ard', 'par', 'ransac', 'tr', 'huber']

            if model_name in models_without_random_state:
                model = create_model(model_name, verbose=False)
            else:
                model = create_model(model_name, verbose=False, random_state=42)

            # 튜닝
            try:
                tuned_model = tune_model(model, optimize='R2', verbose=False)
            except:
                tuned_model = model

            # 최종 모델
            try:
                final_model = finalize_model(tuned_model)
            except:
                final_model = tuned_model

            self.models[organization] = final_model

            # 모델 저장
            self._save_model(organization, model_name)

            # 평가 메트릭
            metrics = pull()
            metrics_dict = metrics.to_dict(orient='records')[0] if metrics is not None else {}

            # 학습 후 메모리 정리
            self.cleanup_after_training(organization)

        except Exception as e:
            logging.error(f"Model training failed: {e}")
            raise
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        return {
            'message': f'Model {model_name} trained successfully for {organization}',
            'organization': organization,
            'model_type': type(self.models[organization]).__name__,
            'model_name': model_name,
            'model_saved': True,
            'metrics': metrics_dict
        }

    def _save_model(self, organization: str, model_name: str = None) -> bool:
        """
        모델 저장 (최신 1개만 유지)

        이전 버전의 모델 파일들을 모두 삭제하고 최신 모델만 저장합니다.
        """
        try:
            if self.models[organization] is None:
                return False

            import glob

            # 1. 기존 모델 파일들 모두 삭제
            pattern = str(self.models_dir / f"company_wide_{organization}_*.pkl")
            old_files = glob.glob(pattern)
            for old_file in old_files:
                try:
                    Path(old_file).unlink()
                    logging.info(f"🗑️ Deleted old model: {Path(old_file).name}")
                except Exception as e:
                    logging.warning(f"Failed to delete {old_file}: {e}")

            # 2. 최신 모델만 latest 이름으로 저장
            latest_path = self.models_dir / f"company_wide_{organization}_latest"
            save_model(self.models[organization], str(latest_path), verbose=False)

            logging.info(f"✅ Model saved (latest only): company_wide_{organization}_latest.pkl")
            return True

        except Exception as e:
            logging.error(f"Failed to save model: {e}")
            return False

    def load_model(self, organization: str, filename: str = 'latest') -> Dict[str, Any]:
        """저장된 모델 로드"""
        try:
            if filename == 'latest':
                filepath = self.models_dir / f"company_wide_{organization}_latest"
            else:
                filepath = self.models_dir / filename

            self.models[organization] = load_model(str(filepath))

            return {
                'message': f'Model loaded successfully for {organization}',
                'organization': organization,
                'model_type': type(self.models[organization]).__name__
            }

        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def get_status(self, organization: str) -> Dict[str, Any]:
        """모델링 상태 확인"""
        has_data = False
        data_rows = 0

        try:
            df = self.get_data_from_db(organization)
            has_data = len(df) > 0
            data_rows = len(df)
        except:
            pass

        # 증강 정보
        is_augmented = self.augmented_data.get(organization) is not None
        augmented_size = len(self.augmented_data[organization]) if is_augmented else 0

        return {
            'organization': organization,
            'pycaret_available': self.check_pycaret_availability(),
            'has_data': has_data,
            'data_rows': data_rows,
            'is_augmented': is_augmented,
            'augmented_size': augmented_size,
            'environment_setup': self.is_setup_complete.get(organization, False),
            'model_trained': self.models.get(organization) is not None,
            'models_compared': self.model_results.get(organization) is not None,
            'current_model_type': type(self.models[organization]).__name__ if self.models.get(organization) else None
        }

    def clear_models(self, organization: str = None) -> Dict[str, Any]:
        """모델 및 실험 초기화"""
        import gc

        if organization:
            # 명시적 메모리 해제
            if self.models.get(organization) is not None:
                del self.models[organization]
            if self.experiments.get(organization) is not None:
                del self.experiments[organization]
            if self.model_results.get(organization) is not None:
                if 'best_models' in self.model_results[organization]:
                    del self.model_results[organization]['best_models']
                del self.model_results[organization]
            if self.augmented_data.get(organization) is not None:
                del self.augmented_data[organization]
            if self.augmentation_info.get(organization) is not None:
                del self.augmentation_info[organization]

            # 초기화
            self.models[organization] = None
            self.experiments[organization] = None
            self.is_setup_complete[organization] = False
            self.model_results[organization] = None
            self.feature_names[organization] = None
            self.augmented_data[organization] = None
            self.augmentation_info[organization] = None
            message = f'Models cleared for {organization}'
        else:
            # 모든 organization 메모리 해제
            for org in ['R&A', 'tonggibon']:
                if self.models.get(org) is not None:
                    del self.models[org]
                if self.experiments.get(org) is not None:
                    del self.experiments[org]
                if self.model_results.get(org) is not None:
                    if 'best_models' in self.model_results[org]:
                        del self.model_results[org]['best_models']
                    del self.model_results[org]
                if self.augmented_data.get(org) is not None:
                    del self.augmented_data[org]
                if self.augmentation_info.get(org) is not None:
                    del self.augmentation_info[org]

            # 초기화
            self.models = {'R&A': None, 'tonggibon': None}
            self.experiments = {'R&A': None, 'tonggibon': None}
            self.is_setup_complete = {'R&A': False, 'tonggibon': False}
            self.model_results = {'R&A': None, 'tonggibon': None}
            self.feature_names = {'R&A': None, 'tonggibon': None}
            self.augmented_data = {'R&A': None, 'tonggibon': None}
            self.augmentation_info = {'R&A': None, 'tonggibon': None}
            message = 'All models cleared'

        # 가비지 컬렉션 강제 실행
        gc.collect()

        return {'message': message}

    def cleanup_after_training(self, organization: str) -> None:
        """학습 후 불필요한 메모리 정리"""
        import gc

        # 모델 결과에서 best_models 리스트 삭제 (recommended_model만 유지)
        if self.model_results.get(organization) is not None:
            if 'best_models' in self.model_results[organization]:
                recommended = self.model_results[organization].get('recommended_model')
                self.model_results[organization]['best_models'] = [recommended] if recommended else []

        # 가비지 컬렉션
        gc.collect()

# 싱글톤 인스턴스
company_wide_modeling_service = CompanyWideModelingService()
