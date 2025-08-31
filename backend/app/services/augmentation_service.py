import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from scipy import stats

class AugmentationService:
    """
    데이터 증강 전문 서비스
    작은 데이터셋을 위한 다양한 증강 기법 제공
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def augment_with_noise(
        self, 
        df: pd.DataFrame, 
        target_column: str,
        year_column: Optional[str] = None,
        factor: int = 10,
        noise_level: float = 0.02,
        preserve_distribution: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        가우시안 노이즈를 추가한 데이터 증강
        
        Args:
            df: 원본 데이터프레임
            target_column: 타겟 컬럼 (변경하지 않음)
            year_column: 연도 컬럼 (변경하지 않음)
            factor: 증강 배수
            noise_level: 노이즈 수준 (0.01 = 1%, 0.02 = 2%)
            preserve_distribution: 원본 분포 유지 여부
        """
        original_size = len(df)
        augmented_rows = []
        
        # 보호할 컬럼 식별
        protected_columns = [target_column]
        if year_column:
            protected_columns.append(year_column)
        
        # 수치형 컬럼만 선택 (보호 컬럼 제외)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        augmentable_columns = [col for col in numeric_columns if col not in protected_columns]
        
        self.logger.info(f"Augmenting {len(augmentable_columns)} columns, protecting {protected_columns}")
        
        for idx, row in df.iterrows():
            # 원본 행 추가
            augmented_rows.append(row.to_dict())
            
            # 증강된 행 생성
            for i in range(factor - 1):
                new_row = row.to_dict()
                
                for col in augmentable_columns:
                    if pd.notna(new_row[col]) and new_row[col] != 0:
                        # 분포 보존 옵션
                        if preserve_distribution:
                            # 원본 값의 스케일에 맞는 노이즈
                            noise = np.random.normal(0, abs(new_row[col]) * noise_level)
                        else:
                            # 고정 스케일 노이즈
                            noise = np.random.normal(0, noise_level)
                        
                        new_row[col] = new_row[col] + noise
                
                augmented_rows.append(new_row)
        
        # 증강된 데이터프레임 생성
        augmented_df = pd.DataFrame(augmented_rows)
        
        # 연도별로 정렬 (시계열 데이터인 경우)
        if year_column and year_column in augmented_df.columns:
            augmented_df = augmented_df.sort_values(year_column).reset_index(drop=True)
        
        info = {
            "method": "gaussian_noise",
            "original_size": original_size,
            "augmented_size": len(augmented_df),
            "factor": factor,
            "noise_level": noise_level,
            "preserve_distribution": preserve_distribution,
            "augmented_columns": augmentable_columns,
            "protected_columns": protected_columns
        }
        
        return augmented_df, info
    
    def augment_with_interpolation(
        self,
        df: pd.DataFrame,
        target_column: str,
        year_column: Optional[str] = None,
        factor: int = 10,
        method: str = 'linear'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        보간법을 사용한 데이터 증강 (시계열 데이터에 적합)
        
        Args:
            df: 원본 데이터프레임
            target_column: 타겟 컬럼
            year_column: 연도 컬럼
            factor: 증강 배수
            method: 보간 방법 ('linear', 'polynomial', 'spline')
        """
        original_size = len(df)
        
        # 연도 컬럼이 있으면 시계열로 처리
        if year_column and year_column in df.columns:
            df_sorted = df.sort_values(year_column).copy()
            
            # 각 연도 사이에 데이터 포인트 추가
            augmented_rows = []
            
            for i in range(len(df_sorted) - 1):
                current_row = df_sorted.iloc[i]
                next_row = df_sorted.iloc[i + 1]
                
                # 현재 행 추가
                augmented_rows.append(current_row.to_dict())
                
                # 보간된 행 생성
                n_interpolated = factor - 1
                for j in range(1, n_interpolated + 1):
                    alpha = j / (n_interpolated + 1)
                    interpolated_row = {}
                    
                    for col in df_sorted.columns:
                        if col == year_column:
                            # 연도는 보간하지 않음
                            interpolated_row[col] = current_row[col]
                        elif col == target_column:
                            # 타겟은 보간하지 않음 (예측 대상)
                            interpolated_row[col] = current_row[col]
                        elif pd.api.types.is_numeric_dtype(df_sorted[col]):
                            # 수치형 컬럼 보간
                            if pd.notna(current_row[col]) and pd.notna(next_row[col]):
                                if method == 'linear':
                                    interpolated_row[col] = (1 - alpha) * current_row[col] + alpha * next_row[col]
                                else:
                                    # 더 복잡한 보간은 추후 구현
                                    interpolated_row[col] = (1 - alpha) * current_row[col] + alpha * next_row[col]
                            else:
                                interpolated_row[col] = current_row[col]
                        else:
                            # 범주형 컬럼은 현재 값 유지
                            interpolated_row[col] = current_row[col]
                    
                    augmented_rows.append(interpolated_row)
            
            # 마지막 행 추가
            augmented_rows.append(df_sorted.iloc[-1].to_dict())
            
            augmented_df = pd.DataFrame(augmented_rows)
        else:
            # 연도 컬럼이 없으면 노이즈 방식으로 대체
            return self.augment_with_noise(df, target_column, year_column, factor)
        
        info = {
            "method": f"interpolation_{method}",
            "original_size": original_size,
            "augmented_size": len(augmented_df),
            "factor": factor,
            "interpolation_method": method
        }
        
        return augmented_df, info
    
    def augment_with_mixup(
        self,
        df: pd.DataFrame,
        target_column: str,
        year_column: Optional[str] = None,
        factor: int = 10,
        alpha: float = 0.2
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Mixup 기법을 사용한 데이터 증강
        두 샘플의 선형 조합으로 새로운 샘플 생성
        
        Args:
            df: 원본 데이터프레임
            target_column: 타겟 컬럼
            year_column: 연도 컬럼
            factor: 증강 배수
            alpha: Beta 분포 파라미터 (혼합 비율 제어)
        """
        original_size = len(df)
        augmented_rows = list(df.to_dict('records'))
        
        # 보호할 컬럼
        protected_columns = [target_column]
        if year_column:
            protected_columns.append(year_column)
        
        # 수치형 컬럼만 선택
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        mixable_columns = [col for col in numeric_columns if col not in protected_columns]
        
        # 필요한 수만큼 새로운 샘플 생성
        n_new_samples = (factor - 1) * original_size
        
        for _ in range(n_new_samples):
            # 랜덤하게 두 샘플 선택
            idx1, idx2 = np.random.choice(original_size, 2, replace=False)
            row1 = df.iloc[idx1]
            row2 = df.iloc[idx2]
            
            # Beta 분포에서 혼합 비율 샘플링
            lam = np.random.beta(alpha, alpha)
            
            # 새로운 행 생성
            new_row = {}
            for col in df.columns:
                if col in mixable_columns:
                    # 수치형 컬럼은 선형 조합
                    if pd.notna(row1[col]) and pd.notna(row2[col]):
                        new_row[col] = lam * row1[col] + (1 - lam) * row2[col]
                    else:
                        new_row[col] = row1[col]
                else:
                    # 보호 컬럼은 첫 번째 샘플 값 사용
                    new_row[col] = row1[col]
            
            augmented_rows.append(new_row)
        
        augmented_df = pd.DataFrame(augmented_rows)
        
        # 연도별로 정렬
        if year_column and year_column in augmented_df.columns:
            augmented_df = augmented_df.sort_values(year_column).reset_index(drop=True)
        
        info = {
            "method": "mixup",
            "original_size": original_size,
            "augmented_size": len(augmented_df),
            "factor": factor,
            "alpha": alpha,
            "mixable_columns": mixable_columns,
            "protected_columns": protected_columns
        }
        
        return augmented_df, info
    
    def smart_augment(
        self,
        df: pd.DataFrame,
        target_column: str,
        year_column: Optional[str] = None,
        target_size: Optional[int] = None,
        method: str = 'auto'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        데이터 특성에 따라 자동으로 최적의 증강 방법 선택
        
        Args:
            df: 원본 데이터프레임
            target_column: 타겟 컬럼
            year_column: 연도 컬럼
            target_size: 목표 데이터 크기
            method: 'auto', 'noise', 'interpolation', 'mixup'
        """
        original_size = len(df)
        
        # 목표 크기 결정
        if target_size is None:
            if original_size < 20:
                target_size = 200  # 매우 작은 데이터는 200개로
            elif original_size < 50:
                target_size = 500  # 작은 데이터는 500개로
            elif original_size < 100:
                target_size = 1000  # 중간 데이터는 1000개로
            else:
                target_size = original_size * 5  # 큰 데이터는 5배로
        
        factor = max(2, target_size // original_size)
        
        # 자동 방법 선택
        if method == 'auto':
            if year_column and year_column in df.columns:
                # 시계열 데이터는 보간법 우선
                if original_size < 10:
                    method = 'noise'  # 너무 작으면 노이즈
                else:
                    method = 'interpolation'
            else:
                # 일반 데이터는 노이즈 또는 mixup
                if original_size < 20:
                    method = 'noise'
                else:
                    method = 'mixup'
        
        self.logger.info(f"Smart augment: {original_size} -> {target_size} using {method}")
        
        # 선택된 방법으로 증강
        if method == 'noise':
            return self.augment_with_noise(df, target_column, year_column, factor, noise_level=0.02)
        elif method == 'interpolation':
            return self.augment_with_interpolation(df, target_column, year_column, factor)
        elif method == 'mixup':
            return self.augment_with_mixup(df, target_column, year_column, factor)
        else:
            raise ValueError(f"Unknown augmentation method: {method}")

# 싱글톤 인스턴스
augmentation_service = AugmentationService()