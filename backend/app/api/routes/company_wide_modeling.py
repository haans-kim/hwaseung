"""
전사 적정인력 산정 모델링 API (R&A, tonggibon)
"""
from fastapi import APIRouter, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
from pydantic import BaseModel

from app.services.company_wide_modeling_service import company_wide_modeling_service

router = APIRouter(
    prefix="/company-wide/modeling",
    tags=["company-wide-modeling"],
    responses={404: {"description": "Not found"}},
)

class SetupRequest(BaseModel):
    organization: str
    use_augmentation: bool = True
    target_size: int = 200

class CompareRequest(BaseModel):
    organization: str
    n_select: int = 3

class TrainRequest(BaseModel):
    organization: str
    model_name: str

@router.post("/setup")
async def setup_pycaret(request: SetupRequest) -> Dict[str, Any]:
    """
    PyCaret 환경 설정 및 데이터 증강

    Args:
        organization: 'R&A' or 'tonggibon'
        use_augmentation: 데이터 증강 사용 여부
        target_size: 증강 목표 크기

    Returns:
        설정 결과
    """
    if request.organization not in ['R&A', 'tonggibon']:
        raise HTTPException(
            status_code=400,
            detail="organization must be 'R&A' or 'tonggibon'"
        )

    try:
        result = company_wide_modeling_service.setup_pycaret_environment(
            organization=request.organization,
            use_augmentation=request.use_augmentation,
            target_size=request.target_size
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

@router.post("/compare")
async def compare_models(request: CompareRequest) -> Dict[str, Any]:
    """
    모델 비교

    Args:
        organization: 'R&A' or 'tonggibon'
        n_select: 선택할 최상위 모델 수

    Returns:
        모델 비교 결과
    """
    if request.organization not in ['R&A', 'tonggibon']:
        raise HTTPException(
            status_code=400,
            detail="organization must be 'R&A' or 'tonggibon'"
        )

    try:
        result = company_wide_modeling_service.compare_models(
            organization=request.organization,
            n_select=request.n_select
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")

@router.post("/train")
async def train_model(request: TrainRequest) -> Dict[str, Any]:
    """
    특정 모델 학습

    Args:
        organization: 'R&A' or 'tonggibon'
        model_name: 모델 이름 (lr, ridge, lasso, rf, gbr 등)

    Returns:
        학습 결과
    """
    if request.organization not in ['R&A', 'tonggibon']:
        raise HTTPException(
            status_code=400,
            detail="organization must be 'R&A' or 'tonggibon'"
        )

    try:
        result = company_wide_modeling_service.train_model(
            organization=request.organization,
            model_name=request.model_name
        )
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@router.get("/status")
async def get_status(
    organization: str = Query(..., description="Organization: 'R&A' or 'tonggibon'")
) -> Dict[str, Any]:
    """
    모델링 상태 확인

    Args:
        organization: 'R&A' or 'tonggibon'

    Returns:
        현재 모델링 상태
    """
    if organization not in ['R&A', 'tonggibon']:
        raise HTTPException(
            status_code=400,
            detail="organization must be 'R&A' or 'tonggibon'"
        )

    try:
        result = company_wide_modeling_service.get_status(organization)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.delete("/clear")
async def clear_models(
    organization: Optional[str] = Query(None, description="Organization to clear (optional)")
) -> Dict[str, Any]:
    """
    모델 및 실험 초기화

    Args:
        organization: 'R&A', 'tonggibon', or None (전체 초기화)

    Returns:
        초기화 결과
    """
    if organization and organization not in ['R&A', 'tonggibon']:
        raise HTTPException(
            status_code=400,
            detail="organization must be 'R&A' or 'tonggibon'"
        )

    try:
        result = company_wide_modeling_service.clear_models(organization)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")

@router.post("/load")
async def load_model(
    organization: str = Body(..., embed=True),
    filename: str = Body('latest', embed=True)
) -> Dict[str, Any]:
    """
    저장된 모델 로드

    Args:
        organization: 'R&A' or 'tonggibon'
        filename: 모델 파일명 (기본: 'latest')

    Returns:
        로드 결과
    """
    if organization not in ['R&A', 'tonggibon']:
        raise HTTPException(
            status_code=400,
            detail="organization must be 'R&A' or 'tonggibon'"
        )

    try:
        result = company_wide_modeling_service.load_model(organization, filename)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {str(e)}")
