"""
Team Features API Routes
조직별 팀 단위 인력 산정 데이터 API
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import os
from app.services.team_service import team_service
from app.core.config import settings

router = APIRouter()

@router.post("/team/upload")
async def upload_team_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    조직인력산정용 Excel 파일 업로드
    - Sheet 1: feature matching (조직 정보)
    - Sheet 2: master (HQ, 팀, 년, 월, 구분, F1-F9, 인력규모)
    """
    # 파일 검증
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Only Excel files (.xlsx, .xls) are allowed")

    try:
        # 파일 저장
        upload_dir = settings.UPLOAD_DIR
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)

        contents = await file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)

        # 파일 검증
        validation_result = team_service.validate_excel_file(file_path)

        if not validation_result['valid']:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=validation_result['error'])

        # 데이터베이스에 저장
        save_result = team_service.save_to_database(
            validation_result['df_master'],
            validation_result['df_matching'],
            validation_result['feature_columns']
        )

        if not save_result['success']:
            raise HTTPException(status_code=500, detail=save_result.get('error', 'Save failed'))

        # 🔧 NEW: 회귀 모델 자동 학습
        model_result = team_service.train_regression_models()

        # 🔧 NEW: 예측값 자동 계산
        prediction_result = team_service.calculate_predictions_from_features()

        return {
            "message": "Team data uploaded successfully",
            "filename": file.filename,
            "validation": {
                "companies": validation_result['companies'],
                "teams": validation_result['teams'],
                "years": [int(y) for y in validation_result['years']],
                "months": [int(m) for m in validation_result['months']],
                "positions": validation_result['positions'],
                "row_count": validation_result['row_count'],
                "team_count": validation_result['team_count'],
                "feature_count": validation_result['feature_count']
            },
            "saved": {
                "count": save_result['saved_count'],
                "errors": save_result.get('errors')
            },
            "models": {
                "trained": model_result.get('models_trained', 0),
                "teams_processed": model_result.get('teams_processed', 0),
                "errors": model_result.get('errors')
            },
            "predictions": {
                "saved": prediction_result.get('predictions_saved', 0),
                "deleted": prediction_result.get('deleted_count', 0),
                "errors": prediction_result.get('errors')
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/team/features")
async def get_team_features(
    company: Optional[str] = Query(None, description="회사명 (선택)")
) -> Dict[str, Any]:
    """Team Features 조회"""
    try:
        features = team_service.get_team_features(company)

        return {
            "message": "Team features retrieved successfully",
            "count": len(features),
            "data": features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving team features: {str(e)}")

@router.get("/team/status")
async def get_team_status() -> Dict[str, Any]:
    """Team 데이터 전체 상태 조회"""
    try:
        status = team_service.get_status()

        return {
            "message": "Team data status retrieved successfully",
            "status": status
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")

@router.get("/team/predictions")
async def get_team_predictions() -> Dict[str, Any]:
    """
    팀별 직급별 예측 인력 데이터 조회
    요약 페이지에서 사용
    """
    try:
        predictions = team_service.get_team_predictions()

        return {
            "message": "Team predictions retrieved successfully",
            "count": len(predictions),
            "data": predictions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving team predictions: {str(e)}")
