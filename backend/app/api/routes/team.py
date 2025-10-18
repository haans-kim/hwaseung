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

    자동으로 다음을 생성:
    - team_metrics: 팀별 월별 업무지표 값
    - team_feature_definitions: F1, F2 등을 실제 feature 이름으로 매핑
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

        # 엑셀 파일 검증
        validation_result = team_service.validate_excel_file(file_path)

        if not validation_result.get('valid'):
            raise HTTPException(status_code=400, detail=validation_result.get('error', 'Validation failed'))

        # DB 저장
        save_result = team_service.save_to_database(
            validation_result['df_master'],
            validation_result['df_matching'],
            validation_result['feature_columns']
        )

        if not save_result.get('success'):
            raise HTTPException(status_code=500, detail=save_result.get('error', 'Save failed'))

        # 회귀모델 자동 훈련
        train_result = team_service.train_regression_models()

        # 예측값 자동 계산
        prediction_result = team_service.calculate_predictions_from_features()

        return {
            "message": "Team data uploaded and processed successfully",
            "filename": file.filename,
            "validation": {
                "team_count": save_result.get('team_count', 0),
                "feature_count": save_result.get('feature_def_count', 0),
                "row_count": save_result.get('saved_count', 0),
            },
            "saved": {
                "count": save_result.get('saved_count', 0),
                "feature_definitions": save_result.get('feature_def_count', 0),
            },
            "models_trained": train_result.get('models_created', 0) if train_result.get('success') else 0,
            "predictions_created": prediction_result.get('predictions_saved', 0) if prediction_result.get('success') else 0,
            "errors": save_result.get('errors', [])
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/team/analysis-ready")
async def get_analysis_ready_teams() -> Dict[str, Any]:
    """분석가능팀 목록 조회 (회귀 모델이 있는 팀)"""
    try:
        teams = team_service.get_analysis_ready_teams()

        return {
            "message": "Analysis-ready teams retrieved successfully",
            "count": len(teams),
            "data": {
                "teams": teams
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis-ready teams: {str(e)}")

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

@router.get("/team/{team_name}/simulation-data")
async def get_team_simulation_data(team_name: str) -> Dict[str, Any]:
    """
    특정 팀의 시뮬레이션에 필요한 모든 데이터 조회
    - Regression models & parameters (총, 책임, 선임, 사원)
    - Team metrics (평균값)
    - Current headcount (최신 데이터)
    - FTE data
    """
    try:
        simulation_data = team_service.get_team_simulation_data(team_name)

        if not simulation_data:
            raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found")

        return {
            "message": "Team simulation data retrieved successfully",
            "data": simulation_data
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving team simulation data: {str(e)}")
