"""
Team Features API Routes
조직별 Feature 매핑 및 팀 단위 인력 산정 데이터 API
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
    - Sheet 1: Feature Mapping (조직, Feature 이름, 설명, 사용여부)
    - Sheet 2: Master Data (연도, 조직, [features], 인원)
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

        # Feature Mapping 저장
        mapping_result = team_service.save_feature_mapping(validation_result['mapping_sheet'])
        if not mapping_result['success']:
            raise HTTPException(status_code=500, detail=mapping_result.get('error', 'Mapping save failed'))

        # Team Features 저장
        features_result = team_service.save_team_features(
            validation_result['master_sheet'],
            validation_result['active_features']
        )
        if not features_result['success']:
            raise HTTPException(status_code=500, detail=features_result.get('error', 'Features save failed'))

        return {
            "message": "Team data uploaded successfully",
            "filename": file.filename,
            "validation": {
                "organizations": validation_result['organizations'],
                "feature_count": validation_result['feature_count'],
                "data_rows": validation_result['data_rows']
            },
            "saved": {
                "mapping_count": mapping_result['saved_count'],
                "features_count": features_result['saved_count']
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/team/mapping")
async def get_feature_mapping(
    organization: Optional[str] = Query(None, description="조직명 (선택)"),
    active_only: bool = Query(True, description="활성 Feature만 조회")
) -> Dict[str, Any]:
    """Feature Mapping 조회"""
    try:
        mappings = team_service.get_feature_mapping(organization, active_only)

        return {
            "message": "Feature mapping retrieved successfully",
            "count": len(mappings),
            "data": mappings
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feature mapping: {str(e)}")

@router.get("/team/features")
async def get_team_features(
    organization: Optional[str] = Query(None, description="조직명 (선택)")
) -> Dict[str, Any]:
    """Team Features 조회"""
    try:
        features = team_service.get_team_features(organization)

        return {
            "message": "Team features retrieved successfully",
            "count": len(features),
            "data": features
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving team features: {str(e)}")

@router.delete("/team/features")
async def delete_team_features(
    organization: str = Query(..., description="삭제할 조직명")
) -> Dict[str, Any]:
    """특정 조직의 Team 데이터 삭제"""
    try:
        result = team_service.delete_team_data(organization)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Delete failed'))

        return {
            "message": f"Team data for '{organization}' deleted successfully",
            "deleted": {
                "mapping_count": result['mapping_deleted'],
                "features_count": result['features_deleted']
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting team data: {str(e)}")

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
