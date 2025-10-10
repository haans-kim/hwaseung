"""
FTE API Routes
FTE 분석 데이터 API (평균FTE)
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import os
from app.services.fte_service import fte_service
from app.core.config import settings

router = APIRouter()

@router.post("/fte/upload")
async def upload_fte_data(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    FTE 분석 Excel 파일 업로드
    - "평균FTE" 시트만 사용
    - 회사, 팀명, 기간, FTE 관련 컬럼들 포함
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
        validation_result = fte_service.validate_excel_file(file_path)

        if not validation_result['valid']:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=validation_result['error'])

        # 데이터베이스에 저장
        save_result = fte_service.save_to_database(validation_result['dataframe'])

        if not save_result['success']:
            raise HTTPException(status_code=500, detail=save_result.get('error', 'Save failed'))

        return {
            "message": "FTE data uploaded successfully",
            "filename": file.filename,
            "validation": {
                "sheet_name": validation_result['sheet_name'],
                "row_count": validation_result['row_count'],
                "companies": validation_result['companies'],
                "company_count": validation_result['company_count'],
                "team_count": validation_result['team_count'],
                "positions": validation_result['positions'],
                "position_count": validation_result['position_count']
            },
            "saved": {
                "count": save_result['saved_count'],
                "errors": save_result.get('errors')
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/fte/data")
async def get_fte_data(
    company: Optional[str] = Query(None, description="회사명 (선택)"),
    team: Optional[str] = Query(None, description="팀명 (선택)")
) -> Dict[str, Any]:
    """FTE 데이터 조회"""
    try:
        data = fte_service.get_fte_data(company, team)

        return {
            "message": "FTE data retrieved successfully",
            "count": len(data),
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving FTE data: {str(e)}")

@router.delete("/fte/data")
async def delete_fte_data(
    company: Optional[str] = Query(None, description="회사명 (선택)"),
    team: Optional[str] = Query(None, description="팀명 (선택)")
) -> Dict[str, Any]:
    """FTE 데이터 삭제"""
    if not company and not team:
        raise HTTPException(status_code=400, detail="회사 또는 팀명을 지정해야 합니다")

    try:
        result = fte_service.delete_fte_data(company, team)

        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Delete failed'))

        return {
            "message": "FTE data deleted successfully",
            "deleted_count": result['deleted_count']
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting FTE data: {str(e)}")

@router.get("/fte/status")
async def get_fte_status() -> Dict[str, Any]:
    """FTE 데이터 전체 상태 조회"""
    try:
        status = fte_service.get_status()

        return {
            "message": "FTE data status retrieved successfully",
            "status": status
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")
