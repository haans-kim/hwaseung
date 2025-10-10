"""
Organization Chart API Routes
조직도 데이터 API (계층 구조 관리)
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import os
from app.services.organization_service import organization_service
from app.core.config import settings

router = APIRouter()

@router.post("/organization-chart/upload")
async def upload_organization_chart(
    file: UploadFile = File(...),
    replace_all: bool = Query(True, description="기존 데이터 전체 교체 여부")
) -> Dict[str, Any]:
    """
    조직도 Excel 파일 업로드
    - 회사, 본부, 담당/사업단/센터, 실, 팀 계층 구조
    - replace_all=True: 기존 데이터 삭제 후 새 데이터 저장
    - replace_all=False: UPSERT 방식
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
        validation_result = organization_service.validate_excel_file(file_path)

        if not validation_result['valid']:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail=validation_result['error'])

        # 데이터베이스에 저장
        save_result = organization_service.save_to_database(
            validation_result['dataframe'],
            replace_all=replace_all
        )

        if not save_result['success']:
            raise HTTPException(status_code=500, detail=save_result.get('error', 'Save failed'))

        return {
            "message": "Organization chart uploaded successfully",
            "filename": file.filename,
            "mode": "replace_all" if replace_all else "upsert",
            "validation": {
                "row_count": validation_result['row_count'],
                "companies": validation_result['companies'],
                "company_count": validation_result['company_count'],
                "division_count": validation_result['division_count'],
                "department_count": validation_result['department_count'],
                "office_count": validation_result['office_count'],
                "team_count": validation_result['team_count']
            },
            "saved": {
                "deleted_count": save_result.get('deleted_count', 0),
                "saved_count": save_result['saved_count'],
                "errors": save_result.get('errors')
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@router.get("/organization-chart/data")
async def get_organization_chart_data(
    company: Optional[str] = Query(None, description="회사명 (선택)")
) -> Dict[str, Any]:
    """조직도 데이터 조회 (평면 구조)"""
    try:
        data = organization_service.get_organization_data(company)

        return {
            "message": "Organization chart data retrieved successfully",
            "count": len(data),
            "data": data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving organization data: {str(e)}")

@router.get("/organization-chart/tree")
async def get_organization_tree(
    company: Optional[str] = Query(None, description="회사명 (선택)")
) -> Dict[str, Any]:
    """조직도 계층 구조 조회 (트리 구조)"""
    try:
        tree = organization_service.get_hierarchy_tree(company)

        return {
            "message": "Organization hierarchy tree retrieved successfully",
            "tree": tree
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving organization tree: {str(e)}")

@router.get("/organization-chart/status")
async def get_organization_status() -> Dict[str, Any]:
    """조직도 데이터 전체 상태 조회"""
    try:
        status = organization_service.get_status()

        return {
            "message": "Organization chart status retrieved successfully",
            "status": status
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving status: {str(e)}")
