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

        # 조직도 데이터 저장
        save_result = organization_service.save_to_database(
            validation_result['dataframe'],
            replace_all=replace_all
        )

        if not save_result['success']:
            raise HTTPException(status_code=500, detail=save_result.get('error', 'Save failed'))

        # Feature 정의 저장
        feature_save_result = None
        if validation_result.get('feature_dataframe') is not None:
            feature_save_result = organization_service.save_feature_definitions(
                validation_result['feature_dataframe'],
                replace_all=replace_all
            )

        # Master 시트 데이터 처리 (team_metrics, team_headcount)
        master_result = organization_service.process_master_sheet(
            file_path,
            replace_all=replace_all
        )

        response_data = {
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

        # Feature 정의 저장 결과 추가
        if feature_save_result:
            response_data["feature_definitions"] = {
                "saved_count": feature_save_result.get('saved_count', 0),
                "deleted_count": feature_save_result.get('deleted_count', 0),
                "errors": feature_save_result.get('errors')
            }

        # Master 데이터 저장 결과 추가
        if master_result and master_result.get('success'):
            response_data["master_data"] = {
                "metrics_saved": master_result.get('metrics_saved', 0),
                "metrics_deleted": master_result.get('metrics_deleted', 0),
                "headcount_saved": master_result.get('headcount_saved', 0),
                "headcount_deleted": master_result.get('headcount_deleted', 0),
                "errors": master_result.get('errors')
            }

        return response_data

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

@router.get("/organization-chart/feature-definitions")
async def get_feature_definitions(
    team: Optional[str] = Query(None, description="팀명 (선택)"),
    company: Optional[str] = Query(None, description="회사명 (선택)")
) -> Dict[str, Any]:
    """Feature 정의 조회"""
    try:
        definitions = organization_service.get_feature_definitions(team=team, company=company)

        return {
            "message": "Feature definitions retrieved successfully",
            "count": len(definitions),
            "data": definitions
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving feature definitions: {str(e)}")

@router.get("/organization-chart/analysis-ready-teams")
async def get_analysis_ready_teams() -> Dict[str, Any]:
    """
    분석가능팀 목록 조회
    조건: Feature 정의가 있고 회귀모델이 있는 팀
    """
    try:
        teams = organization_service.get_analysis_ready_teams()

        return {
            "message": "Analysis-ready teams retrieved successfully",
            "count": len(teams),
            "teams": teams
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analysis-ready teams: {str(e)}")
