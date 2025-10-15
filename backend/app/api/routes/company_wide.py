"""
전사 인력 산정 API 엔드포인트 (R&A, tonggibon)
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
from pydantic import BaseModel
import os
import tempfile
from pathlib import Path

from app.services.company_wide_service import company_wide_service

router = APIRouter(
    prefix="/company-wide",
    tags=["company-wide"],
    responses={404: {"description": "Not found"}},
)

class UploadRequest(BaseModel):
    organization: str  # 'R&A' or 'tonggibon'

@router.post("/upload")
async def upload_company_wide_data(
    file: UploadFile = File(...),
    organization: str = Query(..., description="Organization: 'R&A' or 'tonggibon'")
) -> Dict[str, Any]:
    """
    전사 인력 산정용 데이터 업로드

    Args:
        file: Excel 파일 (1-1. R&A.xlsx 또는 1-2. 통기본.xlsx)
        organization: 'R&A' or 'tonggibon'

    Returns:
        업로드 결과
    """
    # 조직 검증
    if organization not in ['R&A', 'tonggibon']:
        raise HTTPException(
            status_code=400,
            detail="organization must be 'R&A' or 'tonggibon'"
        )

    # 파일 형식 검증
    if not file.filename or not file.filename.endswith('.xlsx'):
        raise HTTPException(
            status_code=400,
            detail="File must be an Excel file (.xlsx)"
        )

    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name

        # 파일 검증
        validation_result = company_wide_service.validate_excel_file(tmp_path, organization)

        if not validation_result['is_valid']:
            os.unlink(tmp_path)
            raise HTTPException(
                status_code=400,
                detail=f"Validation failed: {validation_result.get('error', 'Unknown error')}"
            )

        # 데이터베이스에 저장
        save_result = company_wide_service.save_to_database(
            validation_result['data'],
            organization
        )

        # 임시 파일 삭제
        os.unlink(tmp_path)

        if not save_result['success']:
            raise HTTPException(
                status_code=500,
                detail=f"Database save failed: {save_result.get('error', 'Unknown error')}"
            )

        return {
            "message": f"Company-wide data uploaded successfully for {organization}",
            "organization": organization,
            "filename": file.filename,
            "validation": {
                "rows": validation_result['rows'],
                "years": validation_result['years'],
                "warnings": validation_result.get('warnings', [])
            },
            "save_result": {
                "saved_count": save_result['saved_count'],
                "updated_count": save_result['updated_count'],
                "total": save_result['total']
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        # 임시 파일 정리
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/features")
async def get_company_wide_features(
    organization: str = Query(None, description="Filter by organization: 'R&A' or 'tonggibon'")
) -> Dict[str, Any]:
    """
    전사 인력 Feature 데이터 조회

    Args:
        organization: 조직 필터 (선택사항)

    Returns:
        Feature 데이터
    """
    try:
        if organization:
            if organization not in ['R&A', 'tonggibon']:
                raise HTTPException(
                    status_code=400,
                    detail="organization must be 'R&A' or 'tonggibon'"
                )

            data = company_wide_service.get_features_by_organization(organization)
            return {
                "organization": organization,
                "count": len(data),
                "data": data
            }
        else:
            # 모든 조직 데이터
            all_data = company_wide_service.get_all_features()
            return {
                "R&A": {
                    "count": len(all_data['R&A']),
                    "data": all_data['R&A']
                },
                "tonggibon": {
                    "count": len(all_data['tonggibon']),
                    "data": all_data['tonggibon']
                }
            }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@router.delete("/features")
async def delete_company_wide_features(
    organization: str = Query(..., description="Organization to delete: 'R&A' or 'tonggibon'")
) -> Dict[str, Any]:
    """
    특정 조직의 전사 인력 데이터 삭제

    Args:
        organization: 'R&A' or 'tonggibon'

    Returns:
        삭제 결과
    """
    if organization not in ['R&A', 'tonggibon']:
        raise HTTPException(
            status_code=400,
            detail="organization must be 'R&A' or 'tonggibon'"
        )

    try:
        import sqlite3
        import os
        # 🔧 FIX: Docker 환경을 위해 환경 변수 사용
        db_path = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), '../../../hwaseung_RnD.db'))
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
            DELETE FROM company_wide_features
            WHERE organization = ?
        """, (organization,))

        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()

        return {
            "message": f"Deleted {deleted_count} records for {organization}",
            "organization": organization,
            "deleted_count": deleted_count
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Delete failed: {str(e)}")

@router.get("/status")
async def get_upload_status() -> Dict[str, Any]:
    """
    전사 인력 데이터 업로드 상태 확인

    Returns:
        각 조직별 데이터 존재 여부 및 통계
    """
    try:
        all_data = company_wide_service.get_all_features()

        return {
            "R&A": {
                "has_data": len(all_data['R&A']) > 0,
                "row_count": len(all_data['R&A']),
                "years": [row['year'] for row in all_data['R&A']] if all_data['R&A'] else []
            },
            "tonggibon": {
                "has_data": len(all_data['tonggibon']) > 0,
                "row_count": len(all_data['tonggibon']),
                "years": [row['year'] for row in all_data['tonggibon']] if all_data['tonggibon'] else []
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
