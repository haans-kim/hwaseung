from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

router = APIRouter(
    tags=["organization"],
    responses={404: {"description": "Not found"}},
)

def get_db_connection():
    """데이터베이스 연결"""
    # 프로젝트 루트 폴더의 DB 파일 사용
    db_path = '/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db'

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    return sqlite3.connect(db_path)

@router.get("/headcount-analysis")
async def get_headcount_analysis():
    """조직별 적정인원 분석을 위한 데이터 조회"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 조직 데이터 조회
        cursor.execute("""
            SELECT 회사, 본부, 담당_사업단_센터, 실, 팀
            FROM organization
            WHERE 팀 IS NOT NULL AND 팀 != ''
        """)
        org_data = []
        for row in cursor.fetchall():
            org_data.append({
                '회사': row[0],
                '본부': row[1],
                '담당_사업단_센터': row[2],
                '실': row[3],
                '팀': row[4]
            })

        # FTE 데이터 조회
        cursor.execute("""
            SELECT
                팀명,
                FTE_전체, FTE_책임, FTE_선임, FTE_사원,
                인원수_전체, 인원수_책임, 인원수_선임, 인원수_사원,
                FTE_per_인원_전체, FTE_per_인원_책임, FTE_per_인원_선임, FTE_per_인원_사원
            FROM fte
        """)

        fte_data = []
        for row in cursor.fetchall():
            fte_data.append({
                '팀명': row[0],
                'FTE_전체': float(row[1]),
                'FTE_책임': float(row[2]),
                'FTE_선임': float(row[3]),
                'FTE_사원': float(row[4]),
                '인원수_전체': int(row[5]),
                '인원수_책임': int(row[6]),
                '인원수_선임': int(row[7]),
                '인원수_사원': int(row[8]),
                'FTE_per_인원_전체': float(row[9]),
                'FTE_per_인원_책임': float(row[10]),
                'FTE_per_인원_선임': float(row[11]),
                'FTE_per_인원_사원': float(row[12])
            })

        conn.close()

        return {
            "organization": org_data,
            "fte": fte_data,
            "timestamp": datetime.now().isoformat()
        }

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.post("/export-recommendations")
async def export_recommendations(data: Dict[str, Any]):
    """적정인원 추천 결과를 엑셀로 내보내기"""
    try:
        recommendations = data.get('recommendations', [])
        target_ratio = data.get('targetRatio', 1.2)

        # DataFrame 생성
        df = pd.DataFrame(recommendations)

        # 컬럼명 한글로 변경
        df = df.rename(columns={
            '팀명': '팀명',
            '본부': '본부',
            '현재_인원': '현재 인원',
            '현재_FTE': '현재 FTE',
            'FTE_per_인원': 'FTE/인원',
            '목표_FTE_per_인원': '목표 FTE/인원',
            '적정_인원': '적정 인원',
            '인원_차이': '인원 차이',
            '조정_비율': '조정 비율(%)'
        })

        # 엑셀 파일 생성
        filename = f"organization_headcount_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        filepath = os.path.join('/tmp', filename)

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # 메인 시트
            df.to_excel(writer, sheet_name='적정인원 분석', index=False)

            # 요약 시트
            summary_data = {
                '구분': ['전체 현재 인원', '전체 적정 인원', '인원 차이', '평균 FTE/인원', '목표 FTE/인원'],
                '값': [
                    df['현재 인원'].sum(),
                    df['적정 인원'].sum(),
                    df['적정 인원'].sum() - df['현재 인원'].sum(),
                    df['FTE/인원'].mean(),
                    target_ratio
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='요약', index=False)

            # 본부별 통계
            dept_stats = df.groupby('본부').agg({
                '현재 인원': 'sum',
                '적정 인원': 'sum',
                '현재 FTE': 'sum',
                'FTE/인원': 'mean'
            }).round(2)
            dept_stats['인원 차이'] = dept_stats['적정 인원'] - dept_stats['현재 인원']
            dept_stats.to_excel(writer, sheet_name='본부별 통계')

        # 파일 읽기 및 반환
        with open(filepath, 'rb') as f:
            file_data = f.read()

        # 임시 파일 삭제
        os.remove(filepath)

        return {
            "file": file_data,
            "filename": filename,
            "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export error: {str(e)}")

@router.get("/departments")
async def get_departments():
    """전체 부서 목록 조회"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT 본부
            FROM organization
            WHERE 본부 IS NOT NULL AND 본부 != ''
            ORDER BY 본부
        """)

        departments = [row[0] for row in cursor.fetchall()]
        conn.close()

        return {
            "departments": departments,
            "count": len(departments)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@router.get("/teams/{department}")
async def get_teams_by_department(department: str):
    """특정 부서의 팀 목록 조회"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT 팀
            FROM organization
            WHERE 본부 = ? AND 팀 IS NOT NULL AND 팀 != ''
            ORDER BY 팀
        """, (department,))

        teams = [row[0] for row in cursor.fetchall()]
        conn.close()

        if not teams:
            raise HTTPException(status_code=404, detail=f"No teams found for department: {department}")

        return {
            "department": department,
            "teams": teams,
            "count": len(teams)
        }

    except sqlite3.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@router.get("/fte-summary")
async def get_fte_summary():
    """FTE 요약 통계"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # 전체 통계
        cursor.execute("""
            SELECT
                COUNT(*) as team_count,
                SUM(인원수_전체) as total_headcount,
                SUM(FTE_전체) as total_fte,
                AVG(FTE_per_인원_전체) as avg_fte_per_person
            FROM fte
        """)

        overall_stats = cursor.fetchone()

        # 직급별 통계
        cursor.execute("""
            SELECT
                SUM(인원수_책임) as headcount_책임,
                SUM(인원수_선임) as headcount_선임,
                SUM(인원수_사원) as headcount_사원,
                SUM(FTE_책임) as fte_책임,
                SUM(FTE_선임) as fte_선임,
                SUM(FTE_사원) as fte_사원
            FROM fte
        """)

        position_stats = cursor.fetchone()

        # FTE 상위 10개 팀
        cursor.execute("""
            SELECT 팀명, FTE_전체, 인원수_전체, FTE_per_인원_전체
            FROM fte
            ORDER BY FTE_전체 DESC
            LIMIT 10
        """)

        top_teams = []
        for row in cursor.fetchall():
            top_teams.append({
                '팀명': row[0],
                'FTE': row[1],
                '인원수': row[2],
                'FTE_per_인원': row[3]
            })

        conn.close()

        return {
            "overall": {
                "팀수": overall_stats[0],
                "총인원": overall_stats[1],
                "총FTE": float(overall_stats[2]),
                "평균_FTE_per_인원": float(overall_stats[3])
            },
            "by_position": {
                "책임": {
                    "인원수": position_stats[0],
                    "FTE": float(position_stats[3])
                },
                "선임": {
                    "인원수": position_stats[1],
                    "FTE": float(position_stats[4])
                },
                "사원": {
                    "인원수": position_stats[2],
                    "FTE": float(position_stats[5])
                }
            },
            "top_teams": top_teams
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")