#!/usr/bin/env python3
"""
4개 팀의 기본 예측값을 계산하여 team_predictions 테이블에 저장하는 스크립트
"""

import sqlite3
import sys
from pathlib import Path

# DB 경로 설정
DB_PATH = Path(__file__).parent.parent.parent / "hwaseung_RnD.db"

def calculate_prediction(model_id, metrics, db_conn):
    """회귀 모델로 예측값 계산"""
    cursor = db_conn.cursor()

    # 회귀 계수 조회
    cursor.execute("""
        SELECT parameter_name, coefficient
        FROM regression_parameters
        WHERE model_id = ?
    """, (model_id,))

    parameters = cursor.fetchall()

    # 예측값 계산
    prediction = 0
    for param_name, coefficient in parameters:
        if param_name == 'intercept':
            prediction += coefficient
        elif param_name in metrics:
            prediction += coefficient * metrics[param_name]

    return max(0, round(prediction))

def main():
    print("=" * 80)
    print("팀별 예측값 계산 및 저장 스크립트")
    print("=" * 80)

    # DB 연결
    conn = sqlite3.connect(str(DB_PATH))
    cursor = conn.cursor()

    # team_predictions 테이블 생성 (존재하지 않으면)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS team_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            team_name TEXT NOT NULL,
            position TEXT NOT NULL,
            current_headcount INTEGER NOT NULL,
            predicted_headcount INTEGER NOT NULL,
            change INTEGER NOT NULL,
            change_percent REAL NOT NULL,
            category TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(team_name, position)
        )
    """)

    # 기존 데이터 삭제
    cursor.execute("DELETE FROM team_predictions")
    print("✅ team_predictions 테이블 초기화 완료")

    # 회귀 모델이 있는 팀 조회
    cursor.execute("""
        SELECT DISTINCT org_name
        FROM regression_models
        ORDER BY org_name
    """)
    teams = [row[0] for row in cursor.fetchall()]

    print(f"\n📊 분석 대상 팀: {len(teams)}개")
    for team in teams:
        print(f"  - {team}")

    total_results = []

    for team_name in teams:
        print(f"\n{'='*60}")
        print(f"팀: {team_name}")
        print(f"{'='*60}")

        # 팀 메트릭 평균값 조회
        cursor.execute("""
            SELECT metric_name, AVG(metric_value) as avg_value
            FROM team_metrics
            WHERE team_name = ?
            GROUP BY metric_name
        """, (team_name,))

        metrics = {row[0]: row[1] for row in cursor.fetchall()}
        print(f"메트릭 개수: {len(metrics)}개")

        # 모델 타입별 예측 (총, 책임, 선임, 사원)
        model_types = ['총', '책임', '선임', '사원']
        team_results = {}

        for model_type in model_types:
            # 모델 조회
            cursor.execute("""
                SELECT id FROM regression_models
                WHERE org_name = ? AND model_type = ?
                LIMIT 1
            """, (team_name, model_type))

            result = cursor.fetchone()
            if not result:
                print(f"  ⚠️  {model_type}: 모델 없음")
                continue

            model_id = result[0]

            # 현재 인원 조회
            position_map = {'총': '총합', '책임': '책임', '선임': '선임', '사원': '사원'}
            cursor.execute("""
                SELECT headcount FROM team_headcount
                WHERE team_name = ? AND year = 25 AND month = 8 AND position = ?
                LIMIT 1
            """, (team_name, position_map[model_type]))

            current = cursor.fetchone()
            current_headcount = current[0] if current else 0

            # 예측값 계산
            predicted_headcount = calculate_prediction(model_id, metrics, conn)

            # 변화량 및 변화율 계산
            change = predicted_headcount - current_headcount
            change_percent = (change / current_headcount * 100) if current_headcount > 0 else 0

            # 분류 결정
            if change > 0:
                category = '충원필요'
            elif change < 0:
                category = '감원검토'
            else:
                category = '적정'

            team_results[model_type] = {
                'current': current_headcount,
                'predicted': predicted_headcount,
                'change': change,
                'change_percent': change_percent,
                'category': category
            }

            print(f"  {model_type}: {current_headcount}명 → {predicted_headcount}명 ({change:+d}명, {change_percent:+.1f}%) [{category}]")

            # DB에 저장
            cursor.execute("""
                INSERT OR REPLACE INTO team_predictions
                (team_name, position, current_headcount, predicted_headcount, change, change_percent, category)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (team_name, model_type, current_headcount, predicted_headcount, change, change_percent, category))

        total_results.append({
            'team_name': team_name,
            'results': team_results
        })

    # 변경사항 커밋
    conn.commit()

    # 전체 요약 통계
    print(f"\n{'='*80}")
    print("📈 전체 요약")
    print(f"{'='*80}")

    # 총 분석 인원 (4개 팀의 예상 인력 합계)
    total_current = sum(
        result['results'].get('총', {}).get('current', 0)
        for result in total_results
    )

    total_predicted = sum(
        sum([
            result['results'].get('책임', {}).get('predicted', 0),
            result['results'].get('선임', {}).get('predicted', 0),
            result['results'].get('사원', {}).get('predicted', 0)
        ])
        for result in total_results
    )

    print(f"현재 총 인원: {total_current}명")
    print(f"예상 총 인원: {total_predicted}명")
    print(f"총 변화: {total_predicted - total_current:+d}명")

    # 직급별 변동
    for position in ['책임', '선임', '사원']:
        total_change = sum(
            result['results'].get(position, {}).get('change', 0)
            for result in total_results
        )
        print(f"{position} 변동: {total_change:+d}명")

    # 분류별 통계
    print(f"\n📊 분류별 통계:")
    for category in ['충원필요', '적정', '감원검토']:
        cursor.execute("""
            SELECT COUNT(*) FROM team_predictions
            WHERE category = ?
        """, (category,))
        count = cursor.fetchone()[0]
        print(f"  {category}: {count}건")

    conn.close()
    print(f"\n✅ 예측값 계산 및 저장 완료!")
    print(f"DB 파일: {DB_PATH}")

    return 0

if __name__ == "__main__":
    sys.exit(main())
