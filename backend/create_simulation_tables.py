import sqlite3
import os

# 데이터베이스 경로
db_path = '/Users/hanskim/Projects/Hwaseung/hwaseung_RnD.db'

# 연결 생성
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1. 회귀분석 모델 정보 테이블
cursor.execute('''
CREATE TABLE IF NOT EXISTS regression_models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    org_name TEXT NOT NULL,
    model_type TEXT NOT NULL, -- '전사' or '팀별'
    r_squared REAL,
    adjusted_r_squared REAL,
    f_statistic REAL,
    p_value REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

# 2. 회귀분석 파라미터 테이블
cursor.execute('''
CREATE TABLE IF NOT EXISTS regression_parameters (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id INTEGER NOT NULL,
    parameter_name TEXT NOT NULL,
    coefficient REAL,
    std_error REAL,
    t_statistic REAL,
    p_value REAL,
    FOREIGN KEY (model_id) REFERENCES regression_models(id)
)
''')

# 3. 팀별 월별 업무 지표 테이블
cursor.execute('''
CREATE TABLE IF NOT EXISTS team_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT NOT NULL,
    year INTEGER,
    month INTEGER,
    metric_category TEXT,
    metric_name TEXT,
    metric_value REAL
)
''')

# 4. 팀별 인력 현황 테이블
cursor.execute('''
CREATE TABLE IF NOT EXISTS team_headcount (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name TEXT NOT NULL,
    year INTEGER,
    month INTEGER,
    position TEXT, -- '책임', '선임', '사원'
    headcount INTEGER,
    flow_logins INTEGER
)
''')

# 5. 시뮬레이션 시나리오 저장 테이블
cursor.execute('''
CREATE TABLE IF NOT EXISTS simulation_scenarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scenario_name TEXT NOT NULL,
    org_name TEXT,
    parameters TEXT, -- JSON 형식으로 저장
    predicted_headcount INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
''')

conn.commit()
print("시뮬레이션 관련 테이블이 생성되었습니다.")

# 기존 테이블 확인
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\n현재 데이터베이스의 테이블 목록:")
for table in tables:
    print(f"  - {table[0]}")

conn.close()