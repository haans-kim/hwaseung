-- Migration 005: Simplify FTE table structure to match template

-- Drop old fte table
DROP TABLE IF EXISTS fte;

-- Create simplified fte table matching template structure
CREATE TABLE fte (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,           -- 계열사 (Corp., R&A)
    team TEXT NOT NULL,              -- 부서 (팀명)
    position TEXT NOT NULL,          -- 사용자직위 (전체, 선임, 책임, 사원)
    avg_fte REAL,                    -- 평균FTE
    headcount REAL,                  -- 인원수
    avg_fte_per_person REAL,         -- 평균FTE/인원수
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company, team, position)
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_fte_company ON fte(company);
CREATE INDEX IF NOT EXISTS idx_fte_team ON fte(team);
CREATE INDEX IF NOT EXISTS idx_fte_position ON fte(position);
