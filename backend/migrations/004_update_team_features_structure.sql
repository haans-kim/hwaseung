-- Migration 004: Update team_features table structure for actual data format

-- Drop old tables
DROP TABLE IF EXISTS team_feature_mapping;
DROP TABLE IF EXISTS team_features;

-- Create new team_features table with correct structure
CREATE TABLE team_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    company TEXT NOT NULL,          -- HQ (회사)
    team TEXT NOT NULL,              -- 팀
    year INTEGER NOT NULL,           -- 년
    month INTEGER NOT NULL,          -- 월
    position TEXT,                   -- 구분 (직급: 전체, 선임, 책임, 사원)
    feature_values TEXT NOT NULL,    -- JSON string of feature values (F1-F9)
    headcount INTEGER,               -- 인력규모 (target variable)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company, team, year, month, position)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_team_features_company ON team_features(company);
CREATE INDEX IF NOT EXISTS idx_team_features_team ON team_features(team);
CREATE INDEX IF NOT EXISTS idx_team_features_year ON team_features(year);
CREATE INDEX IF NOT EXISTS idx_team_features_month ON team_features(month);
CREATE INDEX IF NOT EXISTS idx_team_features_position ON team_features(position);
