-- Migration 003: Add unique constraint to fte table

-- Since SQLite doesn't support adding constraints to existing tables,
-- we need to recreate the table with the constraint

-- Create new table with unique constraint
CREATE TABLE fte_new (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    팀명 TEXT NOT NULL,
    기간 TEXT NOT NULL,
    FTE_전체 REAL,
    FTE_책임 REAL,
    FTE_선임 REAL,
    FTE_사원 REAL,
    인원수_전체 INTEGER,
    인원수_책임 INTEGER,
    인원수_선임 INTEGER,
    인원수_사원 INTEGER,
    FTE_per_인원_전체 REAL,
    FTE_per_인원_책임 REAL,
    FTE_per_인원_선임 REAL,
    FTE_per_인원_사원 REAL,
    created_at TEXT,
    updated_at TEXT,
    회사 TEXT NOT NULL,
    UNIQUE(회사, 팀명, 기간)
);

-- Copy existing data
INSERT INTO fte_new SELECT * FROM fte;

-- Drop old table
DROP TABLE fte;

-- Rename new table
ALTER TABLE fte_new RENAME TO fte;

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_fte_company ON fte(회사);
CREATE INDEX IF NOT EXISTS idx_fte_team ON fte(팀명);
CREATE INDEX IF NOT EXISTS idx_fte_period ON fte(기간);
