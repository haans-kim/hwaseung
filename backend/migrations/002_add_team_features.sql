-- Migration 002: Add team features tables

-- Table for team feature mapping (which features apply to which teams)
CREATE TABLE IF NOT EXISTS team_feature_mapping (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization, feature_name)
);

-- Table for team feature values (actual data)
CREATE TABLE IF NOT EXISTS team_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization TEXT NOT NULL,
    year INTEGER NOT NULL,
    feature_values TEXT NOT NULL,  -- JSON string of feature values
    headcount INTEGER,  -- Target variable
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(organization, year)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_team_mapping_org ON team_feature_mapping(organization);
CREATE INDEX IF NOT EXISTS idx_team_mapping_active ON team_feature_mapping(is_active);
CREATE INDEX IF NOT EXISTS idx_team_features_org ON team_features(organization);
CREATE INDEX IF NOT EXISTS idx_team_features_year ON team_features(year);
