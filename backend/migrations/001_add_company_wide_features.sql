-- Migration: Add company_wide_features table for R*A and tonggibon
-- Created: 2025-10-10
-- Purpose: Store company-wide headcount prediction features separated by organization

CREATE TABLE IF NOT EXISTS company_wide_features (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    organization TEXT NOT NULL,  -- 'R*A' or 'tonggibon'
    year INTEGER NOT NULL,

    -- External Environment Indicators (10 features)
    ev_growth_gl REAL,           -- 글로벌 EV시장성장률
    v_growth_gl REAL,            -- 글로벌 자동차 시장성장률
    v_export_kr REAL,            -- 국내 자동차 수출액 증가율
    vp_export_kr REAL,           -- 국내 자동차부품 수출액 증가율
    gdp_growth_kr REAL,          -- GDP성장률
    cpi_kr REAL,                 -- 소비자물가상승률
    exchange_rate_change_krw REAL, -- 환율변화율_원화기준
    scm_index_gl REAL,           -- 글로벌물류비지수
    oil_gl REAL,                 -- 국제유가

    -- Internal Performance Indicators (5 features)
    revenue REAL,                -- 매출액 증감률
    profit REAL,                 -- 영업이익 증감률
    operating_rate REAL,         -- 가동률 증감률
    operating_date REAL,         -- 가동일수 증감률
    labor_cost REAL,             -- 인건비 증감률

    -- Target Variable
    headcount INTEGER,           -- 정원 (target)

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(organization, year)
);

-- Create indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_company_org ON company_wide_features(organization);
CREATE INDEX IF NOT EXISTS idx_company_year ON company_wide_features(year);
CREATE INDEX IF NOT EXISTS idx_company_org_year ON company_wide_features(organization, year);

-- Insert sample data for testing (optional)
-- R*A sample data
-- INSERT INTO company_wide_features (organization, year, ev_growth_gl, v_growth_gl, headcount)
-- VALUES ('R*A', 2021, 1.04, 0.026, 320);

-- tonggibon sample data
-- INSERT INTO company_wide_features (organization, year, ev_growth_gl, v_growth_gl, headcount)
-- VALUES ('tonggibon', 2021, 1.04, 0.026, 250);
