-- 006: FTE 테이블을 프론트엔드 UI에 맞게 피벗 구조로 복원

-- 기존 simple FTE 테이블 삭제
DROP TABLE IF EXISTS fte;

-- 프론트엔드 UI에 맞는 피벗 구조로 재생성
CREATE TABLE fte (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    팀명 TEXT NOT NULL,
    회사 TEXT NOT NULL,

    -- 평균 FTE (직급별)
    FTE_전체 REAL,
    FTE_책임 REAL,
    FTE_선임 REAL,
    FTE_사원 REAL,

    -- 인원수 (직급별)
    인원수_전체 REAL,
    인원수_책임 REAL,
    인원수_선임 REAL,
    인원수_사원 REAL,

    -- 평균 FTE/인원수 (직급별)
    FTE_per_인원_전체 REAL,
    FTE_per_인원_책임 REAL,
    FTE_per_인원_선임 REAL,
    FTE_per_인원_사원 REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(회사, 팀명)
);
