from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from app.api.routes import data, modeling, analysis, dashboard, organization, company_wide, team, fte, organization_chart, company_wide_modeling, company_wide_dashboard
from app.core.config import settings
# from app.middleware.memory_monitor import MemoryMonitorMiddleware, log_memory_stats
import os
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SambioWage API",
    description="머신러닝 기반 임금인상률 예측 대시보드 API",
    version="1.0.0"
)

# Add memory monitoring middleware
# app.add_middleware(MemoryMonitorMiddleware, memory_threshold_mb=300)

# CORS 설정 - 환경에 따라 다르게 적용
if os.getenv("ENVIRONMENT") == "production":
    # Production CORS 설정
    origins = os.getenv("CORS_ORIGINS", "").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins[0] else ["https://sambiowage.vercel.app"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
else:
    # Development CORS 설정 (모든 origin 허용)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 라우터 등록
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(modeling.router, prefix="/api/modeling", tags=["modeling"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(organization.router, prefix="/api/organization", tags=["organization"])
app.include_router(company_wide.router, prefix="/api", tags=["company-wide"])
app.include_router(team.router, prefix="/api", tags=["team"])
app.include_router(fte.router, prefix="/api", tags=["fte"])
app.include_router(organization_chart.router, prefix="/api", tags=["organization-chart"])
app.include_router(company_wide_modeling.router, prefix="/api", tags=["company-wide-modeling"])
app.include_router(company_wide_dashboard.router, prefix="/api", tags=["company-wide-dashboard"])

@app.on_event("startup")
async def startup_event():
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info("🚀 SambioWage API starting up...")
    logger.info(f"   Initial Memory: {memory_info.rss / 1024 / 1024:.1f} MB")

@app.on_event("shutdown")
async def shutdown_event():
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    logger.info("👋 SambioWage API shutting down...")
    logger.info(f"   Final Memory: {memory_info.rss / 1024 / 1024:.1f} MB")

@app.get("/")
async def root():
    return {"message": "SambioWage API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/memory")
async def memory_status():
    """Get current memory status"""
    import psutil
    import gc

    # Trigger garbage collection
    gc.collect()

    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    system_memory = psutil.virtual_memory()

    return {
        "process": {
            "rss_mb": round(memory_info.rss / 1024 / 1024, 1),
            "vms_mb": round(memory_info.vms / 1024 / 1024, 1),
        },
        "system": {
            "total_gb": round(system_memory.total / 1024 / 1024 / 1024, 1),
            "available_gb": round(system_memory.available / 1024 / 1024 / 1024, 1),
            "used_percent": system_memory.percent
        },
        "warning": "Base memory ~230MB (PyCaret), 300MB limit may be too low"
    }

@app.post("/gc")
async def trigger_gc():
    """Manually trigger garbage collection"""
    import gc
    import psutil

    process = psutil.Process(os.getpid())
    before = process.memory_info().rss / 1024 / 1024

    collected = gc.collect()

    after = process.memory_info().rss / 1024 / 1024

    return {
        "collected_objects": collected,
        "memory_before_mb": round(before, 1),
        "memory_after_mb": round(after, 1),
        "freed_mb": round(before - after, 1)
    }