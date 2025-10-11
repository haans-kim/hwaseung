"""
Memory monitoring middleware for tracking memory usage
"""
import psutil
import os
import gc
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)
import time

class MemoryMonitorMiddleware(BaseHTTPMiddleware):
    """Middleware to monitor memory usage per request"""

    def __init__(self, app, memory_threshold_mb: int = 300):
        super().__init__(app)
        self.memory_threshold_mb = memory_threshold_mb
        self.process = psutil.Process(os.getpid())

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        memory_info = self.process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
            'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        }

    async def dispatch(self, request: Request, call_next):
        # Memory usage before request
        mem_before = self.get_memory_usage()
        start_time = time.time()

        logger.info(f"🔍 [{request.method}] {request.url.path}")
        logger.info(f"   Memory before: RSS={mem_before['rss_mb']:.1f}MB, VMS={mem_before['vms_mb']:.1f}MB")

        # Process request
        response = await call_next(request)

        # Memory usage after request
        mem_after = self.get_memory_usage()
        duration = time.time() - start_time

        # Calculate memory delta
        rss_delta = mem_after['rss_mb'] - mem_before['rss_mb']
        vms_delta = mem_after['vms_mb'] - mem_before['vms_mb']

        logger.info(f"   Memory after:  RSS={mem_after['rss_mb']:.1f}MB, VMS={mem_after['vms_mb']:.1f}MB")
        logger.info(f"   Memory delta:  RSS={rss_delta:+.1f}MB, VMS={vms_delta:+.1f}MB")
        logger.info(f"   Duration: {duration:.2f}s")

        # Warning if memory usage is high
        if mem_after['rss_mb'] > self.memory_threshold_mb:
            logger.warning(f"⚠️  HIGH MEMORY USAGE: {mem_after['rss_mb']:.1f}MB (threshold: {self.memory_threshold_mb}MB)")
            logger.warning(f"   Triggering garbage collection...")
            gc.collect()
            mem_after_gc = self.get_memory_usage()
            logger.warning(f"   Memory after GC: RSS={mem_after_gc['rss_mb']:.1f}MB")

        # Add memory info to response headers
        response.headers["X-Memory-RSS-MB"] = f"{mem_after['rss_mb']:.1f}"
        response.headers["X-Memory-Delta-MB"] = f"{rss_delta:+.1f}"

        return response


def log_memory_stats():
    """Log detailed memory statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()

    logger.info("=" * 60)
    logger.info("📊 MEMORY STATISTICS")
    logger.info(f"   RSS (Resident Set Size): {memory_info.rss / 1024 / 1024:.1f} MB")
    logger.info(f"   VMS (Virtual Memory Size): {memory_info.vms / 1024 / 1024:.1f} MB")

    # Only log these if available (not on all platforms)
    if hasattr(memory_info, 'shared'):
        logger.info(f"   Shared: {memory_info.shared / 1024 / 1024:.1f} MB")
    if hasattr(memory_info, 'text'):
        logger.info(f"   Text: {memory_info.text / 1024 / 1024:.1f} MB")
    if hasattr(memory_info, 'data'):
        logger.info(f"   Data: {memory_info.data / 1024 / 1024:.1f} MB")

    # System memory
    system_memory = psutil.virtual_memory()
    logger.info(f"   System Total: {system_memory.total / 1024 / 1024 / 1024:.1f} GB")
    logger.info(f"   System Available: {system_memory.available / 1024 / 1024 / 1024:.1f} GB")
    logger.info(f"   System Used: {system_memory.percent}%")
    logger.info("=" * 60)
