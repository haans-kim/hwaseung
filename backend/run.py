import uvicorn
import os
from app.main import app

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0" if os.getenv("ENVIRONMENT") == "production" else "localhost"
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT") != "production",
        log_level="info"
    )