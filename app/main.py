from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging
from pathlib import Path
import asyncio

from app.services.ml_service import ml_service

from fastapi.responses import RedirectResponse

from app.config import settings
from app.routers import test, home, report, algorithm, researchers, about, resources, privacy

# Set up logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with error handling for Vercel
try:
    if Path("static").exists():
        app.mount("/static", StaticFiles(directory="static"), name="static")
    if Path("ml_assets").exists():
        app.mount("/ml_assets", StaticFiles(directory="ml_assets"), name="ml_assets")
except Exception as e:
    logger.warning(f"Could not mount static files: {e}")

# Templates
templates = Jinja2Templates(directory="templates")

# Include routers
app.include_router(home.router, tags=["Home"])
app.include_router(test.router, prefix="/test", tags=["Testing"])
app.include_router(report.router, prefix="/report", tags=["Reports"])
app.include_router(algorithm.router, prefix="/algorithm", tags=["Algorithm"])
app.include_router(resources.router, prefix="/resources", tags=["Resources"])
app.include_router(researchers.router, prefix="/researchers", tags=["Researchers"])
app.include_router(about.router, prefix="/about", tags=["About"])
app.include_router(privacy.router, prefix="/privacy", tags=["Privacy"])


@app.get("/")
async def root():
    return RedirectResponse(url="/home")


@app.get("/health")
async def health_check():
    model_status = "unknown"
    try:
        model_info = ml_service.get_model_info()
        model_status = model_info.get('status', 'unknown')
    except Exception as e:
        logger.warning(f"Could not get model status: {e}")
        model_status = "error"

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": "production" if not settings.DEBUG else "development",
        "models_status": model_status
    }

@app.get("/model-status")
async def model_status():

    try:
        model_info = ml_service.get_model_info()
        return {
            "success": True,
            "model_info": model_info,
            "preloaded": ml_service.models_loaded
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "preloaded": False
        }


# Enhanced Error Handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    try:
        config_data = settings.get_full_config()
    except Exception:
        config_data = None

    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_code": 404,
            "error_message": "The page you're looking for doesn't exist or has been moved.",
            "config_data": config_data
        },
        status_code=404
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc: Exception):
    try:
        config_data = settings.get_full_config()
    except Exception:
        config_data = None

    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return templates.TemplateResponse(
        "error.html",
        {
            "request": request,
            "error_code": 500,
            "error_message": "Internal server error. Our team has been notified and is working to fix this issue.",
            "config_data": config_data
        },
        status_code=500
    )


async def preload_ml_models():
    try:
        logger.info("🔄 Starting ML model preloading...")

        # Start model loading in background
        success = ml_service.load_models()

        if success:
            logger.info("✅ ML models preloaded successfully!")
        else:
            logger.warning("⚠️ ML models failed to preload, will use fallback")

    except Exception as e:
        logger.error(f"❌ Error during model preloading: {e}")


# Startup event with model preloading
@app.on_event("startup")
async def startup_event():
    # Create necessary directories with error handling
    try:
        os.makedirs("static/uploads", exist_ok=True)
        os.makedirs("reports", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
    except Exception as e:
        logger.warning(f"Could not create directories: {e}")

    print(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting up...")
    print(f"📱 Debug mode: {settings.DEBUG}")
    print(f"🔧 Device: {settings.DEVICE}")
    print(f"📁 Model path: {settings.MODEL_PATH}")

    # Get model download config
    download_config = settings.get_model_download_config()
    if download_config.get('preload_on_startup', True):
        print("🔄 Preloading ML models...")

        # Start model preloading in background task
        asyncio.create_task(preload_ml_models())
    else:
        print("⏭️ Model preloading disabled, will load on first request")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    print(f"🛑 {settings.APP_NAME} shutting down...")


# For local development
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )