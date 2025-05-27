from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import os
import time
import uuid
from pathlib import Path
import shutil

from app.services.ml_service import ml_service
from app.models.schemas import (
    PredictionResponse,
    UploadResponse,
    SampleImagesResponse,
    ModelInfoResponse,
    ErrorResponse
)
from app.config import settings

router = APIRouter()


@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded ML models"""
    try:
        info = ml_service.get_model_info()
        return ModelInfoResponse(**info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sample-images", response_model=SampleImagesResponse)
async def get_sample_images():
    try:
        samples = ml_service.get_sample_images()
        return SampleImagesResponse(
            cancer_samples=samples['cancer_samples'],
            non_cancer_samples=samples['non_cancer_samples'],
            total_samples=len(samples['cancer_samples']) + len(samples['non_cancer_samples'])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload", response_model=UploadResponse)
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file for prediction"""
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")

        file_extension = Path(file.filename).suffix.lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"File type {file_extension} not allowed. Allowed types: {settings.ALLOWED_EXTENSIONS}"
            )

        # Check file size
        file_content = await file.read()
        if len(file_content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024 * 1024):.1f}MB"
            )

        # Generate unique filename
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        file_path = Path(settings.UPLOAD_PATH) / unique_filename

        # Save file
        with open(file_path, "wb") as buffer:
            buffer.write(file_content)

        return UploadResponse(
            success=True,
            message="File uploaded successfully",
            filename=unique_filename,
            file_path=str(file_path),
            file_size=len(file_content)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
async def predict_image(
        file_path: str = Form(...),
        include_attention: bool = Form(default=True)
):
    """Predict cancer probability for an uploaded image"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Image file not found")

        # Record start time
        start_time = time.time()

        # Make prediction
        result = ml_service.predict_image(file_path)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Get model info
        model_info = ml_service.get_model_info()

        return PredictionResponse(
            image_path=result['image_path'],
            prediction=result['prediction'],
            probability=result['probability'],
            confidence=result['confidence'],
            prediction_class=result['prediction_class'],
            processing_time=processing_time,
            model_info=model_info
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.post("/predict-sample")
async def predict_sample_image(sample_path: str = Form(...)):
    """Predict cancer probability for a sample image"""
    try:
        # Validate sample path
        full_path = Path(sample_path)
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="Sample image not found")

        # Check if it's in the allowed sample directories
        allowed_dirs = ["ml_assets/test_images/cancer_samples", "ml_assets/test_images/non_cancer_samples"]
        if not any(str(full_path).startswith(allowed_dir) for allowed_dir in allowed_dirs):
            raise HTTPException(status_code=403, detail="Access denied to this file")

        # Record start time
        start_time = time.time()

        # Make prediction
        result = ml_service.predict_image(str(full_path))

        # Calculate processing time
        processing_time = time.time() - start_time

        # Get model info
        model_info = ml_service.get_model_info()

        return PredictionResponse(
            image_path=result['image_path'],
            prediction=result['prediction'],
            probability=result['probability'],
            confidence=result['confidence'],
            prediction_class=result['prediction_class'],
            processing_time=processing_time,
            model_info=model_info
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.delete("/cleanup")
async def cleanup_uploads():
    """Clean up uploaded files (for testing purposes)"""
    try:
        upload_dir = Path(settings.UPLOAD_PATH)
        deleted_count = 0

        for file_path in upload_dir.glob("*"):
            if file_path.is_file() and file_path.name != ".gitkeep":
                file_path.unlink()
                deleted_count += 1

        return {"message": f"Deleted {deleted_count} uploaded files"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.get("/health")
async def test_health():
    """Health check for the test service"""
    try:
        model_info = ml_service.get_model_info()
        return {
            "status": "healthy",
            "ml_service": model_info['status'],
            "upload_path": settings.UPLOAD_PATH,
            "model_path": settings.MODEL_PATH
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }