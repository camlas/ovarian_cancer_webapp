from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
import os
from pathlib import Path
import traceback

from app.config import settings

router = APIRouter()

# Import report generator with error handling
try:
    from app.services.report_service import report_generator

    REPORT_SERVICE_AVAILABLE = True
except Exception as e:
    REPORT_SERVICE_AVAILABLE = False


@router.get("/test")
async def test_report_service():
    """Test if report service is working"""
    try:
        if not REPORT_SERVICE_AVAILABLE:
            return {
                "status": "error",
                "message": "Report service not available",
                "available": False
            }

        # Test basic functionality
        test_result = {
            "prediction": "Cancer",
            "probability": 0.85,
            "confidence": 0.92,
            "processing_time": 1.23,
            "model_info": {
                "device": "mps",
                "status": "loaded",
                "selected_features_count": 500
            }
        }

        report_path = report_generator.generate_report(test_result)

        return {
            "status": "success",
            "message": "Report service is working",
            "test_report_generated": True,
            "report_path": report_path
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Report service test failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


@router.post("/quick-generate")
async def quick_generate_report(request: Request):
    """Quick report generation for JavaScript calls"""
    try:
        if not REPORT_SERVICE_AVAILABLE:
            raise HTTPException(status_code=503, detail="Report service not available")

        # Get JSON data
        request_data = await request.json()

        prediction_result = request_data.get('prediction_result')
        if not prediction_result:
            raise HTTPException(status_code=400, detail="Missing prediction result")


        # Generate report
        report_path = report_generator.generate_report(
            prediction_result=prediction_result,
            patient_info=None,
            image_path=None
        )

        # Get report ID
        report_filename = os.path.basename(report_path)
        report_id = report_filename.split('_')[-1].replace('.pdf', '')


        return {
            "success": True,
            "report_id": report_id,
            "download_url": f"/report/download/{report_id}",
            "filename": f"ovarian_cancer_report_{report_id}.pdf",
            "message": "Report generated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@router.get("/download/{report_id}")
async def download_report(report_id: str):
    """Download a generated PDF report by ID"""
    try:
        # Find report file by ID
        reports_dir = Path(settings.REPORTS_PATH)
        report_files = list(reports_dir.glob(f"*_{report_id}.pdf"))

        if not report_files:
            raise HTTPException(status_code=404, detail=f"Report not found: {report_id}")

        report_path = report_files[0]

        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")


        # Return file response
        return FileResponse(
            path=str(report_path),
            media_type='application/pdf',
            filename=f"ovarian_cancer_report_{report_id}.pdf",
            headers={"Content-Disposition": "attachment"}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")


@router.get("/health")
async def report_health_check():
    """Health check for report service"""
    try:
        # Check if reports directory exists and is writable
        reports_dir = Path(settings.REPORTS_PATH)
        reports_dir.mkdir(exist_ok=True)

        return {
            "status": "healthy" if REPORT_SERVICE_AVAILABLE else "unhealthy",
            "reports_directory": str(reports_dir),
            "directory_exists": reports_dir.exists(),
            "report_service_available": REPORT_SERVICE_AVAILABLE
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "report_service_available": False
        }