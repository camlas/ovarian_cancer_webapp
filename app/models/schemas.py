from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for image prediction"""
    image_path: str = Field(..., description="Path to the image file")
    include_attention: bool = Field(default=True, description="Include attention maps in response")

class PredictionResponse(BaseModel):
    """Response model for image prediction"""
    image_path: str
    prediction: str  # "Cancer" or "Non-Cancer"
    probability: float = Field(..., ge=0.0, le=1.0, description="Prediction probability")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    prediction_class: int = Field(..., ge=0, le=1, description="0 for Non-Cancer, 1 for Cancer")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    model_info: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    """Response model for file upload"""
    success: bool
    message: str
    filename: Optional[str] = None
    file_path: Optional[str] = None
    file_size: Optional[int] = None

class SampleImagesResponse(BaseModel):
    """Response model for sample images"""
    cancer_samples: List[str]
    non_cancer_samples: List[str]
    total_samples: int

class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    status: str
    device: Optional[str] = None
    feature_extractor: Optional[str] = None
    classifier: Optional[str] = None
    selected_features_count: Optional[int] = None
    confidence_threshold: Optional[float] = None

class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: bool = True
    message: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

class ReportRequest(BaseModel):
    """Request model for report generation"""
    prediction_result: Dict[str, Any]
    include_visualizations: bool = Field(default=True)
    include_attention_maps: bool = Field(default=True)
    patient_info: Optional[Dict[str, str]] = None

class ReportResponse(BaseModel):
    """Response model for report generation"""
    success: bool
    report_id: Optional[str] = None
    report_path: Optional[str] = None
    download_url: Optional[str] = None
    message: str