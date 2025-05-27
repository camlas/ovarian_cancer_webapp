from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Dict, Any
import yaml
from pathlib import Path
from datetime import datetime
import os
import json
from dotenv import load_dotenv

load_dotenv()


def _parse_cors_origins() -> List[str]:
    cors_env = os.getenv("CORS_ORIGINS", "")

    try:
        if cors_env.startswith('[') and cors_env.endswith(']'):
            return json.loads(cors_env)

        # Parse as comma-separated string
        origins = [origin.strip().strip('"\'') for origin in cors_env.split(',')]
        return [origin for origin in origins if origin]

    except (json.JSONDecodeError, ValueError) as e:
        return ["http://localhost:8000", "http://127.0.0.1:8000"]


def _load_yaml_config() -> Dict[str, Any]:
    """Load configuration from config.yaml file - static function"""
    config_path = Path("config.yaml")
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Warning: Failed to load config.yaml: {e}")
            return {}
    return {}


# Load config once at module level
_config = _load_yaml_config()
_app_config = _config.get("app", {})


class Settings(BaseSettings):
    # Set these from YAML config with fallback defaults
    APP_NAME: str = Field(default=_app_config.get("name", "Ovarian Cancer Detection System"))
    APP_VERSION: str = Field(default=_app_config.get("version", "1.0.0"))
    APP_DESCRIPTION: str = Field(default=_app_config.get("description", "AI-powered ovarian cancer detection using medical imaging"))

    DEBUG: bool = Field(default=False)
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)

    # ML Model Settings
    MODEL_PATH: str = Field(default="./ml_assets/saved_models")
    UPLOAD_PATH: str = Field(default="./static/uploads")
    REPORTS_PATH: str = Field(default="./reports")
    MAX_FILE_SIZE: int = Field(default=10485760)  # 10MB
    ALLOWED_EXTENSIONS: List[str] = Field(default=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"])

    # Security
    SECRET_KEY: str = Field(default_factory=lambda: os.getenv("SECRET_KEY"))
    CORS_ORIGINS: List[str] = Field(default_factory=_parse_cors_origins)

    # ML Processing
    DEVICE: str = Field(default="auto")
    BATCH_SIZE: int = Field(default=1)
    CONFIDENCE_THRESHOLD: float = Field(default=0.5)
    FEATURE_EXTRACTOR: str = Field(default="AttentionResNet50")
    CLASSIFIER: str = Field(default="AttCNNClassifier")

    # Report Generation
    REPORT_TEMPLATE: str = Field(default="default")
    INCLUDE_VISUALIZATIONS: bool = Field(default=True)
    INCLUDE_ATTENTION_MAPS: bool = Field(default=True)

    # Logging
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE: str = Field(default="./logs/app.log")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml file"""
        return _config

    def get_project_status(self) -> str:
        """Get project status from config.yaml"""
        return _config.get('project_status', 'ACTIVE')

    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics from config.yaml"""
        return _config.get('performance', {})

    def get_researchers(self) -> List[Dict[str, Any]]:
        """Get researchers information from config.yaml"""
        return _config.get('researchers', [])

    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get algorithm information from config.yaml"""
        return _config.get('algorithm', {})

    def get_about_info(self) -> Dict[str, Any]:
        """Get about page information from config.yaml"""
        return _config.get('about', {})

    def get_sample_images(self) -> Dict[str, List[str]]:
        """Get sample images configuration from config.yaml"""
        return _config.get('sample_images', {
            'cancer_samples': [],
            'non_cancer_samples': []
        })

    def get_project_name(self) -> Dict[str, str]:
        return _config.get('project_name', {})

    def get_app_info(self) -> Dict[str, str]:
        return _config.get('app', {})

    def get_project_metadata(self) -> Dict[str, Any]:
        return _config.get('project_metadata', {})

    def get_page_titles(self) -> Dict[str, str]:
        return _config.get('page_titles', {})

    def get_collaborative_excellence(self) -> Dict[str, Any]:
        """Get collaborative excellence information from config.yaml"""
        return _config.get('collaborative_excellence', {
            'title': 'Collaborative Excellence',
            'subtitle': 'Our interdisciplinary approach brings together diverse expertise',
            'expertise_areas': []
        })

    def get_research_impact(self) -> Dict[str, Any]:
        """Get research impact information from config.yaml"""
        return _config.get('research_impact', {
            'title': 'Research Impact',
            'subtitle': 'Our research contributions',
            'metrics': []
        })

    def get_contact_info(self) -> Dict[str, str]:
        return {
            'github_url': _config.get('github_url', ''),
            'website_url': _config.get('website_url', ''),
            'contact_email': _config.get('contact_email', ''),
            'dataset_url': _config.get('dataset_url', '')
        }

    def get_research_page_title(self) -> Dict[str, str]:
        return _config.get('research_page', {})

    def get_citation_info(self) -> Dict[str, Any]:
        return _config.get('citation', {})

    def get_model_download_config(self) -> Dict[str, Any]:
        """Get model download configuration from config.yaml"""
        config = self.load_yaml_config()
        return config.get('model_download', {
            'github_base_url': 'https://github.com/camlas/ovarian_cancer_webapp/releases/download/feature_extractor_resnet50',
            'required_files': [
                'feature_extractor_resnet50.pth',
                'rf_feature_selector.pkl',
                'selected_features.pkl',
                'feature_scaler.pkl',
                'best_attcnn_classifier.pth'
            ],
            'enable_download': True,
            'fallback_to_mock': True,
            'preload_on_startup': True,
            'download_timeout': 300,
            'show_progress': True
        })

    def get_github_model_urls(self) -> Dict[str, str]:
        download_config = self.get_model_download_config()
        base_url = download_config['github_base_url']
        required_files = download_config['required_files']

        return {
            filename: f"{base_url}/{filename}"
            for filename in required_files
        }

    def get_full_config(self) -> Dict[str, Any]:
        contact_info = self.get_contact_info()

        return {
            'page_titles': self.get_page_titles(),
            'project_metadata': self.get_project_metadata(),
            'app_name': self.APP_NAME,
            'app': self.get_app_info(),
            'project_name': self.get_project_name(),
            'project_status': _config.get('project_status', 'ACTIVE'),
            'performance': self.get_performance_metrics(),
            'researchers': self.get_researchers(),
            'algorithm': self.get_algorithm_info(),
            'about': self.get_about_info(),
            'sample_images': self.get_sample_images(),
            'github_url': contact_info['github_url'],
            'website_url': contact_info['website_url'],
            'contact_email': contact_info['contact_email'],
            'contact_info': self.get_contact_info(),
            'current_year': datetime.now().year,
            'collaborative_excellence': self.get_collaborative_excellence(),
            'research_impact': self.get_research_impact(),
            'research_page': self.get_research_page_title(),
            'citation': self.get_citation_info(),
        }


# Create global settings instance
settings = Settings()

# Ensure paths exist after settings are loaded
Path(settings.MODEL_PATH).mkdir(parents=True, exist_ok=True)
Path(settings.UPLOAD_PATH).mkdir(parents=True, exist_ok=True)
Path(settings.REPORTS_PATH).mkdir(parents=True, exist_ok=True)
Path(settings.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)