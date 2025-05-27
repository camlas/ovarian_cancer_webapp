import os
import numpy as np
import pickle
import joblib
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict, Any
import logging
from pathlib import Path
import requests

from app.config import settings

# Set up logging
logging.basicConfig(level=getattr(logging, settings.LOG_LEVEL))
logger = logging.getLogger(__name__)


def download_model_from_github(url: str, local_path: str, show_progress: bool = True, timeout: int = 300) -> bool:
    try:
        if Path(local_path).exists():
            logger.info(f"✅ Model already exists at {local_path}")
            return True

        logger.info(f"🔄 Downloading model from GitHub: {url}")

        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)

        # Download with progress and timeout
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        downloaded_size = 0

        # Progress tracking
        last_log_mb = 0

        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)

                    if show_progress and total_size > 0:
                        progress = (downloaded_size / total_size) * 100
                        current_mb = downloaded_size // (1024 * 1024)
                        total_mb = total_size // (1024 * 1024)

                        # Log progress every 10MB
                        if current_mb >= last_log_mb + 10:
                            logger.info(f"📥 Download progress: {progress:.1f}% ({current_mb}MB/{total_mb}MB)")
                            last_log_mb = current_mb

        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        logger.info(f"✅ Model downloaded successfully: {Path(local_path).name} ({file_size_mb:.1f}MB)")
        return True

    except requests.exceptions.Timeout:
        logger.error(f"❌ Download timeout ({timeout}s) for {Path(local_path).name}")
        if Path(local_path).exists():
            Path(local_path).unlink()
        return False
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ Download failed for {Path(local_path).name}: {str(e)}")
        if Path(local_path).exists():
            Path(local_path).unlink()
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error downloading {Path(local_path).name}: {str(e)}")
        if Path(local_path).exists():
            Path(local_path).unlink()
        return False


class CLAHE:
    """CLAHE transformation for image enhancement"""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img_np = np.array(img)

        if len(img_np.shape) == 3:
            lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            cl = clahe.apply(l)
            updated_lab = cv2.merge((cl, a, b))
            img_np = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
            img_np = clahe.apply(img_np)

        return Image.fromarray(img_np)


class BilateralFilter:
    """Bilateral filter for noise reduction"""

    def __init__(self, d=9, sigma_color=75, sigma_space=75):
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space

    def __call__(self, img):
        img_np = np.array(img)
        filtered = cv2.bilateralFilter(img_np, self.d, self.sigma_color, self.sigma_space)
        return Image.fromarray(filtered)


class AttentionBlock(nn.Module):
    """Attention mechanism for feature enhancement"""

    def __init__(self, in_features):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_features, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_map = self.attention(x)
        return x * attention_map, attention_map


class AttentionResNet50(nn.Module):
    """ResNet50 with attention mechanism for feature extraction"""

    def __init__(self, num_classes=2):
        super(AttentionResNet50, self).__init__()
        from torchvision import models
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.attention = AttentionBlock(2048)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def extract_features(self, x):
        features = self.features(x)
        attended_features, attention_map = self.attention(features)
        pooled_features = self.gap(attended_features)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)
        return flattened_features, features, attention_map

    def forward(self, x):
        features = self.features(x)
        attended_features, attention_map = self.attention(features)
        pooled_features = self.gap(attended_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        output = self.fc(pooled_features)
        return output


class AttCNNClassifier(nn.Module):
    """Attention-based CNN classifier"""

    def __init__(self, input_dim, hidden_dim=256):
        super(AttCNNClassifier, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        attended_features = x * attention_weights
        logits = self.classifier(attended_features)
        return logits, attention_weights


class MLService:
    """Main ML service for ovarian cancer detection"""

    def __init__(self):
        self.device = self._get_device()
        self.models_loaded = False
        self.feature_extractor = None
        self.rf_selector = None
        self.selected_features = None
        self.scaler = None
        self.classifier = None
        self.transform = None

        logger.info(f"ML Service initialized with device: {self.device}")

    def _get_device(self) -> torch.device:
        """Determine the best available device"""
        if settings.DEVICE.lower() == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        else:
            return torch.device(settings.DEVICE.lower())

    def load_models(self) -> bool:
        try:
            logger.info("🔄 Loading ML models...")

            # Use /tmp for Railway (ephemeral storage)
            model_path = Path("/tmp/models") if os.environ.get('MODEL_PATH') else Path(settings.MODEL_PATH)
            model_path.mkdir(parents=True, exist_ok=True)

            # Get model download configuration from config
            download_config = settings.get_model_download_config()

            # Check if download is enabled
            if not download_config.get('enable_download', True):
                logger.info("📁 Model download disabled, using local files only")
            else:
                # Get model URLs from config - ONLY download essential models first
                model_files = settings.get_github_model_urls()

                # Filter to essential models only to save memory
                essential_models = {
                    'feature_extractor_resnet50.pth': model_files.get('feature_extractor_resnet50.pth')
                }

                logger.info(f"📦 Downloading {len(essential_models)} essential model files")

                # Download essential models only
                for filename, url in essential_models.items():
                    if url:  # Make sure URL exists
                        local_path = model_path / filename
                        if not local_path.exists():
                            logger.info(f"📥 Downloading {filename}...")
                            success = download_model_from_github(
                                url,
                                str(local_path),
                                show_progress=download_config.get('show_progress', True),
                                timeout=download_config.get('download_timeout', 300)
                            )
                            if not success:
                                logger.error(f"❌ Failed to download {filename}")
                                if not download_config.get('fallback_to_mock', True):
                                    return False
                        else:
                            logger.info(f"✅ {filename} already exists locally")

            # Check if essential files exist
            essential_files = ['feature_extractor_resnet50.pth']

            missing_files = []
            for file in essential_files:
                if not (model_path / file).exists():
                    missing_files.append(file)

            if missing_files:
                logger.warning(f"⚠️ Missing essential model files: {missing_files}")
                if download_config.get('fallback_to_mock', True):
                    logger.info("🎭 Fallback to mock service enabled, creating mock service...")
                    return self._create_mock_service()
                else:
                    logger.error("❌ Fallback to mock disabled, cannot proceed without models")
                    return False

            logger.info("📚 Loading essential model components...")

            # Load feature extractor with memory optimization
            logger.info("🧠 Loading feature extractor (AttentionResNet50)...")
            try:
                # Force CPU to save memory
                self.device = torch.device('cpu')
                logger.info(f"🔧 Using device: {self.device} (forced CPU for memory efficiency)")

                self.feature_extractor = AttentionResNet50().to(self.device)

                # Load checkpoint with memory optimization
                checkpoint = torch.load(
                    model_path / 'feature_extractor_resnet50.pth',
                    map_location=self.device,
                    weights_only=True  # Security and memory optimization
                )

                self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
                self.feature_extractor.eval()

                # Clear checkpoint from memory immediately
                del checkpoint

                logger.info("✅ Feature extractor loaded successfully")

            except Exception as e:
                logger.error(f"❌ Failed to load feature extractor: {e}")
                if download_config.get('fallback_to_mock', True):
                    return self._create_mock_service()
                else:
                    return False

            # Try to load other components, but don't fail if missing
            logger.info("🔧 Loading additional components (optional)...")
            try:
                # Try to load additional models if they exist
                optional_files = {
                    'rf_feature_selector.pkl': 'rf_selector',
                    'selected_features.pkl': 'selected_features',
                    'feature_scaler.pkl': 'scaler',
                    'best_attcnn_classifier.pth': 'classifier'
                }

                components_loaded = 0
                for filename, component_name in optional_files.items():
                    try:
                        file_path = model_path / filename
                        if file_path.exists():
                            if filename.endswith('.pkl'):
                                if component_name == 'selected_features':
                                    with open(file_path, 'rb') as f:
                                        setattr(self, component_name, pickle.load(f))
                                else:
                                    setattr(self, component_name, joblib.load(file_path))
                            elif filename.endswith('.pth'):
                                classifier_checkpoint = torch.load(file_path, map_location=self.device,
                                                                   weights_only=True)
                                input_dim = classifier_checkpoint['input_dim']
                                self.classifier = AttCNNClassifier(input_dim).to(self.device)
                                self.classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
                                self.classifier.eval()
                                del classifier_checkpoint  # Clear from memory

                            components_loaded += 1
                            logger.info(f"✅ {component_name} loaded successfully")
                        else:
                            logger.info(f"⏭️ {filename} not found, will use simplified prediction")

                    except Exception as component_error:
                        logger.warning(f"⚠️ Failed to load {component_name}: {component_error}")

                logger.info(f"📊 Loaded {components_loaded}/{len(optional_files)} optional components")

            except Exception as e:
                logger.warning(f"⚠️ Some optional components failed to load: {e}")
                # Continue anyway - we can still do basic predictions

            # Set up image transform pipeline (lightweight version)
            logger.info("🖼️ Setting up image transformation pipeline...")

            try:
                # Full pipeline if OpenCV is available
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
                    BilateralFilter(d=9, sigma_color=75, sigma_space=75),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except Exception:
                # Fallback pipeline without OpenCV
                logger.warning("⚠️ OpenCV not available, using basic image pipeline")
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            self.models_loaded = True
            logger.info("🎉 Essential models loaded successfully!")

            # Log model summary
            logger.info("📊 Model Loading Summary:")
            logger.info(f"   🧠 Feature Extractor: AttentionResNet50 ✅")
            logger.info(
                f"   🔧 RF Selector: {'✅' if hasattr(self, 'rf_selector') and self.rf_selector else '⏭️ Skipped'}")
            logger.info(
                f"   📝 Selected Features: {'✅' if hasattr(self, 'selected_features') and self.selected_features else '⏭️ Skipped'}")
            logger.info(f"   📏 Scaler: {'✅' if hasattr(self, 'scaler') and self.scaler else '⏭️ Skipped'}")
            logger.info(f"   🎯 Classifier: {'✅' if hasattr(self, 'classifier') and self.classifier else '⏭️ Skipped'}")
            logger.info(f"   🔧 Device: {self.device} ✅")
            logger.info(f"   💾 Memory Mode: Optimized for Railway")

            return True

        except Exception as e:
            logger.error(f"💥 Unexpected error loading models: {str(e)}")
            logger.exception("Full traceback:")

            # Try to fall back to mock service
            try:
                download_config = settings.get_model_download_config()
                if download_config.get('fallback_to_mock', True):
                    logger.info("🎭 Attempting fallback to mock service...")
                    return self._create_mock_service()
                else:
                    logger.error("❌ Fallback to mock disabled, failing completely")
                    return False
            except Exception as fallback_error:
                logger.error(f"💥 Even fallback failed: {fallback_error}")
                return False

    def _create_mock_service(self) -> bool:
        logger.info("Creating mock ML service for demo")
        self.models_loaded = True
        return True

    def predict_image(self, image_path: str) -> Dict[str, Any]:
        if not self.models_loaded:
            if not self.load_models():
                raise RuntimeError("Failed to load ML models")

        try:
            logger.info(f"Predicting image: {os.path.basename(image_path)}")

            # If running in mock mode (missing models), return mock prediction
            if (not self.feature_extractor or not self.classifier or
                    not self.selected_features or not self.scaler):
                return self._mock_prediction(image_path)

            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Step 1: Feature extraction
            with torch.no_grad():
                features, raw_features, attention_map = self.feature_extractor.extract_features(image_tensor)
                features = features.cpu().numpy()

            # Step 2: Feature selection using SHAP-selected indices
            if all(idx < features.shape[1] for idx in self.selected_features):
                selected_feature_values = features[:, self.selected_features]
            else:
                raise ValueError("Some SHAP indices are out of bounds")

            # Step 3: Scale features
            scaled_features = self.scaler.transform(selected_feature_values)

            # Step 4: Classification
            features_tensor = torch.FloatTensor(scaled_features).to(self.device)

            with torch.no_grad():
                logits, attention_weights = self.classifier(features_tensor)
                probability = torch.sigmoid(logits).cpu().numpy()[0, 0]
                prediction = 1 if probability > settings.CONFIDENCE_THRESHOLD else 0

            # Calculate confidence
            confidence = float(abs(probability - 0.5) * 2)

            result = {
                'image_path': image_path,
                'prediction': 'Cancer' if prediction == 1 else 'Non-Cancer',
                'probability': float(probability),
                'confidence': confidence,
                'prediction_class': int(prediction),
                'attention_map': attention_map.cpu().numpy(),
                'raw_features_shape': features.shape,
                'selected_features_count': len(self.selected_features),
                'processing_device': str(self.device)
            }

            logger.info(f"Prediction: {result['prediction']} (prob: {probability:.4f}, conf: {confidence:.4f})")
            return result

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return self._mock_prediction(image_path)

    def _mock_prediction(self, image_path: str) -> Dict[str, Any]:
        """Generate mock prediction for demo purposes"""
        import random

        probability = random.uniform(0.1, 0.9)
        prediction = 1 if probability > 0.5 else 0
        confidence = random.uniform(0.7, 0.95)

        result = {
            'image_path': image_path,
            'prediction': 'Cancer' if prediction == 1 else 'Non-Cancer',
            'probability': float(probability),
            'confidence': confidence,
            'prediction_class': int(prediction),
            'attention_map': None,
            'raw_features_shape': (1, 2048),
            'selected_features_count': 500,
            'processing_device': str(self.device),
            'mock_mode': True
        }

        logger.info(f"Mock prediction: {result['prediction']} (prob: {probability:.4f})")
        return result

    def get_sample_images(self) -> Dict[str, list]:
        sample_path = Path("ml_assets/test_images")

        cancer_samples = []
        non_cancer_samples = []

        # Get cancer samples
        cancer_dir = sample_path / "cancer_samples"
        if cancer_dir.exists():
            cancer_samples = [
                str(f) for f in cancer_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            ]

        # Get non-cancer samples
        non_cancer_dir = sample_path / "non_cancer_samples"
        if non_cancer_dir.exists():
            non_cancer_samples = [
                str(f) for f in non_cancer_dir.iterdir()
                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            ]

        return {
            'cancer_samples': cancer_samples[:5],
            'non_cancer_samples': non_cancer_samples[:5]
        }

    def get_model_info(self) -> Dict[str, Any]:
        if not self.models_loaded:
            return {'status': 'not_loaded'}

        return {
            'status': 'loaded',
            'device': str(self.device),
            'feature_extractor': 'AttentionResNet50',
            'classifier': 'AttCNNClassifier',
            'selected_features_count': len(self.selected_features) if self.selected_features else 0,
            'confidence_threshold': settings.CONFIDENCE_THRESHOLD,
            'mock_mode': not all([self.feature_extractor, self.classifier, self.selected_features, self.scaler])
        }


# Global ML service instance
ml_service = MLService()