import os
import numpy as np
import pickle
import joblib
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from typing import Dict, Any, Optional, Tuple
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
        """Load all trained models and components"""
        try:
            logger.info("🔄 Loading ML models...")

            model_path = Path(settings.MODEL_PATH)

            # Get model download configuration from config
            download_config = settings.get_model_download_config()

            # Check if download is enabled
            if not download_config.get('enable_download', True):
                logger.info("📁 Model download disabled, using local files only")
            else:
                # Get model URLs from config
                model_files = settings.get_github_model_urls()

                logger.info(f"📦 Found {len(model_files)} model files to check/download")

                # Download models if they don't exist locally
                for filename, url in model_files.items():
                    local_path = model_path / filename
                    if not local_path.exists():
                        logger.info(f"📥 Downloading {filename} from config URL...")
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

            # Check if all required files exist (local or downloaded)
            required_files = download_config.get('required_files', [
                'feature_extractor_resnet50.pth',
                'rf_feature_selector.pkl',
                'selected_features.pkl',
                'feature_scaler.pkl',
                'best_attcnn_classifier.pth'
            ])

            missing_files = []
            for file in required_files:
                if not (model_path / file).exists():
                    missing_files.append(file)

            if missing_files:
                logger.warning(f"⚠️ Missing model files: {missing_files}")
                # Check if fallback to mock is enabled
                if download_config.get('fallback_to_mock', True):
                    logger.info("🎭 Fallback to mock service enabled, creating mock service...")
                    return self._create_mock_service()
                else:
                    logger.error("❌ Fallback to mock disabled, cannot proceed without models")
                    return False

            logger.info("📚 Loading model components...")

            # Load feature extractor
            logger.info("🧠 Loading feature extractor (AttentionResNet50)...")
            self.feature_extractor = AttentionResNet50().to(self.device)

            try:
                checkpoint = torch.load(
                    model_path / 'feature_extractor_resnet50.pth',
                    map_location=self.device
                )
                self.feature_extractor.load_state_dict(checkpoint['model_state_dict'])
                self.feature_extractor.eval()
                logger.info("✅ Feature extractor loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load feature extractor: {e}")
                if download_config.get('fallback_to_mock', True):
                    return self._create_mock_service()
                else:
                    return False

            # Load other components (if they exist)
            try:
                logger.info("🔧 Loading RF feature selector...")
                self.rf_selector = joblib.load(model_path / 'rf_feature_selector.pkl')
                logger.info("✅ RF feature selector loaded successfully")

                logger.info("📝 Loading selected features...")
                with open(model_path / 'selected_features.pkl', 'rb') as f:
                    self.selected_features = pickle.load(f)
                logger.info(f"✅ Selected features loaded ({len(self.selected_features)} features)")

                logger.info("📏 Loading feature scaler...")
                self.scaler = joblib.load(model_path / 'feature_scaler.pkl')
                logger.info("✅ Feature scaler loaded successfully")

                logger.info("🎯 Loading AttCNN classifier...")
                classifier_checkpoint = torch.load(
                    model_path / 'best_attcnn_classifier.pth',
                    map_location=self.device
                )
                input_dim = classifier_checkpoint['input_dim']
                self.classifier = AttCNNClassifier(input_dim).to(self.device)
                self.classifier.load_state_dict(classifier_checkpoint['model_state_dict'])
                self.classifier.eval()
                logger.info(f"✅ AttCNN classifier loaded successfully (input_dim: {input_dim})")

            except Exception as e:
                logger.warning(f"⚠️ Some model components missing or failed to load: {e}")
                if download_config.get('fallback_to_mock', True):
                    logger.info("🎭 Creating mock service due to component loading failure...")
                    return self._create_mock_service()
                else:
                    logger.error("❌ Cannot proceed without all model components")
                    return False

            # Define image transform pipeline
            logger.info("🖼️ Setting up image transformation pipeline...")
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
                BilateralFilter(d=9, sigma_color=75, sigma_space=75),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            self.models_loaded = True
            logger.info("🎉 All models loaded successfully!")

            # Log model summary
            logger.info("📊 Model Loading Summary:")
            logger.info(f"   🧠 Feature Extractor: AttentionResNet50 ✅")
            logger.info(f"   🔧 RF Selector: {type(self.rf_selector).__name__} ✅")
            logger.info(f"   📝 Selected Features: {len(self.selected_features)} features ✅")
            logger.info(f"   📏 Scaler: {type(self.scaler).__name__} ✅")
            logger.info(f"   🎯 Classifier: AttCNNClassifier ✅")
            logger.info(f"   🔧 Device: {self.device} ✅")

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