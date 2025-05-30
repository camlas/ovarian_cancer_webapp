# Ovarian Cancer AI Detection System

<div align="center">

<img src="static/images/camlas-background.png" alt="CAMLAS Logo" height="100">

**Advanced AI-powered ovarian cancer detection using deep learning and medical imaging analysis**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red)](https://pytorch.org)
[![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-3.4-blue)](https://tailwindcss.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CAMLAS](https://img.shields.io/badge/CAMLAS-Innovation%20Hub-purple)](https://camlaslab.org.bd)

[Demo](https://your-demo-url.com) • [Documentation](https://your-docs-url.com) • [Research Paper](https://your-paper-url.com) • [Dataset](https://your-dataset-url.com)

</div>

## 🔬 Overview

This project presents a cutting-edge **multi-stage deep learning system** for automated ovarian cancer detection from histopathological images. Developed by the CAMLAS Innovation Hub, our AI system achieves **99.09% accuracy** using advanced attention mechanisms and explainable AI techniques.

### 🎯 Key Features

- **🧠 Advanced AI**: Multi-stage CNN with attention mechanisms
- **📊 High Accuracy**: 99.09% accuracy, 99.21% sensitivity
- **🔍 Explainable AI**: SHAP-based feature interpretation
- **⚡ Real-time Analysis**: Fast processing with confidence scoring
- **📱 Web Interface**: User-friendly FastAPI-based application
- **🎨 Modern UI**: Built with Tailwind CSS for responsive design
- **📋 PDF Reports**: Comprehensive analysis reports generation
- **🔒 Research Grade**: Designed for clinical research applications

## 🏗️ System Architecture

```mermaid
graph TB
    A[Medical Image Upload] --> B[Image Preprocessing]
    B --> C[CLAHE Enhancement]
    C --> D[Bilateral Filtering]
    D --> E[AttentionResNet50 Feature Extraction]
    E --> F[SHAP Feature Selection]
    F --> G[AttCNN Classification]
    G --> H[Confidence Estimation]
    H --> I[Result Visualization]
    I --> J[PDF Report Generation]
```

### 📋 Technical Pipeline

1. **Image Preprocessing**
   - CLAHE (Contrast Limited Adaptive Histogram Equalization)
   - Bilateral filtering for noise reduction
   - Standardized resizing and normalization

2. **Feature Extraction**
   - Pre-trained ResNet-50 with custom attention mechanisms
   - Deep feature extraction (2048 features)
   - Attention-based region focusing

3. **Feature Selection**
   - SHAP (SHapley Additive exPlanations) analysis
   - Top 500 most important features selection
   - Explainable AI for interpretability

4. **Classification**
   - Attention-based CNN classifier
   - Probabilistic output with confidence estimation
   - Binary classification (Cancer/Non-Cancer)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 16+ (for Tailwind CSS)
- CUDA-compatible GPU (optional, for faster processing)
- 8GB+ RAM recommended

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/camlas/ovarian-cancer-detection.git
   cd ovarian-cancer-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install Node.js dependencies and build CSS**
   ```bash
   npm install
   npm run build-css-prod
   ```

5. **Download pre-trained models**
   ```bash
   # Models will be automatically downloaded on first run
   # Or manually place models in ml_assets/saved_models/
   ```

6. **Run the application**
   ```bash
   python -m app.main
   ```

7. **Access the web interface**
   - Open your browser and navigate to `http://localhost:8000`
   - Upload medical images for analysis
   - View results and download reports

## 🎨 Frontend Development

### Tailwind CSS Setup

This project uses **Tailwind CSS CLI** for optimal performance and customization:

#### Development Workflow
```bash
# Watch for changes and rebuild CSS automatically
npm run watch-css

# Build CSS for development
npm run build-css

# Build minified CSS for production
npm run build-css-prod
```

#### Custom Configuration
The project includes custom Tailwind configuration in `tailwind.config.js`:

- **Custom Colors**: Primary, secondary, accent, and CAMLAS brand colors
- **Custom Animations**: Float effects, slide-up transitions, gentle bounces
- **Extended Theme**: Additional spacing, fonts, and utilities
- **Component Classes**: Pre-built navigation, citation, and UI components

#### File Structure
```
static/css/
├── src/
│   └── input.css          # Source Tailwind file
├── output.css             # Generated CSS (committed to repo)
└── custom.css             # Additional custom styles
```

#### Key Features
- **Optimized Bundle**: Only includes used CSS classes (~50-200KB vs 3.7MB CDN)
- **Custom Components**: Reusable UI components with consistent styling
- **Responsive Design**: Mobile-first approach with custom breakpoints
- **Performance**: No runtime compilation, better caching

## 📊 Performance Metrics

| Metric | Score |
|--------|-------|
| **Accuracy** | 99.09% |
| **Sensitivity** | 99.21% |
| **Specificity** | 98.96% |
| **Precision** | 99.17% |
| **F1-Score** | 99.06% |
| **AUC** | 98.18% |

## 📁 Project Structure

```
ovarian_cancer_webapp/
├── app/                          # Main application
│   ├── routers/                  # API route handlers
│   ├── services/                 # Business logic
│   ├── models/                   # Data models
│   ├── config.py                 # Configuration
│   └── main.py                   # Application entry point
├── templates/                    # HTML templates
├── static/                       # CSS, JS, images
│   ├── css/
│   │   ├── src/input.css        # Tailwind source
│   │   ├── output.css           # Generated CSS
│   │   └── custom.css           # Custom styles
│   └── js/                      # JavaScript files
├── ml_assets/                    # ML models and test data
├── reports/                      # Generated PDF reports
├── node_modules/                 # Node.js dependencies
├── package.json                  # Node.js configuration
├── tailwind.config.js            # Tailwind configuration
├── config.yaml                   # Application configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
app:
  name: "Ovarian Cancer Detection System"
  version: "1.0.0"
  description: "AI-powered ovarian cancer detection"

# Model settings
model:
  device: "auto"  # auto, cpu, cuda, mps
  confidence_threshold: 0.5
  batch_size: 1

# Performance metrics
performance:
  accuracy: 0.9909
  sensitivity: 0.9921
  specificity: 0.9896
```

## 🧪 API Documentation

### REST API Endpoints

- **POST** `/test/upload` - Upload medical image
- **POST** `/test/predict` - Get AI prediction
- **GET** `/test/sample-images` - Get sample images
- **POST** `/report/quick-generate` - Generate PDF report
- **GET** `/health` - Health check

### Example Usage

```python
import requests

# Upload and analyze image
files = {'file': open('medical_image.jpg', 'rb')}
response = requests.post('http://localhost:8000/test/upload', files=files)
upload_result = response.json()

# Get prediction
data = {'file_path': upload_result['file_path']}
response = requests.post('http://localhost:8000/test/predict', data=data)
prediction = response.json()

print(f"Prediction: {prediction['prediction']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

## 🧬 Dataset

Our system is trained on the **STRAMPN Histopathological Images** dataset:

- **Total Images**: 10,000+ high-resolution histopathological images
- **Classes**: Cancer (5,000+) and Non-Cancer (5,000+) samples
- **Resolution**: 224x224 pixels (standardized)
- **Format**: JPG, PNG, TIFF supported
- **Source**: Clinical research institutions

## 🎓 Research Team

- **Md. Faruk Hosen**
- **Francis Rudra D Cruze**
- **Dr. Goh Kah Ong Michael**
- **Dr. S M Hasan Mahmud**

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   npm install
   ```
4. Start development servers:
   ```bash
   npm run watch-css  # Terminal 1: Watch CSS changes
   python -m app.main # Terminal 2: Run FastAPI server
   ```
5. Make your changes and test
6. Build production CSS: `npm run build-css-prod`
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## 👨‍💻 Development

This application was developed by **Francis Rudra D Cruze** ([@rudradcruze](https://github.com/rudradcruze)) as part of the CAMLAS Innovation Hub research initiative.

## 🛡️ Security & Privacy

- **Data Privacy**: No medical images are permanently stored
- **Secure Processing**: All data processing happens locally
- **HIPAA Compliance**: Designed with healthcare data protection in mind
- **Research Use**: Intended for research and educational purposes only

## ⚠️ Disclaimer

This system is designed for **research and educational purposes only**. It should not be used as the sole basis for medical diagnosis or treatment decisions. Always consult with qualified healthcare professionals for medical concerns.

## 📞 Support & Contact

- **Email**: camlaslab1@gmail.com
- **Website**: [CAMLAS Innovation Hub](https://camlas.github.io/)

## 🙏 Acknowledgments

- CAMLAS Innovation Hub Bangladesh for research support
- Contributing medical institutions for dataset provision
- Open-source community for foundational libraries
- Healthcare professionals for domain expertise validation

---

<div align="center">

**Made by CAMLAS Innovation Hub Bangladesh**

[Website](https://camlas.github.io/) • [Research](https://camlas.github.io/publications) • [Contact](mailto:camlaslab1@gmail.com)

</div>