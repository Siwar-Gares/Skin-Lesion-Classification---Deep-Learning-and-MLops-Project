# 🔬 Skin Lesion Classification - Deep Learning & MLOps Project

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **production-ready deep learning system** for automated skin lesion classification with complete MLOps pipeline. Built with PyTorch, FastAPI, Streamlit, and modern DevOps tools.

## 🎯 Overview

This project implements an end-to-end solution for classifying dermatoscopic images into 7 types of skin lesions using state-of-the-art deep learning and MLOps best practices.

### Key Features

✨ **Deep Learning**
- EfficientNet-B3 architecture (12M parameters)
- Trained on HAM10000 dataset (10,015 images)
- 85%+ validation accuracy
- Data augmentation with Albumentations

🚀 **Production API**
- FastAPI backend with Swagger docs
- RESTful endpoints for predictions
- Docker containerization

🎨 **Web Interface**
- Interactive Streamlit application
- Real-time predictions
- Model information dashboard

📊 **MLOps Pipeline**
- **MLflow**: Experiment tracking
- **DVC**: Data versioning
- **GitHub Actions**: CI/CD
- **Monitoring**: Drift detection
- **Testing**: Unit & integration tests

## 🏥 Supported Lesion Types

| Code | Description | Severity |
|------|-------------|----------|
| **akiec** | Actinic Keratoses | ⚠️ Precancerous |
| **bcc** | Basal Cell Carcinoma | 🔴 Malignant |
| **bkl** | Benign Keratosis | 🟢 Benign |
| **df** | Dermatofibroma | 🟢 Benign |
| **mel** | Melanoma | 🔴 Malignant |
| **nv** | Melanocytic Nevi | 🟢 Benign |
| **vasc** | Vascular Lesions | 🟡 Vascular |

## 🚀 Quick Start

### Docker (Recommended)

```bash
docker-compose up -d
```

Access:
- API Docs: http://localhost:8000/docs
- Web App: http://localhost:8501
- MLflow: http://localhost:5000

### Local Development

```bash
# Setup
setup_mlops.bat

# Train
python train.py --epochs 50

# Run API
uvicorn src.api.main:app --port 8000

# Run UI
streamlit run streamlit_app/app.py
```

## 📊 MLOps Features

### Experiment Tracking
```bash
mlflow ui --port 5000
```

### Data Versioning
```bash
dvc init
dvc add D:/dl_data/data/HAM10000
dvc push
```

### CI/CD Pipeline
- Automated testing on every push
- Code quality checks
- Docker builds

### Model Monitoring
```bash
streamlit run src/monitoring/dashboard.py
```

## 📁 Project Structure

```
├── src/
│   ├── api/           # FastAPI backend
│   ├── models/        # Model architecture
│   ├── data/          # Data loading
│   ├── monitoring/    # MLOps monitoring
│   └── utils/         # Utilities
├── streamlit_app/     # Web interface
├── tests/             # Unit tests
├── .github/workflows/ # CI/CD
├── train.py           # Training script
└── docker-compose.yml # Container orchestration
```

## 🛠️ Technology Stack

- **Deep Learning**: PyTorch 2.6, EfficientNet-B3
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit
- **MLOps**: MLflow, DVC, GitHub Actions
- **Deployment**: Docker, Docker Compose
- **Testing**: Pytest
- **Monitoring**: Scipy (drift detection)

## ⚠️ Medical Disclaimer

This application is for **educational and research purposes only**. NOT intended for medical diagnosis or clinical decision-making. Always consult qualified healthcare professionals for medical concerns.

## 📚 Documentation

- [MLOps Setup Guide](MLOPS_SETUP.md)
- [Quick Start Guide](QUICK_START_MLOPS.md)
- [GitHub Deployment](GITHUB_DEPLOYMENT_GUIDE.md)

## 📄 License

MIT License - see LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please read contributing guidelines first.

## 📧 Contact

For questions or collaborations, please open an issue on GitHub.

---

**Built with ❤️ for advancing AI in healthcare**