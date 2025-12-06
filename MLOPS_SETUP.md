# MLOps Setup Guide

## 🚀 Quick Start

### 1. Run Setup Script
```cmd
setup_mlops.bat
```

This will:
- Create virtual environment
- Install all dependencies (main + MLOps)
- Initialize DVC
- Create necessary directories
- Install Git LFS

### 2. Configure DVC Remote Storage

#### Option A: Google Drive
```cmd
dvc remote add -d storage gdrive://your_folder_id
dvc remote modify storage gdrive_acknowledge_abuse true
```

#### Option B: Amazon S3
```cmd
dvc remote add -d storage s3://mybucket/path
dvc remote modify storage access_key_id YOUR_KEY
dvc remote modify storage secret_access_key YOUR_SECRET
```

#### Option C: Azure Blob
```cmd
dvc remote add -d storage azure://mycontainer/path
dvc remote modify storage account_name YOUR_ACCOUNT
```

### 3. Track Data with DVC
```cmd
:: Track dataset
dvc add D:/dl_data/data/HAM10000

:: Track model checkpoints
dvc add checkpoints/best_model.pth

:: Commit DVC files
git add D:/dl_data/data/HAM10000.dvc checkpoints/best_model.pth.dvc .dvc
git commit -m "Track data and models with DVC"
```

### 4. Run DVC Pipeline
```cmd
:: Run entire pipeline
dvc repro

:: Run specific stage
dvc repro train

:: Show pipeline visualization
dvc dag
```

### 5. Push Everything
```cmd
:: Push code to GitHub
git push

:: Push data to DVC remote
dvc push
```

---

## 📊 Experiment Tracking with MLflow

### Start MLflow Server
```cmd
mlflow ui --port 5000
```

Visit: http://localhost:5000

### Log Experiments
MLflow is integrated into the training pipeline. Every run automatically logs:
- Hyperparameters
- Metrics (loss, accuracy, F1, etc.)
- Model artifacts
- Training plots

---

## 🔄 CI/CD with GitHub Actions

### Workflows Included

#### 1. CI/CD Pipeline (`.github/workflows/ci-cd.yml`)
Runs on every push/PR:
- Code linting (flake8)
- Unit tests (pytest)
- Code coverage
- Docker builds
- Code quality checks (black, isort, pylint)

#### 2. Model Training (`.github/workflows/train-model.yml`)
Manual workflow for training:
- Triggered manually from GitHub Actions tab
- Configure epochs and architecture
- Uploads trained model as artifact

### Enable GitHub Actions
1. Push workflows to GitHub
2. Go to repository → Actions tab
3. Enable workflows
4. Set up secrets if needed (for cloud deployment)

---

## 📈 Model Monitoring

### 1. Data Drift Detection
```python
from src.monitoring import DriftDetector

# Create detector with reference data
detector = DriftDetector(reference_data=train_df, threshold=0.05)

# Detect drift
drift_report = detector.detect_dataset_drift(current_data)

# Save report
detector.save_report('metrics/drift_report.json')

# Print summary
print(detector.get_summary())
```

### 2. Performance Monitoring
```python
from src.monitoring import ModelMonitor

# Create monitor
monitor = ModelMonitor(model_name='skin_lesion_classifier')

# Log predictions
monitor.log_prediction(
    prediction='mel',
    confidence=0.95,
    true_label='mel',
    metadata={'image_id': '123'}
)

# Compute metrics
metrics = monitor.compute_performance_metrics()
monitor.save_logs('metrics/monitoring_logs.json')
```

### 3. Monitoring Dashboard
```cmd
streamlit run src/monitoring/dashboard.py --server.port 8502
```

Visit: http://localhost:8502

---

## 🐳 Docker Deployment

### Local Development
```cmd
:: Build and start all services
docker-compose up --build

:: Run in background
docker-compose up -d

:: Stop services
docker-compose down

:: View logs
docker-compose logs -f api
docker-compose logs -f streamlit
```

### Production Deployment
```cmd
:: Build for production
docker-compose -f docker-compose.prod.yml up -d

:: Scale services
docker-compose up -d --scale api=3
```

---

## 🧪 Testing

### Run All Tests
```cmd
pytest tests/ -v
```

### Run with Coverage
```cmd
pytest tests/ -v --cov=src --cov-report=html
```

### Run Specific Test File
```cmd
pytest tests/test_models.py -v
```

### Run Single Test
```cmd
pytest tests/test_models.py::TestSkinLesionClassifier::test_model_creation -v
```

---

## 📋 DVC Pipeline Stages

### 1. Data Preparation
```yaml
stages:
  prepare_data:
    cmd: python src/data/preprocessing.py
    deps: [src/data/preprocessing.py, data/raw]
    outs: [data/processed]
```

### 2. Training
```yaml
  train:
    cmd: python train.py --epochs 50
    deps: [train.py, data/processed]
    outs: [checkpoints/best_model.pth]
    metrics: [metrics/training_metrics.json]
```

### 3. Evaluation
```yaml
  evaluate:
    cmd: python -m src.models.evaluate
    deps: [checkpoints/best_model.pth]
    metrics: [metrics/evaluation_metrics.json]
```

---

## 🔧 Configuration Management

### params.yaml
Edit `params.yaml` to configure:
- Data paths and splits
- Model architecture
- Training hyperparameters
- Augmentation settings
- Monitoring thresholds

### Update Parameters
```cmd
:: Edit params.yaml
notepad params.yaml

:: Rerun pipeline with new params
dvc repro
```

---

## 📊 Metrics & Plots

### Tracked Metrics
- `metrics/training_metrics.json` - Training history
- `metrics/evaluation_metrics.json` - Test set performance
- `metrics/drift_report.json` - Data drift analysis
- `metrics/monitoring_logs.json` - Production predictions

### Generated Plots
- `plots/training_curves.png` - Loss/accuracy curves
- `plots/confusion_matrix.png` - Confusion matrix
- `plots/roc_curves.png` - ROC curves per class

---

## 🚨 Troubleshooting

### DVC Issues
```cmd
:: Reset DVC cache
dvc cache clear

:: Fix broken links
dvc checkout --force

:: Repair DVC
dvc repair
```

### Docker Issues
```cmd
:: Remove all containers
docker-compose down -v

:: Clean build
docker system prune -a
docker-compose up --build --force-recreate
```

### Git Issues
```cmd
:: Large files detected
git rm --cached large_file
git add .gitignore
git commit --amend

:: Reset to last commit
git reset --hard HEAD
```

---

## 📚 Additional Resources

- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Docker Compose Documentation](https://docs.docker.com/compose/)

---

## ✅ Checklist

- [ ] Virtual environment created
- [ ] All dependencies installed
- [ ] DVC initialized
- [ ] DVC remote configured
- [ ] Data tracked with DVC
- [ ] MLflow UI accessible
- [ ] Docker containers running
- [ ] Tests passing
- [ ] GitHub Actions enabled
- [ ] Monitoring dashboard working

---

**Last Updated**: December 2024  
**Status**: Production Ready
