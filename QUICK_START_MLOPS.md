# Quick Start Guide - MLOps Features

## 🚀 5-Minute Setup

### Step 1: Install Dependencies (2 min)
```cmd
setup_mlops.bat
```

### Step 2: Start Services (1 min)
```cmd
docker-compose up -d
```

### Step 3: Access Applications (1 min)
- **API Docs**: http://localhost:8000/docs
- **Web App**: http://localhost:8501
- **MLflow**: http://localhost:5000

### Step 4: Make a Prediction (1 min)
1. Go to http://localhost:8501
2. Click "Make Prediction" page
3. Upload a skin lesion image
4. View results!

---

## 📊 MLOps Features Overview

### 1. Experiment Tracking (MLflow)
**What it does**: Tracks all training experiments, metrics, and models

**How to use**:
```cmd
:: Start MLflow
mlflow ui --port 5000

:: Training automatically logs to MLflow
python train.py --epochs 50
```

**View results**: http://localhost:5000

### 2. Data Versioning (DVC)
**What it does**: Version control for datasets and models (like Git for data)

**How to use**:
```cmd
:: Track your data
dvc add D:/dl_data/data/HAM10000

:: Track your model
dvc add checkpoints/best_model.pth

:: Commit
git add .
git commit -m "Track data with DVC"

:: Push data to cloud
dvc push
```

### 3. CI/CD Pipeline (GitHub Actions)
**What it does**: Automatically tests code on every push

**How to use**:
- Push to GitHub
- Actions run automatically
- Check status in GitHub Actions tab

**What's tested**:
- ✅ Code quality (linting)
- ✅ Unit tests
- ✅ Docker builds
- ✅ Coverage reports

### 4. Model Monitoring
**What it does**: Detects data drift and tracks model performance

**How to use**:
```python
from src.monitoring import DriftDetector, ModelMonitor

# Detect drift
detector = DriftDetector(reference_data, threshold=0.05)
report = detector.detect_dataset_drift(current_data)

# Monitor predictions
monitor = ModelMonitor()
monitor.log_prediction('mel', confidence=0.95)
```

**Dashboard**: 
```cmd
streamlit run src/monitoring/dashboard.py --server.port 8502
```

---

## 🎯 Common Tasks

### Train a Model
```cmd
python train.py --epochs 50 --batch-size 32 --lr 1e-4
```

### Run All Tests
```cmd
pytest tests/ -v
```

### Deploy with Docker
```cmd
docker-compose up -d
```

### Check Data Drift
```python
python -c "
from src.monitoring import DriftDetector
import pandas as pd

detector = DriftDetector(pd.read_csv('train.csv'))
report = detector.detect_dataset_drift(pd.read_csv('new_data.csv'))
print(detector.get_summary())
"
```

### View Training History
```cmd
mlflow ui --port 5000
```

### Push to GitHub
```cmd
git add .
git commit -m "Update project"
git push
```

### Share Model Weights
```cmd
:: Option 1: DVC (recommended)
dvc push

:: Option 2: Git LFS
git lfs track "*.pth"
git add checkpoints/best_model.pth
git push
```

---

## 🔍 Monitoring Dashboard Features

### Performance Tab
- Total predictions
- Average confidence
- Accuracy (if labels available)
- Confidence distribution
- Predictions over time

### Data Drift Tab
- Features analyzed
- Drift detected
- Statistical tests (KS, Chi-square)
- P-values and statistics

### Logs Tab
- Recent predictions
- Timestamps
- Confidence scores
- True labels (if available)

---

## 📦 Project Structure

```
skin-lesion-classification/
├── src/
│   ├── data/              # Data loading
│   ├── models/            # Model architecture
│   ├── api/               # FastAPI backend
│   ├── monitoring/        # MLOps monitoring ← NEW!
│   └── utils/             # Utilities
├── streamlit_app/         # Web interface
├── tests/                 # Unit tests ← NEW!
├── .github/workflows/     # CI/CD ← NEW!
├── checkpoints/           # Model weights
├── metrics/               # Monitoring data ← NEW!
├── train.py               # Training script
├── dvc.yaml               # DVC pipeline ← NEW!
├── params.yaml            # Configuration ← NEW!
└── docker-compose.yml     # Container orchestration
```

---

## 🎓 Learn More

### Documentation
- `MLOPS_SETUP.md` - Detailed MLOps guide
- `GITHUB_DEPLOYMENT_GUIDE.md` - GitHub deployment
- `README.md` - Project overview

### Tutorials
1. **First Training Run**: `python train.py --epochs 5`
2. **Track with DVC**: `dvc add checkpoints/best_model.pth`
3. **Monitor Model**: Check `src/monitoring/dashboard.py`
4. **Deploy**: `docker-compose up -d`

---

## ⚡ Pro Tips

1. **Fast Iteration**: Use `--epochs 5` for quick tests
2. **Save Memory**: Set `--num-workers 0` on Windows
3. **GPU Training**: Ensure CUDA is installed for faster training
4. **Experiment Tracking**: Check MLflow after each run
5. **Code Quality**: Run `black src/` before committing

---

## 🆘 Common Issues

### Issue: "DVC not found"
**Solution**: Run `setup_mlops.bat` or `pip install dvc`

### Issue: "Docker containers won't start"
**Solution**: 
```cmd
docker-compose down -v
docker-compose up --build
```

### Issue: "Model not found"
**Solution**: Train a model first: `python train.py --epochs 5`

### Issue: "Import errors"
**Solution**: Activate virtual environment: `venv\Scripts\activate.bat`

---

**Ready to go!** Start with `docker-compose up -d` and visit http://localhost:8501 🚀
