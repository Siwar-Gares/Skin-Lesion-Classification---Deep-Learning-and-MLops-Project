"""
About page
"""
import streamlit as st

st.set_page_config(page_title="About", page_icon="ℹ️", layout="wide")

st.title("ℹ️ About This Project")

# Project overview
st.markdown("""
## 🎯 Project Overview

This is a **complete end-to-end deep learning project** for skin lesion classification, 
combining state-of-the-art computer vision with modern MLOps practices.

### Key Features

✅ **Deep Learning Model**
- EfficientNet-B3 architecture
- Trained on HAM10000 dataset
- 7-class skin lesion classification
- 85%+ validation accuracy

✅ **Production-Ready API**
- FastAPI backend with Swagger documentation
- RESTful endpoints for predictions
- Batch processing support
- Health checks and monitoring

✅ **Interactive Web Interface**
- Streamlit multi-page application
- Real-time predictions
- Visualization of results
- User-friendly design

✅ **MLOps Pipeline**
- MLflow experiment tracking
- DVC for data versioning
- GitHub Actions CI/CD
- Docker containerization
- Model monitoring & drift detection
""")

st.markdown("---")

# Dataset information
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    ## 📊 HAM10000 Dataset
    
    The **Human Against Machine with 10000 training images** (HAM10000) dataset is a 
    large collection of multi-source dermatoscopic images of pigmented lesions.
    
    ### Dataset Statistics
    - **Total Images**: 10,015 dermatoscopic images
    - **Source**: International Skin Imaging Collaboration (ISIC)
    - **Resolution**: Various (resized to 224x224)
    - **Format**: JPEG images
    - **Metadata**: Patient demographics, lesion location, diagnosis
    
    ### Class Distribution
    1. **nv** (Melanocytic Nevi): 6,705 samples (67%)
    2. **mel** (Melanoma): 1,113 samples (11%)
    3. **bkl** (Benign Keratosis): 1,099 samples (11%)
    4. **bcc** (Basal Cell Carcinoma): 514 samples (5%)
    5. **akiec** (Actinic Keratoses): 327 samples (3%)
    6. **vasc** (Vascular Lesions): 142 samples (1%)
    7. **df** (Dermatofibroma): 115 samples (1%)
    """)

with col2:
    st.markdown("""
    ## 🏥 Medical Context
    
    ### What are Skin Lesions?
    
    Skin lesions are areas of skin that are abnormal compared to the surrounding skin. 
    They can be benign (non-cancerous) or malignant (cancerous).
    
    ### Importance of Early Detection
    
    - **Melanoma** is the deadliest form of skin cancer
    - Early detection dramatically improves survival rates
    - Regular skin checks by dermatologists are crucial
    - AI can assist (but not replace) medical professionals
    
    ### Clinical Workflow
    
    1. **Visual Inspection**: Dermatologist examines lesion
    2. **Dermoscopy**: Specialized imaging technique
    3. **AI Analysis**: Supplementary decision support
    4. **Biopsy**: Tissue sample for definitive diagnosis
    5. **Treatment**: Based on confirmed diagnosis
    
    ### ABCDE Rule for Melanoma
    
    - **A**symmetry: One half unlike the other
    - **B**order: Irregular, scalloped, or poorly defined
    - **C**olor: Varied from one area to another
    - **D**iameter: Larger than 6mm (pencil eraser)
    - **E**volving: Changing in size, shape, or color
    """)

st.markdown("---")

# Technical architecture
st.markdown("""
## 🏗️ System Architecture

This project follows a microservices architecture with three main components:

### 1. Training Pipeline
- Data preprocessing and augmentation
- Model training with MLflow tracking
- Hyperparameter optimization
- Checkpoint saving and versioning

### 2. API Service (Port 8000)
- FastAPI application
- Model inference endpoint
- Batch prediction support
- Health monitoring

### 3. Web Application (Port 8501)
- Streamlit interface
- Image upload and visualization
- Real-time predictions
- Results analysis

### 4. Experiment Tracking (Port 5000)
- MLflow UI
- Training metrics visualization
- Model registry
- Run comparison
""")

st.markdown("---")

# MLOps features
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ## 🔄 MLOps Features
    
    ### Data Version Control (DVC)
    - Track dataset versions
    - Remote storage integration
    - Reproducible pipelines
    - Efficient data sharing
    
    ### Continuous Integration/Deployment
    - GitHub Actions workflows
    - Automated testing
    - Docker image builds
    - Code quality checks
    
    ### Monitoring
    - Model performance tracking
    - Data drift detection
    - Prediction logging
    - Custom dashboards
    """)

with col2:
    st.markdown("""
    ## 📈 Performance Optimization
    
    ### Training Optimizations
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling
    - Early stopping
    
    ### Inference Optimizations
    - Model quantization
    - Batch processing
    - Caching strategies
    - GPU acceleration
    
    ### Deployment
    - Docker containerization
    - Multi-stage builds
    - Health checks
    - Auto-scaling ready
    """)

st.markdown("---")

# Development
st.markdown("""
## 👨‍💻 Development

### Project Structure
```
skin-lesion-classification/
├── src/
│   ├── data/           # Data loading and preprocessing
│   ├── models/         # Model architecture and training
│   ├── api/            # FastAPI application
│   ├── utils/          # Utility functions
│   └── monitoring/     # MLOps monitoring
├── streamlit_app/      # Streamlit web interface
├── checkpoints/        # Model checkpoints
├── .github/            # CI/CD workflows
├── docker-compose.yml  # Container orchestration
└── requirements.txt    # Python dependencies
```

### Key Technologies
- **Python 3.10+**: Programming language
- **PyTorch 2.6**: Deep learning framework
- **FastAPI**: Modern web framework
- **Streamlit**: Interactive dashboards
- **MLflow**: Experiment tracking
- **Docker**: Containerization
- **DVC**: Data versioning
- **GitHub Actions**: CI/CD pipeline
""")

st.markdown("---")

# Future improvements
st.markdown("""
## 🚀 Future Improvements

### Model Enhancements
- [ ] Ensemble of multiple architectures
- [ ] Attention mechanisms visualization
- [ ] Grad-CAM for explainability
- [ ] Semi-supervised learning

### Data & Training
- [ ] External validation datasets
- [ ] Active learning for rare classes
- [ ] Federated learning
- [ ] Multi-modal inputs (clinical data + images)

### Deployment & Monitoring
- [ ] A/B testing framework
- [ ] Real-time performance monitoring
- [ ] Automated retraining pipeline
- [ ] Mobile application

### Clinical Integration
- [ ] DICOM support
- [ ] HL7 FHIR integration
- [ ] Clinical decision support
- [ ] Regulatory compliance (FDA, CE)
""")

st.markdown("---")

# References
st.markdown("""
## 📚 References

### Dataset
- Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source 
  dermatoscopic images of common pigmented skin lesions. *Sci. Data* 5, 180161 (2018).
  [DOI: 10.1038/sdata.2018.161](https://doi.org/10.1038/sdata.2018.161)

### Model Architecture
- Tan, M. & Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. 
  *ICML* (2019). [arXiv:1905.11946](https://arxiv.org/abs/1905.11946)

### Related Work
- Esteva, A. et al. Dermatologist-level classification of skin cancer with deep neural networks. 
  *Nature* 542, 115–118 (2017).
- Haenssle, H. A. et al. Man against machine: diagnostic performance of a deep learning convolutional 
  neural network for dermoscopic melanoma recognition in comparison to 58 dermatologists. 
  *Ann. Oncol.* 29, 1836–1842 (2018).
""")

st.markdown("---")

# Medical disclaimer
st.error("""
## ⚠️ Medical Disclaimer

**IMPORTANT**: This application is designed for **educational and research purposes only**.

- ❌ NOT intended for medical diagnosis or clinical decision-making
- ❌ NOT a substitute for professional medical advice
- ❌ NOT validated for clinical use
- ❌ NOT FDA approved or CE marked

**Always consult qualified healthcare professionals for any medical concerns.**

Skin cancer can be deadly if not detected and treated early. If you notice any suspicious 
skin lesions, please see a board-certified dermatologist immediately.
""")

st.markdown("---")

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <h3>Built with ❤️ for advancing AI in healthcare</h3>
    <p>
        <strong>Technologies:</strong> PyTorch • FastAPI • Streamlit • Docker • MLflow • DVC<br>
        <strong>Dataset:</strong> HAM10000 (ISIC)<br>
        <strong>Model:</strong> EfficientNet-B3
    </p>
    <p style="margin-top: 1rem;">
        © 2024 Skin Lesion Classification Project<br>
        <small>Educational & Research Use Only</small>
    </p>
</div>
""", unsafe_allow_html=True)
