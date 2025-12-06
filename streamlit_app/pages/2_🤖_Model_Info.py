"""
Model information page
"""
import streamlit as st
import requests
import plotly.graph_objects as go
from PIL import Image
import os

API_URL = "http://api:8000"

st.set_page_config(page_title="Model Info", page_icon="🤖", layout="wide")

st.title("🤖 Model Information")
st.markdown("Learn about the deep learning model powering this application")

# Get model info from API
try:
    response = requests.get(f"{API_URL}/model_info")
    if response.status_code == 200:
        model_info = response.json()
        
        # Model overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Architecture", model_info['architecture'])
        
        with col2:
            st.metric("Total Parameters", f"{model_info['total_parameters']:,}")
        
        with col3:
            st.metric("Number of Classes", model_info['num_classes'])
        
        st.markdown("---")
        
        # Detailed information
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📐 Model Architecture")
            st.markdown(f"""
            - **Base Model**: {model_info['architecture']}
            - **Input Size**: {model_info['input_size'][0]}x{model_info['input_size'][1]} pixels
            - **Total Parameters**: {model_info['total_parameters']:,}
            - **Trainable Parameters**: {model_info['trainable_parameters']:,}
            - **Device**: {model_info['device']}
            
            #### Architecture Details
            - **Backbone**: EfficientNet-B3 (pretrained on ImageNet)
            - **Classifier Head**: Custom FC layers with dropout
            - **Activation**: ReLU + Softmax
            - **Regularization**: Dropout (0.3) + Batch Normalization
            """)
        
        with col2:
            st.markdown("### 🎯 Training Details")
            st.markdown("""
            #### Dataset
            - **Name**: HAM10000 (Human Against Machine with 10000 training images)
            - **Size**: 10,015 dermatoscopic images
            - **Classes**: 7 types of skin lesions
            - **Source**: International Skin Imaging Collaboration (ISIC)
            
            #### Training Configuration
            - **Optimizer**: AdamW
            - **Learning Rate**: 1e-4 with Cosine Annealing
            - **Batch Size**: 32
            - **Epochs**: 50 (with early stopping)
            - **Loss Function**: CrossEntropyLoss with class weights
            - **Data Augmentation**: Albumentations library
            """)
        
except Exception as e:
    st.error(f"Failed to fetch model info: {str(e)}")

st.markdown("---")

# Performance metrics
st.markdown("### 📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    #### Key Metrics
    - **Validation Accuracy**: ~85%
    - **Test Accuracy**: ~83%
    - **F1-Score (Macro)**: ~0.78
    - **Precision (Macro)**: ~0.80
    - **Recall (Macro)**: ~0.77
    
    *Note: Actual metrics may vary based on the specific training run*
    """)

with col2:
    st.markdown("""
    #### Class Balance
    The HAM10000 dataset is imbalanced:
    - **nv** (Nevi): ~67% of samples
    - **mel** (Melanoma): ~11%
    - **bkl**: ~11%
    - **bcc**: ~5%
    - **akiec**: ~3%
    - **vasc**: ~1%
    - **df**: ~1%
    
    Class weights are used during training to handle imbalance.
    """)

st.markdown("---")

# Model capabilities
st.markdown("### ✨ Model Capabilities")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### ✅ Strengths
    - High accuracy on common lesions (nv, mel, bkl)
    - Robust to various image qualities
    - Fast inference (~100ms)
    - Pretrained on ImageNet
    """)

with col2:
    st.markdown("""
    #### ⚠️ Limitations
    - Lower accuracy on rare classes (df, vasc)
    - Sensitive to image quality
    - May struggle with unusual presentations
    - Not a replacement for medical diagnosis
    """)

with col3:
    st.markdown("""
    #### 🔄 Improvements
    - Data augmentation for rare classes
    - Ensemble methods
    - External validation datasets
    - Continuous retraining
    """)

st.markdown("---")

# EfficientNet architecture
st.markdown("### 🏗️ EfficientNet Architecture")

st.markdown("""
EfficientNet is a family of convolutional neural networks that achieve state-of-the-art accuracy 
with fewer parameters through compound scaling:

1. **Compound Scaling**: Uniformly scales network width, depth, and resolution
2. **Mobile Inverted Bottleneck (MBConv)**: Efficient building blocks
3. **Squeeze-and-Excitation**: Attention mechanism for feature recalibration
4. **Swish Activation**: Self-gated activation function

**Why EfficientNet-B3?**
- Balanced trade-off between accuracy and computational cost
- 12M parameters (manageable for deployment)
- Superior performance on medical imaging tasks
- Pretrained weights provide excellent feature extraction
""")

# Technology stack
st.markdown("---")
st.markdown("### 🛠️ Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### Deep Learning
    - PyTorch 2.6
    - torchvision
    - EfficientNet-B3
    - Albumentations
    """)

with col2:
    st.markdown("""
    #### Backend & API
    - FastAPI
    - Uvicorn
    - Pydantic
    - Python 3.10+
    """)

with col3:
    st.markdown("""
    #### MLOps
    - MLflow
    - DVC
    - GitHub Actions
    - Docker
    """)

# Footer
st.markdown("---")
st.info("""
💡 **Want to learn more?** 
Check out the [EfficientNet paper](https://arxiv.org/abs/1905.11946) and the 
[HAM10000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
""")
