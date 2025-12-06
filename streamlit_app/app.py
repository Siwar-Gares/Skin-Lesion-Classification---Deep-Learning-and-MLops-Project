"""
Streamlit Web Application for Skin Lesion Classification
"""
import streamlit as st
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Skin Lesion Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FFF3E0;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #FB8C00;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #E8F5E9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #43A047;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main page
def main():
    st.markdown('<div class="main-header">🔬 Skin Lesion Classification</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">AI-Powered Dermatology Assistant</div>', unsafe_allow_html=True)
    
    # Welcome section
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h3>Welcome to the Skin Lesion Classifier! 👋</h3>
            <p>This application uses deep learning (EfficientNet-B3) to classify skin lesions 
            into 7 different categories from the HAM10000 dataset.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Features
    st.markdown("---")
    st.markdown("### 🎯 Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### 📸 Image Prediction
        - Upload skin lesion images
        - Get instant AI predictions
        - View confidence scores
        - See detailed probabilities
        """)
    
    with col2:
        st.markdown("""
        #### 🤖 Model Information
        - EfficientNet-B3 architecture
        - Trained on HAM10000 dataset
        - 12M+ parameters
        - State-of-the-art accuracy
        """)
    
    with col3:
        st.markdown("""
        #### 📊 Analytics
        - Performance metrics
        - Confusion matrix
        - Class distribution
        - Training history
        """)
    
    # Disease categories
    st.markdown("---")
    st.markdown("### 🏥 Supported Lesion Types")
    
    categories = {
        "akiec": ("Actinic Keratoses and Intraepithelial Carcinoma", "⚠️ Precancerous"),
        "bcc": ("Basal Cell Carcinoma", "🔴 Malignant"),
        "bkl": ("Benign Keratosis-like Lesions", "🟢 Benign"),
        "df": ("Dermatofibroma", "🟢 Benign"),
        "mel": ("Melanoma", "🔴 Malignant"),
        "nv": ("Melanocytic Nevi", "🟢 Benign"),
        "vasc": ("Vascular Skin Lesions", "🟡 Vascular")
    }
    
    col1, col2 = st.columns(2)
    
    for idx, (code, (name, severity)) in enumerate(categories.items()):
        if idx % 2 == 0:
            col1.markdown(f"**{code.upper()}** - {name} {severity}")
        else:
            col2.markdown(f"**{code.upper()}** - {name} {severity}")
    
    # Important notice
    st.markdown("---")
    st.markdown("""
    <div class="warning-box">
        <h3>⚠️ Important Medical Disclaimer</h3>
        <p><strong>This tool is for educational and research purposes only.</strong></p>
        <ul>
            <li>NOT a substitute for professional medical diagnosis</li>
            <li>NOT intended for clinical decision making</li>
            <li>Always consult a qualified dermatologist for any skin concerns</li>
            <li>Early detection by medical professionals is crucial for skin cancer</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("---")
    st.markdown("### 🧭 Getting Started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("📸 **Step 1**\n\nGo to 'Make Prediction' page")
    
    with col2:
        st.info("📤 **Step 2**\n\nUpload a skin lesion image")
    
    with col3:
        st.info("🎯 **Step 3**\n\nGet instant AI prediction")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        Built with ❤️ using PyTorch, FastAPI, and Streamlit<br>
        <small>EfficientNet-B3 | HAM10000 Dataset | MLOps Pipeline</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
