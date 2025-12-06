"""
Prediction page for skin lesion classification
"""
import streamlit as st
import requests
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px

# API endpoint
API_URL = "http://api:8000"

# Page config
st.set_page_config(page_title="Make Prediction", page_icon="📸", layout="wide")

st.title("📸 Make Prediction")
st.markdown("Upload a skin lesion image to get an AI-powered classification")

# Sidebar info
with st.sidebar:
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Upload a clear image of the skin lesion
    2. Supported formats: JPG, JPEG, PNG
    3. Best results with well-lit, focused images
    4. Click 'Predict' to get results
    """)
    
    st.markdown("---")
    
    st.markdown("### ⚙️ Settings")
    show_probabilities = st.checkbox("Show all probabilities", value=True)
    show_chart = st.checkbox("Show probability chart", value=True)

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a skin lesion image"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Predict button
        if st.button("🎯 Predict", type="primary", use_container_width=True):
            with st.spinner("🔄 Analyzing image..."):
                try:
                    # Send request to API
                    files = {'file': uploaded_file.getvalue()}
                    response = requests.post(f"{API_URL}/predict", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Store result in session state
                        st.session_state['prediction_result'] = result
                        st.session_state['uploaded_image'] = image
                        
                        st.success("✅ Prediction complete!")
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {response.text}")
                        
                except Exception as e:
                    st.error(f"❌ Failed to connect to API: {str(e)}")

with col2:
    st.markdown("### Results")
    
    if 'prediction_result' in st.session_state:
        result = st.session_state['prediction_result']
        
        # Main prediction
        st.markdown(f"""
        <div style="background-color: #E8F5E9; padding: 2rem; border-radius: 10px; border-left: 5px solid #43A047;">
            <h2 style="color: #2E7D32; margin: 0;">Predicted Class</h2>
            <h1 style="color: #1B5E20; margin: 0.5rem 0;">{result['predicted_class'].upper()}</h1>
            <h3 style="color: #388E3C; margin: 0;">Confidence: {result['confidence_percentage']:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Class descriptions
        descriptions = {
            'akiec': ('Actinic Keratoses and Intraepithelial Carcinoma', '⚠️ Precancerous lesion - requires medical attention'),
            'bcc': ('Basal Cell Carcinoma', '🔴 Most common skin cancer - treatable when detected early'),
            'bkl': ('Benign Keratosis-like Lesions', '🟢 Benign growth - generally harmless'),
            'df': ('Dermatofibroma', '🟢 Benign fibrous tissue growth'),
            'mel': ('Melanoma', '🔴 Most dangerous skin cancer - immediate medical consultation needed'),
            'nv': ('Melanocytic Nevi', '🟢 Common moles - usually benign'),
            'vasc': ('Vascular Skin Lesions', '🟡 Blood vessel related lesions')
        }
        
        predicted_class = result['predicted_class']
        if predicted_class in descriptions:
            full_name, description = descriptions[predicted_class]
            st.markdown(f"""
            <div style="background-color: #E3F2FD; padding: 1rem; border-radius: 5px; margin-top: 1rem;">
                <strong>{full_name}</strong><br>
                {description}
            </div>
            """, unsafe_allow_html=True)
        
        # All probabilities
        if show_probabilities:
            st.markdown("---")
            st.markdown("### 📊 All Class Probabilities")
            
            for pred in result['all_predictions']:
                prob_pct = pred['percentage']
                st.markdown(f"**{pred['class'].upper()}**: {prob_pct:.2f}%")
                st.progress(prob_pct / 100)
        
        # Probability chart
        if show_chart:
            st.markdown("---")
            st.markdown("### 📈 Probability Distribution")
            
            # Prepare data
            classes = [p['class'].upper() for p in result['all_predictions']]
            probabilities = [p['percentage'] for p in result['all_predictions']]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=classes,
                    y=probabilities,
                    marker_color=['#43A047' if c == predicted_class.upper() else '#1E88E5' for c in classes],
                    text=[f"{p:.1f}%" for p in probabilities],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Prediction Probabilities by Class",
                xaxis_title="Class",
                yaxis_title="Probability (%)",
                yaxis_range=[0, 100],
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Medical disclaimer
        st.markdown("---")
        st.warning("""
        ⚠️ **Medical Disclaimer**: This prediction is for educational purposes only. 
        Always consult a qualified dermatologist for proper diagnosis and treatment.
        """)
        
    else:
        st.info("👆 Upload an image and click 'Predict' to see results")

# Additional info
st.markdown("---")
st.markdown("### 💡 Tips for Best Results")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **📷 Image Quality**
    - Well-lit images
    - Clear focus
    - Lesion centered
    """)

with col2:
    st.markdown("""
    **📏 Size & Format**
    - JPG, JPEG, or PNG
    - Any reasonable size
    - No filters applied
    """)

with col3:
    st.markdown("""
    **🎯 Accuracy**
    - Close-up shots work best
    - Avoid shadows
    - Capture full lesion
    """)
