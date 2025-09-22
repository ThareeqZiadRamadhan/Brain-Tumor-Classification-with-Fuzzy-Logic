# PowerShell script to create brain_tumor_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Import fuzzy model functions
from fuzzy_model import extract_features_from_image, fuzzy_classify

def preprocess_uploaded_image(uploaded_file):
    """Convert uploaded file to opencv format"""
    # Convert to PIL Image
    image = Image.open(uploaded_file)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale for feature extraction
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    return image_array, image_gray

def create_feature_visualization(mean_val, contrast, entropy_val):
    """Create visualization of extracted features"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    features = ['Mean Intensity', 'Contrast (GLCM)', 'Entropy']
    values = [mean_val, contrast, entropy_val]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    for i, (feature, value, color) in enumerate(zip(features, values, colors)):
        axes[i].bar([feature], [value], color=color, alpha=0.7)
        axes[i].set_title(f'{feature}\nValue: {value:.3f}')
        axes[i].set_ylabel('Feature Value')
    
    plt.tight_layout()
    return fig

def create_prediction_chart(classes, scores):
    """Create bar chart for prediction scores"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax.bar(classes, scores, color=colors, alpha=0.7)
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Fuzzy Score', fontsize=12)
    ax.set_title('Brain Tumor Classification Scores', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(scores) + 0.1)
    
    # Highlight the winning class
    max_idx = np.argmax(scores)
    bars[max_idx].set_alpha(1.0)
    bars[max_idx].set_edgecolor('black')
    bars[max_idx].set_linewidth(2)
    
    plt.tight_layout()
    return fig

# Streamlit App
st.set_page_config(
    page_title="Brain Tumor Fuzzy Classifier",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 Brain Tumor Classification using Fuzzy Logic")
st.markdown("Upload an MRI image to classify brain tumor type using fuzzy logic algorithms")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **This app classifies brain tumors into:**
    - 🔴 **Glioma**: Most common primary brain tumor
    - 🟡 **Meningioma**: Tumor of brain/spinal cord membranes  
    - 🔵 **Pituitary**: Tumor in pituitary gland
    
    **Features Used:**
    - Mean intensity
    - GLCM contrast
    - Entropy
    
    **Algorithm:** Fuzzy Logic Classification
    """)
    
    st.header("📋 Instructions")
    st.markdown("""
    1. Upload an MRI brain scan image
    2. Wait for feature extraction
    3. View fuzzy classification results
    4. Analyze feature visualizations
    """)

# Create temp directory
if not os.path.exists("temp"):
    os.makedirs("temp")

# File uploader
uploaded_file = st.file_uploader(
    "📁 Choose an MRI image...",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
    help="Upload a brain MRI scan image for classification"
)

if uploaded_file is not None:
    try:
        # Process uploaded image
        original_image, gray_image = preprocess_uploaded_image(uploaded_file)
        
        # Display original image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_image, use_column_width=True)
            
        with col2:
            st.subheader("⚫ Grayscale Image")
            st.image(gray_image, use_column_width=True, channels="GRAY")
        
        st.divider()
        
        # Extract features
        with st.spinner("🔍 Extracting image features..."):
            mean_val, contrast, entropy_val = extract_features_from_image(gray_image)
        
        # Display extracted features
        st.subheader("📊 Extracted Features")
        
        feature_col1, feature_col2, feature_col3 = st.columns(3)
        
        with feature_col1:
            st.metric(
                label="Mean Intensity",
                value=f"{mean_val:.3f}",
                help="Average pixel intensity value"
            )
            
        with feature_col2:
            st.metric(
                label="GLCM Contrast", 
                value=f"{contrast:.3f}",
                help="Gray Level Co-occurrence Matrix contrast measure"
            )
            
        with feature_col3:
            st.metric(
                label="Entropy",
                value=f"{entropy_val:.3f}",
                help="Measure of randomness/disorder in image"
            )
        
        # Feature visualization
        st.subheader("📈 Feature Visualization")
        feature_fig = create_feature_visualization(mean_val, contrast, entropy_val)
        st.pyplot(feature_fig)
        
        st.divider()
        
        # Fuzzy classification
        with st.spinner("🧮 Running fuzzy classification..."):
            predicted_class, scores = fuzzy_classify(mean_val, contrast, entropy_val)
        
        # Display results
        st.subheader("🎯 Classification Results")
        
        # Main prediction
        result_col1, result_col2 = st.columns([1, 2])
        
        with result_col1:
            # Color coding for different classes
            if predicted_class == "Glioma":
                st.success(f"**Predicted Class:** {predicted_class}")
            elif predicted_class == "Meningioma":
                st.warning(f"**Predicted Class:** {predicted_class}")
            else:  # Pituitary
                st.info(f"**Predicted Class:** {predicted_class}")
            
            # Confidence score
            max_score = max(scores)
            st.metric("Confidence Score", f"{max_score:.3f}")
            
        with result_col2:
            # Prediction chart
            prediction_fig = create_prediction_chart(['Glioma', 'Meningioma', 'Pituitary'], scores)
            st.pyplot(prediction_fig)
        
        # Detailed scores
        st.subheader("📋 Detailed Scores")
        score_data = {
            "Class": ["Glioma", "Meningioma", "Pituitary"],
            "Fuzzy Score": [f"{score:.4f}" for score in scores],
            "Percentage": [f"{score/sum(scores)*100:.1f}%" for score in scores]
        }
        
        st.table(score_data)
        
        # Medical disclaimer
        st.warning("⚠️ **Medical Disclaimer**: This tool is for educational/research purposes only. Always consult qualified medical professionals for actual diagnosis.")
        
    except Exception as e:
        st.error(f"❌ Error processing image: {str(e)}")
        st.info("💡 Make sure you uploaded a valid image file and all required libraries are installed.")

else:
    # Landing page content
    st.info("👆 Upload an MRI brain scan image to begin classification")
    
    # Example workflow
    st.subheader("🔬 How It Works")
    
    workflow_col1, workflow_col2, workflow_col3 = st.columns(3)
    
    with workflow_col1:
        st.markdown("""
        **1. Feature Extraction**
        - Mean intensity calculation
        - GLCM contrast analysis
        - Entropy measurement
        """)
    
    with workflow_col2:
        st.markdown("""
        **2. Fuzzy Logic Processing**  
        - Membership function evaluation
        - Rule-based inference
        - Score calculation
        """)
    
    with workflow_col3:
        st.markdown("""
        **3. Classification**
        - Compare fuzzy scores
        - Determine tumor type
        - Generate confidence
        """)
    
    # Sample results preview
    st.subheader("📊 Sample Output")
    st.image("https://via.placeholder.com/800x300/f0f0f0/666666?text=Upload+MRI+Image+to+See+Results", 
             caption="Feature visualizations and classification results will appear here")
