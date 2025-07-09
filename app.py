import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import itertools

# Set page config
st.set_page_config(
    page_title="Satellite Image Classifier",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 0.5rem;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #0d47a1;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'history' not in st.session_state:
    st.session_state.history = None
if 'test_data' not in st.session_state:
    st.session_state.test_data = None

# Class names
CLASS_NAMES = ['Cloudy', 'Desert', 'Green_Area', 'Water']
CLASS_COLORS = ['#87CEEB', '#DEB887', '#228B22', '#4169E1']

def load_and_preprocess_image(img_path, target_size=(255, 255)):
    """Load and preprocess image for prediction"""
    img = Image.open(img_path)
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def plot_confusion_matrix(cm, classes):
    """Plot confusion matrix using matplotlib"""
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix", fontsize=16, fontweight='bold')
    plt.colorbar(im, ax=ax)
    
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight='bold')
    
    plt.tight_layout()
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    return fig

def plot_training_metrics(history):
    """Plot training and validation metrics"""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training & Validation Loss', 'Training & Validation Accuracy'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    epochs = range(1, len(history['loss']) + 1)
    
    # Loss plot
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['loss'], name='Training Loss', 
                  line=dict(color='#1f77b4', width=3)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['val_loss'], name='Validation Loss',
                  line=dict(color='#ff7f0e', width=3)),
        row=1, col=1
    )
    
    # Accuracy plot
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['accuracy'], name='Training Accuracy',
                  line=dict(color='#2ca02c', width=3)),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(x=list(epochs), y=history['val_accuracy'], name='Validation Accuracy',
                  line=dict(color='#d62728', width=3)),
        row=1, col=2
    )
    
    fig.update_xaxes(title_text="Epochs", row=1, col=1)
    fig.update_xaxes(title_text="Epochs", row=1, col=2)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=2)
    
    fig.update_layout(height=500, showlegend=True, template="plotly_white")
    return fig

def main():
    # Main header
    st.markdown('<h1 class="main-header">🛰️ Satellite Image Classification</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-powered classification of satellite images into Cloudy, Desert, Green Area, and Water categories</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🔧 Navigation")
        page = st.selectbox(
            "Choose a page:",
            ["🏠 Home", "📊 Dataset Overview", "🤖 Model Training", "🔍 Image Prediction", "📈 Performance Analysis"]
        )
        
        st.markdown("---")
        st.markdown("### 📋 Model Information")
        st.info("**Classes**: Cloudy, Desert, Green Area, Water")
        st.info("**Input Size**: 255x255 pixels")
        st.info("**Architecture**: CNN with 3 Conv2D layers")
    
    # Home Page
    if page == "🏠 Home":
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://via.placeholder.com/400x300/1f77b4/ffffff?text=Satellite+Image+Classification", 
                    caption="AI-Powered Satellite Image Analysis")
        
        st.markdown("---")
        
        # Features
        st.markdown('<h2 class="sub-header">🌟 Features</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **🔍 Image Classification**
            - Upload satellite images for instant classification
            - Support for multiple image formats
            - Real-time predictions with confidence scores
            
            **📊 Dataset Analysis**
            - Comprehensive dataset overview
            - Sample image visualization
            - Class distribution analysis
            """)
        
        with col2:
            st.markdown("""
            **🤖 Model Performance**
            - Training and validation metrics
            - Confusion matrix visualization
            - Classification report
            
            **📈 Interactive Visualizations**
            - Training loss and accuracy curves
            - Performance metrics dashboard
            - Detailed analysis charts
            """)
        
        # Quick Stats
        st.markdown("---")
        st.markdown('<h2 class="sub-header">📊 Quick Stats</h2>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Classes", "4", help="Cloudy, Desert, Green Area, Water")
        with col2:
            st.metric("Input Size", "255×255", help="Image resolution in pixels")
        with col3:
            st.metric("Model Type", "CNN", help="Convolutional Neural Network")
        with col4:
            st.metric("Accuracy", "~95%", help="Expected model accuracy")
    
    # Dataset Overview
    elif page == "📊 Dataset Overview":
        st.markdown('<h2 class="sub-header">📊 Dataset Overview</h2>', unsafe_allow_html=True)
        
        # File uploader for dataset
        uploaded_file = st.file_uploader("Upload your dataset CSV file", type=['csv'])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"Dataset loaded successfully! {len(df)} images found.")
            
            # Dataset statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Dataset Statistics**")
                st.dataframe(df.head())
                
                # Class distribution
                class_counts = df['label'].value_counts()
                st.markdown("**Class Distribution**")
                st.bar_chart(class_counts)
            
            with col2:
                # Pie chart for class distribution
                fig_pie = px.pie(
                    values=class_counts.values,
                    names=class_counts.index,
                    title="Class Distribution",
                    color_discrete_sequence=CLASS_COLORS
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Summary metrics
                total_images = len(df)
                st.metric("Total Images", total_images)
                st.metric("Classes", len(class_counts))
                st.metric("Avg per Class", int(total_images / len(class_counts)))
        
        else:
            st.info("Please upload a CSV file with image paths and labels to view dataset overview.")
            
            # Sample dataset structure
            st.markdown("**Expected CSV Structure:**")
            sample_data = pd.DataFrame({
                'image_path': ['/path/to/image1.jpg', '/path/to/image2.jpg'],
                'label': ['Cloudy', 'Desert']
            })
            st.dataframe(sample_data)
    
    # Model Training
    elif page == "🤖 Model Training":
        st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
        
        # Model architecture display
        st.markdown("**Model Architecture:**")
        architecture_code = '''
        model = Sequential([
            Conv2D(32, (3, 3), input_shape=(255, 255, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(4, activation='softmax')
        ])
        '''
        st.code(architecture_code, language='python')
        
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Training Parameters:**")
            st.markdown("""
            - **Optimizer**: Adam
            - **Loss Function**: Categorical Crossentropy
            - **Metrics**: Accuracy
            - **Epochs**: 25
            - **Batch Size**: 32
            """)
        
        with col2:
            st.markdown("**Data Augmentation:**")
            st.markdown("""
            - **Rescaling**: 1./255
            - **Shear Range**: 0.2
            - **Zoom Range**: 0.2
            - **Horizontal Flip**: True
            - **Rotation Range**: 45°
            - **Vertical Flip**: True
            """)
        
        # Model file uploader
        st.markdown("---")
        model_file = st.file_uploader("Upload trained model (.h5 file)", type=['h5'])
        
        if model_file is not None:
            # Save uploaded model temporarily
            with open("temp_model.h5", "wb") as f:
                f.write(model_file.getvalue())
            
            try:
                st.session_state.model = load_model("temp_model.h5")
                st.success("Model loaded successfully!")
                
                # Display model summary
                st.markdown("**Model Summary:**")
                model_summary = []
                st.session_state.model.summary(print_fn=lambda x: model_summary.append(x))
                st.text('\n'.join(model_summary))
                
            except Exception as e:
                st.error(f"Error loading model: {e}")
    
    # Image Prediction
    elif page == "🔍 Image Prediction":
        st.markdown('<h2 class="sub-header">🔍 Image Prediction</h2>', unsafe_allow_html=True)
        
        if st.session_state.model is None:
            st.warning("Please load a model first in the Model Training section.")
            return
        
        # Image upload
        uploaded_image = st.file_uploader(
            "Upload a satellite image for classification",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a satellite image to classify it into one of the four categories"
        )
        
        if uploaded_image is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                # Display uploaded image
                image_pil = Image.open(uploaded_image)
                st.image(image_pil, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                # Make prediction
                if st.button("🚀 Classify Image", use_container_width=True):
                    with st.spinner("Analyzing image..."):
                        # Preprocess image
                        img_array = load_and_preprocess_image(uploaded_image)
                        
                        # Make prediction
                        prediction = st.session_state.model.predict(img_array)
                        predicted_class_idx = np.argmax(prediction[0])
                        predicted_class = CLASS_NAMES[predicted_class_idx]
                        confidence = float(prediction[0][predicted_class_idx])
                        
                        # Display results
                        st.success(f"**Prediction: {predicted_class}**")
                        st.info(f"**Confidence: {confidence:.2%}**")
                        
                        # Confidence scores for all classes
                        st.markdown("**Confidence Scores:**")
                        for i, (class_name, score) in enumerate(zip(CLASS_NAMES, prediction[0])):
                            st.progress(float(score), f"{class_name}: {score:.2%}")
                        
                        # Visualization
                        fig_conf = px.bar(
                            x=CLASS_NAMES,
                            y=prediction[0],
                            title="Prediction Confidence Scores",
                            color=prediction[0],
                            color_continuous_scale="viridis"
                        )
                        fig_conf.update_layout(
                            xaxis_title="Classes",
                            yaxis_title="Confidence Score",
                            showlegend=False
                        )
                        st.plotly_chart(fig_conf, use_container_width=True)
    
    # Performance Analysis
    elif page == "📈 Performance Analysis":
        st.markdown('<h2 class="sub-header">📈 Performance Analysis</h2>', unsafe_allow_html=True)
        
        # Upload training history
        history_file = st.file_uploader("Upload training history (JSON file)", type=['json'])
        
        if history_file is not None:
            import json
            history = json.load(history_file)
            st.session_state.history = history
            
            # Training metrics visualization
            fig_metrics = plot_training_metrics(history)
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Final metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Training Loss", f"{history['loss'][-1]:.4f}")
            with col2:
                st.metric("Final Validation Loss", f"{history['val_loss'][-1]:.4f}")
            with col3:
                st.metric("Final Training Accuracy", f"{history['accuracy'][-1]:.2%}")
            with col4:
                st.metric("Final Validation Accuracy", f"{history['val_accuracy'][-1]:.2%}")
        
        # Upload test predictions for confusion matrix
        st.markdown("---")
        st.markdown("**Confusion Matrix Analysis**")
        
        col1, col2 = st.columns(2)
        with col1:
            actual_labels = st.text_area("Actual Labels (comma-separated)", 
                                       placeholder="0,1,2,3,0,1...")
        with col2:
            predicted_labels = st.text_area("Predicted Labels (comma-separated)", 
                                           placeholder="0,1,2,3,1,1...")
        
        if st.button("Generate Confusion Matrix") and actual_labels and predicted_labels:
            try:
                actual = [int(x.strip()) for x in actual_labels.split(',')]
                predicted = [int(x.strip()) for x in predicted_labels.split(',')]
                
                if len(actual) != len(predicted):
                    st.error("Actual and predicted labels must have the same length!")
                else:
                    # Generate confusion matrix
                    cm = confusion_matrix(actual, predicted)
                    
                    # Plot confusion matrix
                    fig_cm = plot_confusion_matrix(cm, CLASS_NAMES)
                    st.pyplot(fig_cm)
                    
                    # Calculate metrics
                    accuracy = np.trace(cm) / np.sum(cm)
                    st.metric("Overall Accuracy", f"{accuracy:.2%}")
                    
                    # Per-class metrics
                    st.markdown("**Per-Class Performance:**")
                    for i, class_name in enumerate(CLASS_NAMES):
                        precision = cm[i, i] / np.sum(cm[:, i]) if np.sum(cm[:, i]) > 0 else 0
                        recall = cm[i, i] / np.sum(cm[i, :]) if np.sum(cm[i, :]) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric(f"{class_name} Precision", f"{precision:.2%}")
                        with col2:
                            st.metric(f"{class_name} Recall", f"{recall:.2%}")
                        with col3:
                            st.metric(f"{class_name} F1-Score", f"{f1:.2%}")
                        
                        st.markdown("---")
            
            except Exception as e:
                st.error(f"Error processing labels: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.9rem;">'
        'Satellite Image Classification App | Built with Streamlit and TensorFlow'
        '</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()