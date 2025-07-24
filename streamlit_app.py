import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import pickle

# Load models and class names
@st.cache_resource
def load_models():
    try:
        custom_model = keras.models.load_model('custom_cnn_model.h5')
        transfer_model = keras.models.load_model('transfer_learning_model.h5')
        with open('class_names.pkl', 'rb') as f:
            class_names = pickle.load(f)
        return custom_model, transfer_model, class_names
    except:
        st.error("Models not found. Please train the models first.")
        return None, None, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for prediction"""
    image = image.resize(target_size)
    image_array = np.array(image)
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    else:
        st.error("Please upload a valid RGB image.")
        return None

def main():
    st.set_page_config(
        page_title="Brain Tumor MRI Classifier",
        page_icon="🧠",
        layout="wide"
    )

    st.title("🧠 Brain Tumor MRI Image Classification")
    st.markdown("Upload an MRI image to classify the type of brain tumor")

    # Load models
    custom_model, transfer_model, class_names = load_models()
    
    if custom_model is None:
        st.stop()

    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["Transfer Learning (MobileNetV2)", "Custom CNN"]
    )
    selected_model = transfer_model if model_choice.startswith("Transfer") else custom_model

    # Columns for layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a brain MRI image for tumor classification"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Image", use_column_width=True)

            processed_image = preprocess_image(image)

            if processed_image is not None:
                with st.spinner("Analyzing image..."):
                    predictions = selected_model.predict(processed_image)
                    predicted_class_idx = np.argmax(predictions[0])
                    predicted_class = class_names[predicted_class_idx]
                    confidence = predictions[0][predicted_class_idx] * 100

                # Display results in second column
                with col2:
                    st.header("Prediction Results")
                    st.success(f"**Predicted Tumor Type:** {predicted_class}")
                    st.info(f"**Confidence:** {confidence:.2f}%")

                    # Show all predictions
                    st.subheader("All Class Probabilities:")
                    for i, class_name in enumerate(class_names):
                        prob = predictions[0][i] * 100
                        st.write(f"{class_name}: {prob:.2f}%")
                        st.progress(prob / 100)

                    st.subheader("Model Information")
                    st.write(f"**Model Used:** {model_choice}")
                    st.write(f"**Total Classes:** {len(class_names)}")

    st.markdown("---")
    st.markdown("Built with TensorFlow and Streamlit | Brain Tumor MRI Classification Project")

if __name__ == "__main__":
    main()
