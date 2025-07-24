# Brain Tumor MRI Image Classification Project
# Complete implementation with custom CNN and transfer learning

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import cv2
from pathlib import Path
import streamlit as st
from PIL import Image
import pickle

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class BrainTumorClassifier:
    def __init__(self, img_size=(224, 224), batch_size=32):
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = None
        self.class_names = None
        self.custom_model = None
        self.transfer_model = None
        self.history_custom = None
        self.history_transfer = None
        
    def setup_data_generators(self, train_dir, validation_dir):
        """Setup data generators with augmentation"""
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            validation_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        self.num_classes = self.train_generator.num_classes
        self.class_names = list(self.train_generator.class_indices.keys())
        
        print(f"Found {self.train_generator.samples} training images")
        print(f"Found {self.val_generator.samples} validation images")
        print(f"Classes: {self.class_names}")
        
    def build_custom_cnn(self):
        """Build custom CNN model"""
        model = keras.Sequential([
            # First Conv Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(*self.img_size, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Conv Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Conv Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Conv Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.custom_model = model
        return model
        
    def build_transfer_model(self, base_model_name='MobileNetV2'):
        """Build transfer learning model"""
        if base_model_name == 'MobileNetV2':
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom top layers
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.transfer_model = model
        return model
        
    def train_model(self, model, model_name, epochs=10):
        """Train a model with callbacks"""
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ModelCheckpoint(
                f'{model_name}_best.h5',
                monitor='val_accuracy',
                save_best_only=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        history = model.fit(
            self.train_generator,
            epochs=epochs,
            validation_data=self.val_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
        
    def plot_training_history(self, history, model_name):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title(f'{model_name} - Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title(f'{model_name} - Loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def evaluate_model(self, model, model_name):
        """Evaluate model performance"""
        # Get predictions
        predictions = model.predict(self.val_generator)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.val_generator.classes
        
        # Classification report
        print(f"\n{model_name} - Classification Report:")
        print(classification_report(y_true, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate accuracy
        accuracy = np.sum(y_pred == y_true) / len(y_true)
        print(f"{model_name} Accuracy: {accuracy:.4f}")
        
        return accuracy, cm
        
    def compare_models(self):
        """Compare both models"""
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        
        # Evaluate both models
        custom_acc, _ = self.evaluate_model(self.custom_model, "Custom CNN")
        transfer_acc, _ = self.evaluate_model(self.transfer_model, "Transfer Learning")
        
        # Summary comparison
        comparison_data = {
            'Model': ['Custom CNN', 'Transfer Learning (MobileNetV2)'],
            'Validation Accuracy': [custom_acc, transfer_acc],
            'Parameters': [self.custom_model.count_params(), self.transfer_model.count_params()]
        }
        
        print(f"\nFinal Comparison:")
        print(f"Custom CNN Accuracy: {custom_acc:.4f}")
        print(f"Transfer Learning Accuracy: {transfer_acc:.4f}")
        print(f"Best Model: {'Transfer Learning' if transfer_acc > custom_acc else 'Custom CNN'}")
        
        return comparison_data
        
    def save_models(self):
        """Save trained models"""
        if self.custom_model:
            self.custom_model.save('custom_cnn_model.h5')
            print("Custom CNN model saved as 'custom_cnn_model.h5'")
            
        if self.transfer_model:
            self.transfer_model.save('transfer_learning_model.h5')
            print("Transfer learning model saved as 'transfer_learning_model.h5'")
            
       
        with open('class_names.pkl', 'wb') as f:
            pickle.dump(self.class_names, f)
        print("Class names saved as 'class_names.pkl'")

def create_streamlit_app():
    """Create Streamlit web application"""
    streamlit_code = '''
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
    
    # Sidebar
    st.sidebar.header("Model Selection")
    model_choice = st.sidebar.selectbox(
        "Choose Model:",
        ["Transfer Learning (MobileNetV2)", "Custom CNN"]
    )
    
    selected_model = transfer_model if model_choice.startswith("Transfer") else custom_model
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Upload MRI Image")
        uploaded_file = st.file_uploader(
            "Choose an MRI image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a brain MRI image for tumor classification"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Image", use_column_width=True)
            
            # Preprocess and predict
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
                        st.progress(prob/100)
                        st.write(f"{class_name}: {prob:.2f}%")
                        
                    # Model info
                    st.subheader("Model Information")
                    st.write(f"**Model Used:** {model_choice}")
                    st.write(f"**Total Classes:** {len(class_names)}")
    
    # Footer
    st.markdown("---")
    st.markdown("Built with TensorFlow and Streamlit | Brain Tumor MRI Classification Project")

if __name__ == "__main__":
    main()
'''
    
    with open('streamlit_app.py', 'w') as f:
        f.write(streamlit_code)
    print("Streamlit app saved as 'streamlit_app.py'")


def main():
    print("🧠 Brain Tumor MRI Image Classification Project")
    print("=" * 50)
   
    classifier = BrainTumorClassifier()
    
    
    train_dir = "dataset/train"  # Update with your path
    val_dir = "dataset/validation"  # Update with your path
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print("⚠️  Dataset directories not found!")
        print("Please download the dataset and organize it as:")
        print("dataset/")
        print("  ├── train/")
        print("  │   ├── class1/")
        print("  │   ├── class2/")
        print("  │   └── ...")
        print("  └── validation/")
        print("      ├── class1/")
        print("      ├── class2/")
        print("      └── ...")
        return
    
    # Setup data generators
    print("📊 Setting up data generators...")
    classifier.setup_data_generators(train_dir, val_dir)
    
    # Build models
    print("🏗️  Building models...")
    custom_model = classifier.build_custom_cnn()
    transfer_model = classifier.build_transfer_model()
    
    print("Custom CNN Model Summary:")
    custom_model.summary()
    
    print("\nTransfer Learning Model Summary:")
    transfer_model.summary()
    
    # Train models
    epochs = 10  # Adjust based on your time constraint
    
    print("🚀 Training Custom CNN...")
    classifier.history_custom = classifier.train_model(custom_model, "custom_cnn", epochs)
    
    print("🚀 Training Transfer Learning Model...")
    classifier.history_transfer = classifier.train_model(transfer_model, "transfer_learning", epochs)
    
    # Plot training history
    classifier.plot_training_history(classifier.history_custom, "Custom CNN")
    classifier.plot_training_history(classifier.history_transfer, "Transfer Learning")
    
    # Compare models
    comparison = classifier.compare_models()
    
    # Save models
    classifier.save_models()
    
    # Create Streamlit app
    create_streamlit_app()
    
    print("\n✅ Project completed successfully!")
    print("\nNext steps:")
    print("1. Run 'streamlit run streamlit_app.py' to launch the web app")
    print("2. Upload MRI images to test the classifier")
    print("3. Check the saved model files and plots")

if __name__ == "__main__":
    main()