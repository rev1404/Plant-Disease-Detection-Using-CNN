"""
Plant Disease Detection using Convolutional Neural Networks
FIXED VERSION - Uses synthetic data for demonstration
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import random

print("TensorFlow version:", tf.__version__)
print("🚀 Starting Plant Disease Detection Project...")

class PlantDiseaseDetector:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.class_names = [
            'Apple_Healthy', 'Apple_Scab', 'Apple_Black_Rot',
            'Corn_Healthy', 'Corn_Common_Rust', 
            'Potato_Healthy', 'Potato_Early_Blight', 'Potato_Late_Blight',
            'Tomato_Healthy', 'Tomato_Early_Blight', 'Tomato_Late_Blight'
        ]
        
    def generate_synthetic_data(self, num_samples=5000):
        """
        Generate synthetic plant leaf images for demonstration
        """
        print("📊 Generating synthetic plant disease data...")
        
        # Image dimensions
        img_height, img_width = 128, 128
        num_classes = len(self.class_names)
        
        # Create empty arrays for data
        x_data = []
        y_data = []
        
        for i in range(num_samples):
            # Create base leaf image (green background)
            img = np.zeros((img_height, img_width, 3))
            
            # Make it green (leaf color)
            img[:, :, 1] = 0.6 + 0.2 * np.random.rand()  # Green channel
            img[:, :, 0] = 0.1 + 0.1 * np.random.rand()  # Red channel
            img[:, :, 2] = 0.1 + 0.1 * np.random.rand()  # Blue channel
            
            # Add leaf shape (oval)
            center_y, center_x = img_height // 2, img_width // 2
            y, x = np.ogrid[:img_height, :img_width]
            mask = ((x - center_x) ** 2 / (center_x ** 2) + 
                   (y - center_y) ** 2 / (center_y ** 2) <= 1)
            
            # Add texture (veins)
            for _ in range(3):
                start_x = random.randint(10, img_width-10)
                start_y = random.randint(10, img_height-10)
                end_x = random.randint(10, img_width-10)
                end_y = random.randint(10, img_height-10)
                cv2.line(img, (start_x, start_y), (end_x, end_y), 
                        [0.3, 0.8, 0.3], 1)
            
            # Assign class and add disease patterns
            class_id = i % num_classes
            y_data.append(class_id)
            
            # Add disease patterns for non-healthy classes
            if 'Healthy' not in self.class_names[class_id]:
                if 'Scab' in self.class_names[class_id] or 'Rust' in self.class_names[class_id]:
                    # Add small spots
                    for _ in range(random.randint(10, 30)):
                        spot_x = random.randint(20, img_width-20)
                        spot_y = random.randint(20, img_height-20)
                        radius = random.randint(2, 8)
                        color = [0.8, 0.2, 0.1]  # Reddish-brown
                        cv2.circle(img, (spot_x, spot_y), radius, color, -1)
                
                elif 'Blight' in self.class_names[class_id]:
                    # Add larger irregular patches
                    for _ in range(random.randint(3, 8)):
                        patch_x = random.randint(30, img_width-30)
                        patch_y = random.randint(30, img_height-30)
                        width = random.randint(15, 40)
                        height = random.randint(15, 40)
                        color = [0.7, 0.3, 0.1]  # Brown
                        cv2.ellipse(img, (patch_x, patch_y), (width, height), 
                                   0, 0, 360, color, -1)
            
            # Add some noise
            noise = np.random.normal(0, 0.02, (img_height, img_width, 3))
            img = np.clip(img + noise, 0, 1)
            
            x_data.append(img)
        
        x_data = np.array(x_data)
        y_data = np.array(y_data)
        
        # Convert labels to categorical
        y_data_categorical = keras.utils.to_categorical(y_data, num_classes)
        
        # Split the data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x_data, y_data_categorical, test_size=0.2, random_state=42
        )
        
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=42
        )
        
        print(f"✅ Data generated successfully!")
        print(f"📊 Training samples: {len(self.x_train)}")
        print(f"📊 Validation samples: {len(self.x_val)}")
        print(f"📊 Test samples: {len(self.x_test)}")
        print(f"🎯 Number of classes: {num_classes}")
        
    def build_model(self):
        """
        Build a CNN model for plant disease classification
        """
        print("🧠 Building neural network model...")
        
        self.model = keras.Sequential([
            # First convolutional block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second convolutional block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third convolutional block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth convolutional block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Classifier
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        # Compile the model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built successfully!")
        self.model.summary()
        
    def train_model(self, epochs=10):
        """
        Train the plant disease detection model
        """
        print(f"🎯 Training model for {epochs} epochs...")
        
        # Callbacks for better training
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-7
            )
        ]
        
        self.history = self.model.fit(
            self.x_train, self.y_train,
            batch_size=32,
            epochs=epochs,
            validation_data=(self.x_val, self.y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        
    def evaluate_model(self):
        """
        Evaluate the model on test data
        """
        print("📈 Evaluating model...")
        
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"🎉 Test Accuracy: {test_accuracy:.4f}")
        print(f"📉 Test Loss: {test_loss:.4f}")
        
        # Make predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Print classification report
        print("\n📊 Classification Report:")
        print(classification_report(y_true_classes, y_pred_classes, 
                                  target_names=[name[:15] for name in self.class_names]))
        
        return test_accuracy, test_loss
        
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available!")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_sample_predictions(self, num_samples=12):
        """
        Plot sample predictions
        """
        print("👀 Generating sample predictions...")
        
        # Get random samples
        indices = np.random.choice(len(self.x_test), num_samples, replace=False)
        
        fig, axes = plt.subplots(3, 4, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(indices):
            # Get image and true label
            img = self.x_test[idx]
            true_class = np.argmax(self.y_test[idx])
            true_label = self.class_names[true_class]
            
            # Make prediction
            prediction = self.model.predict(np.expand_dims(img, axis=0), verbose=0)
            predicted_class = np.argmax(prediction)
            predicted_label = self.class_names[predicted_class]
            confidence = np.max(prediction)
            
            # Plot the image
            axes[i].imshow(img)
            
            # Color code based on correctness
            if true_class == predicted_class:
                color = 'green'
                status = '✓ CORRECT'
            else:
                color = 'red'
                status = '✗ WRONG'
            
            axes[i].set_title(
                f'True: {true_label}\nPred: {predicted_label}\n'
                f'Conf: {confidence:.2f}\n{status}',
                fontsize=9, color=color, fontweight='bold'
            )
            axes[i].axis('off')
            
            # Add border color
            for spine in axes[i].spines.values():
                spine.set_color(color)
                spine.set_linewidth(3)
        
        plt.suptitle('Plant Disease Detection - Sample Predictions', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrix(self):
        """
        Plot confusion matrix
        """
        print("📈 Generating confusion matrix...")
        
        # Get predictions
        y_pred = self.model.predict(self.x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(self.y_test, axis=1)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[name[:12] + '...' for name in self.class_names],
                   yticklabels=[name[:12] + '...' for name in self.class_names])
        plt.title('Confusion Matrix - Plant Disease Detection', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()
        
    def save_model(self, filepath='plant_disease_model.h5'):
        """
        Save the trained model
        """
        self.model.save(filepath)
        print(f"💾 Model saved as {filepath}")

# Import cv2 for image processing
import cv2

def main():
    """
    Main function to run the complete plant disease detection pipeline
    """
    print("=" * 70)
    print("🌿 PLANT DISEASE DETECTION USING DEEP LEARNING")
    print("=" * 70)
    
    # Create detector instance
    detector = PlantDiseaseDetector()
    
    # Step 1: Generate synthetic data
    detector.generate_synthetic_data(num_samples=3000)
    
    # Step 2: Build the model
    detector.build_model()
    
    # Step 3: Train the model
    detector.train_model(epochs=8)
    
    # Step 4: Evaluate the model
    accuracy, loss = detector.evaluate_model()
    
    # Step 5: Plot training history
    detector.plot_training_history()
    
    # Step 6: Plot sample predictions
    detector.plot_sample_predictions()
    
    # Step 7: Plot confusion matrix
    detector.plot_confusion_matrix()
    
    # Step 8: Save the model
    detector.save_model()
    
    print("\n" + "="*60)
    print("🎊 PROJECT SUMMARY:")
    print("="*60)
    print(f"📊 Final Test Accuracy: {accuracy:.2%}")
    print(f"📉 Final Test Loss: {loss:.4f}")
    print(f"🎯 Number of Classes: {len(detector.class_names)}")
    print("🌱 Classes:", ", ".join(detector.class_names))
    print("💾 Model saved as 'plant_disease_model.h5'")
    print("✅ Project completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()