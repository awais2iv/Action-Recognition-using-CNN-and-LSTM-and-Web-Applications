"""
PHASE 3: CNN + LSTM Model Architecture

This module defines the CNN-LSTM hybrid model for Human Action Recognition.
The model combines:
- CNN layers for spatial feature extraction from images
- LSTM layers for temporal pattern recognition (sequential learning)
- Dense layers for final classification

Architecture Explanation:
1. TimeDistributed CNN: Applies same CNN to each frame in sequence
2. Feature Extraction: Extracts spatial features from each image
3. LSTM Processing: Learns temporal dependencies between frames
4. Classification: Dense layers with softmax for action prediction

Author: Deep Learning Assignment
Date: January 2026
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


def build_cnn_lstm_model(input_shape, num_classes, sequence_length=1, 
                         use_pretrained=True, lstm_units=128):
    """
    Build CNN-LSTM Model for Human Action Recognition
    
    Args:
        input_shape (tuple): Shape of each frame (height, width, channels)
        num_classes (int): Number of action classes
        sequence_length (int): Number of frames in sequence
        use_pretrained (bool): Use pretrained CNN (MobileNetV2)
        lstm_units (int): Number of LSTM units
    
    Returns:
        keras.Model: Compiled model ready for training
    
    Architecture Flow:
        Input → TimeDistributed(CNN) → GlobalAvgPooling → LSTM → Dense → Softmax
    """
    
    print("\n🏗️  Building CNN-LSTM Model...")
    print(f"   Input shape per frame: {input_shape}")
    print(f"   Sequence length: {sequence_length}")
    print(f"   Number of classes: {num_classes}")
    print(f"   Use pretrained CNN: {use_pretrained}")
    
    # Input: (batch_size, sequence_length, height, width, channels)
    input_layer = layers.Input(shape=(sequence_length, *input_shape))
    
    # ==========================================
    # CNN FEATURE EXTRACTOR (Spatial Features)
    # ==========================================
    
    if use_pretrained:
        # Use pretrained MobileNetV2 for efficiency
        # WHY? MobileNetV2 is lightweight and already learned useful features
        base_model = MobileNetV2(
            include_top=False,  # Remove classification head
            weights='imagenet',  # Use ImageNet pretrained weights
            input_shape=input_shape,
            pooling=None
        )
        
        # Freeze base model initially (transfer learning)
        base_model.trainable = False
        print("   ✅ Using pretrained MobileNetV2 (frozen)")
        
        # Apply CNN to each frame in the sequence
        # TimeDistributed: Applies the same layer to each timestep
        cnn_features = layers.TimeDistributed(base_model)(input_layer)
        
    else:
        # Build custom CNN from scratch
        print("   ✅ Building custom CNN")
        
        # Custom CNN architecture
        cnn = models.Sequential([
            # Block 1: Initial feature extraction
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 2: Deeper features
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 3: Complex patterns
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            
            # Block 4: High-level features
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
        ])
        
        # Apply to each frame
        cnn_features = layers.TimeDistributed(cnn)(input_layer)
    
    # ==========================================
    # FEATURE POOLING (Reduce spatial dimensions)
    # ==========================================
    
    # Global Average Pooling: Reduces (H, W, C) to (C,)
    # WHY? Reduces dimensionality while preserving important features
    pooled_features = layers.TimeDistributed(
        layers.GlobalAveragePooling2D()
    )(cnn_features)
    
    print(f"   📐 Feature shape after CNN+Pooling: (sequence, features)")
    
    # ==========================================
    # TEMPORAL MODELING (LSTM for sequences)
    # ==========================================
    
    if sequence_length > 1:
        # Use LSTM to learn temporal patterns
        # WHY? LSTM can remember long-term dependencies in sequences
        
        # First LSTM layer with return sequences
        x = layers.LSTM(
            lstm_units,
            return_sequences=True,  # Return full sequence
            dropout=0.3,  # Prevent overfitting
            recurrent_dropout=0.3
        )(pooled_features)
        
        # Second LSTM layer
        x = layers.LSTM(
            lstm_units // 2,
            return_sequences=False,  # Return only last output
            dropout=0.3,
            recurrent_dropout=0.3
        )(x)
        
        print(f"   ✅ Added LSTM layers (units: {lstm_units})")
    else:
        # For single-frame, just flatten
        # WHY? No temporal modeling needed for static images
        x = layers.Flatten()(pooled_features)
        print(f"   ℹ️  Single frame mode - skipping LSTM")
    
    # ==========================================
    # CLASSIFICATION HEAD (Dense layers)
    # ==========================================
    
    # Dense layer with dropout
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Dense layer 2
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    
    # Output layer with softmax
    # WHY? Softmax converts logits to probabilities for multi-class classification
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    
    # ==========================================
    # CREATE MODEL
    # ==========================================
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    print(f"\n✅ Model created successfully!")
    
    return model


def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer, loss, and metrics
    
    Args:
        model: Keras model to compile
        learning_rate: Learning rate for Adam optimizer
    
    Loss Function:
        - Categorical Crossentropy: For multi-class classification
        - WHY? Our labels are one-hot encoded
    
    Optimizer:
        - Adam: Adaptive learning rate optimizer
        - WHY? Efficient, requires less tuning, works well in practice
    
    Metrics:
        - Accuracy: Primary metric for classification
        - Top-3 Accuracy: Useful for difficult cases
    """
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',  # Multi-class classification
        metrics=[
            'accuracy',
            keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
    )
    
    print(f"\n⚙️  Model Compiled:")
    print(f"   Optimizer: Adam (lr={learning_rate})")
    print(f"   Loss: Categorical Crossentropy")
    print(f"   Metrics: Accuracy, Top-3 Accuracy")
    
    return model


def get_callbacks(model_save_path='models/har_cnn_lstm.h5'):
    """
    Get training callbacks
    
    Callbacks:
        1. EarlyStopping: Stop if validation loss doesn't improve
        2. ModelCheckpoint: Save best model based on validation accuracy
        3. ReduceLROnPlateau: Reduce learning rate when stuck
    
    Returns:
        list: List of callback objects
    """
    
    callbacks = [
        # Early Stopping: Prevent overfitting
        # WHY? Stop training if model stops improving
        EarlyStopping(
            monitor='val_loss',
            patience=10,  # Wait 10 epochs before stopping
            restore_best_weights=True,  # Restore best model
            verbose=1
        ),
        
        # Model Checkpoint: Save best model
        # WHY? Keep the model with best validation performance
        ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        
        # Reduce Learning Rate on Plateau
        # WHY? Help model converge when stuck
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,  # Reduce by half
            patience=5,  # Wait 5 epochs
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    print(f"\n📋 Training Callbacks:")
    print(f"   ✅ Early Stopping (patience=10)")
    print(f"   ✅ Model Checkpoint → {model_save_path}")
    print(f"   ✅ Reduce LR on Plateau (factor=0.5)")
    
    return callbacks


def print_model_summary(model):
    """Print detailed model summary"""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("="*80)
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) 
                           for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\n📊 Parameter Summary:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Non-trainable parameters: {non_trainable_params:,}")


def main():
    """Test model building"""
    print("🧪 Testing Model Architecture...\n")
    
    # Model parameters
    input_shape = (224, 224, 3)
    num_classes = 15
    sequence_length = 1  # Single frame
    
    # Build model
    model = build_cnn_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        sequence_length=sequence_length,
        use_pretrained=True,
        lstm_units=128
    )
    
    # Compile model
    model = compile_model(model, learning_rate=0.001)
    
    # Print summary
    print_model_summary(model)
    
    print("\n✅ Model testing completed successfully!")


if __name__ == "__main__":
    main()
