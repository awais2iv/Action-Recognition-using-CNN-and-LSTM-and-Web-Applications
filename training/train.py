"""
PHASE 4: Model Training & Evaluation

This script trains the CNN-LSTM model and evaluates its performance.
It includes comprehensive evaluation metrics and visualizations.

Training Process:
1. Load and preprocess data
2. Build and compile model
3. Train with callbacks
4. Evaluate on validation set
5. Generate performance reports
6. Save trained model

Author: Deep Learning Assignment
Date: January 2026
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
import time

# Import custom modules
from data_loader import HARDataLoader
from model import (build_cnn_lstm_model, compile_model,
                   get_callbacks, print_model_summary)


# Configuration
DATA_ROOT = "/Users/muhammadawais/CS-7B/Deep Learning Th/Assignment/Human Action Recognition"
MODEL_SAVE_PATH = "models/har_cnn_lstm.h5"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"

# Hyperparameters
IMG_SIZE = (224, 224)
SEQUENCE_LENGTH = 1  # Single frame classification
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
USE_PRETRAINED = True
LSTM_UNITS = 128


def train_model():
    """
    Main training function

    Steps:
        1. Load data
        2. Build model
        3. Train model
        4. Save model and encoder

    Returns:
        tuple: (model, history, data_loader)
    """

    print("\n" + "🚀 "*30)
    print("STARTING MODEL TRAINING")
    print("🚀 "*30 + "\n")

    start_time = time.time()
    print(f"⏰ Training started at: {time.strftime('%H:%M:%S', time.localtime(start_time))}")

    # ==========================================
    # STEP 1: LOAD AND PREPARE DATA
    # ==========================================

    print("\n" + "="*80)
    print("📂 STEP 1: Loading and Preparing Data")
    print("="*80)

    step_start = time.time()
    print("🔄 Initializing data loader...")
    print(f"   📁 Data root: {DATA_ROOT}")
    print(f"   🖼️  Image size: {IMG_SIZE}")
    print(f"   📊 Sequence length: {SEQUENCE_LENGTH}")
    print(f"   🔄 Normalization: True")

    data_loader = HARDataLoader(
        data_root=DATA_ROOT,
        img_size=IMG_SIZE,
        sequence_length=SEQUENCE_LENGTH,
        normalize=True
    )

    print("🔄 Loading training and validation data...")
    print("   This may take a few minutes depending on dataset size...")

    # Get data
    (X_train, y_train), (X_val, y_val), (X_test, _) = data_loader.get_data_generators(
        batch_size=BATCH_SIZE,
        augment=True
    )

    print("💾 Saving label encoder...")
    data_loader.save_label_encoder(LABEL_ENCODER_PATH)

    num_classes = data_loader.num_classes
    input_shape = IMG_SIZE + (3,)  # (224, 224, 3)

    step_time = time.time() - step_start
    print(f"\n✅ Data loading completed in {step_time:.1f} seconds!")
    print(f"   📊 Training samples: {len(X_train)}")
    print(f"   📊 Validation samples: {len(X_val)}")
    print(f"   📊 Test samples: {len(X_test)}")
    print(f"   🏷️  Number of classes: {num_classes}")
    print(f"   📐 Input shape: {input_shape}")
    print(f"   💾 Label encoder saved to: {LABEL_ENCODER_PATH}")
    
    # ==========================================
    # STEP 2: BUILD MODEL
    # ==========================================

    print("\n" + "="*80)
    print("🏗️  STEP 2: Building CNN-LSTM Model")
    print("="*80)

    step_start = time.time()
    print("🔄 Building model architecture...")
    print(f"   🏗️  Model type: CNN-LSTM Hybrid")
    print(f"   🖼️  Input shape: {input_shape}")
    print(f"   🏷️  Number of classes: {num_classes}")
    print(f"   📊 Sequence length: {SEQUENCE_LENGTH}")
    print(f"   🤖 Use pretrained: {USE_PRETRAINED}")
    print(f"   🧠 LSTM units: {LSTM_UNITS}")

    model = build_cnn_lstm_model(
        input_shape=input_shape,
        num_classes=num_classes,
        sequence_length=SEQUENCE_LENGTH,
        use_pretrained=USE_PRETRAINED,
        lstm_units=LSTM_UNITS
    )

    print("🔄 Compiling model...")
    print(f"   ⚙️  Optimizer: Adam (lr={LEARNING_RATE})")
    print(f"   📉 Loss function: Categorical Crossentropy")
    print(f"   📊 Metrics: Accuracy, Top-3 Accuracy")

    model = compile_model(model, learning_rate=LEARNING_RATE)

    step_time = time.time() - step_start
    print(f"\n✅ Model building completed in {step_time:.1f} seconds!")

    print("📋 Model Summary:")
    print_model_summary(model)
    
    # ==========================================
    # STEP 3: TRAIN MODEL
    # ==========================================

    print("\n" + "="*80)
    print("🎯 STEP 3: Training Model")
    print("="*80)

    step_start = time.time()
    print("🔄 Setting up training callbacks...")

    callbacks = get_callbacks(model_save_path=MODEL_SAVE_PATH)

    print("📋 Training Configuration:")
    print(f"   📊 Total epochs: {EPOCHS}")
    print(f"   📦 Batch size: {BATCH_SIZE}")
    print(f"   ⚡ Learning rate: {LEARNING_RATE}")
    print(f"   💾 Model save path: {MODEL_SAVE_PATH}")
    print(f"   ⏸️  Early stopping patience: 10 epochs")
    print(f"   📉 Learning rate reduction: factor=0.5, patience=5")

    # Calculate training steps
    steps_per_epoch = len(X_train)
    validation_steps = len(X_val)
    print(f"   📈 Steps per epoch: {steps_per_epoch}")
    print(f"   ✅ Validation steps: {validation_steps}")

    print(f"\n🏃 Starting training at: {time.strftime('%H:%M:%S', time.localtime(time.time()))}")
    print("   This will take approximately 20-30 minutes...")
    print("   Progress will be shown below:\n")

    # Custom callback for progress logging
    class TrainingProgressCallback(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            self.epoch_start_time = time.time()
            print(f"\n🔄 Epoch {epoch+1:2d}/{EPOCHS} - Started at {time.strftime('%H:%M:%S', time.localtime(self.epoch_start_time))}")

        def on_epoch_end(self, epoch, logs=None):
            epoch_time = time.time() - self.epoch_start_time
            train_acc = logs.get('accuracy', 0) * 100
            val_acc = logs.get('val_accuracy', 0) * 100
            train_loss = logs.get('loss', 0)
            val_loss = logs.get('val_loss', 0)

            print(f"   ✅ Completed in {epoch_time:.1f}s")
            print(f"      📈 Train Acc: {train_acc:.2f}% | Loss: {train_loss:.4f}")
            print(f"      ✅ Val Acc: {val_acc:.2f}% | Loss: {val_loss:.4f}")

            # Progress bar
            progress = (epoch + 1) / EPOCHS * 100
            bar_length = 30
            filled_length = int(bar_length * (epoch + 1) // EPOCHS)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            print(f"      📊 Progress: [{bar}] {progress:.1f}%")

    # Add our custom callback
    callbacks.append(TrainingProgressCallback())

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=0  # We handle logging with our callback
    )

    step_time = time.time() - step_start
    print(f"\n✅ Training completed in {step_time:.1f} seconds ({step_time/60:.1f} minutes)!")
    print(f"   🏆 Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")
    print(f"   💾 Model saved to: {MODEL_SAVE_PATH}")

    return model, history, data_loader, (X_val, y_val)


def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation metrics

    Args:
        history: Keras History object
        save_path: Path to save the plot

    Visualizations:
        - Loss curves (training vs validation)
        - Accuracy curves (training vs validation)

    Why visualize?
        - Check for overfitting (gap between train/val)
        - Verify model is learning
        - Identify optimal stopping point
    """

    print("\n📊 Generating training history plots...")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Training history plot saved: {save_path}")

    # Print final metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    print("📈 Final Training Metrics:")
    print(f"   📉 Training Loss: {final_train_loss:.4f}")
    print(f"   📈 Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"   📉 Validation Loss: {final_val_loss:.4f}")
    print(f"   ✅ Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")

    # Check for overfitting
    loss_diff = final_train_loss - final_val_loss
    acc_diff = final_train_acc - final_val_acc

    if abs(loss_diff) > 0.2 or abs(acc_diff) > 0.1:
        print("⚠️  Warning: Possible overfitting detected!")
        print(".3f")
        print(".3f")
    else:
        print("✅ Good fit: Training and validation metrics are well-aligned")


def evaluate_model(model, X_val, y_val, data_loader):
    """
    Comprehensive model evaluation

    Args:
        model: Trained Keras model
        X_val: Validation images
        y_val: Validation labels (one-hot encoded)
        data_loader: Data loader with label encoder

    Evaluation Metrics:
        1. Accuracy
        2. Confusion Matrix
        3. Classification Report (precision, recall, F1-score)

    Why these metrics?
        - Accuracy: Overall performance
        - Confusion Matrix: See which classes are confused
        - Classification Report: Per-class performance
    """

    print("\n" + "="*80)
    print("📊 STEP 4: Model Evaluation")
    print("="*80)

    eval_start = time.time()
    print("🔮 Running predictions on validation set...")
    print(f"   📊 Validation samples: {len(X_val)}")
    print(f"   📦 Batch size: 32")

    # Make predictions
    y_pred_probs = model.predict(X_val, batch_size=32, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_val, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n✅ Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================

    print("\n📊 Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=data_loader.label_encoder.classes_,
        yticklabels=data_loader.label_encoder.classes_
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("💾 Confusion matrix saved: confusion_matrix.png")

    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================

    print("\n📋 Generating classification report...")
    report = classification_report(
        y_true, y_pred,
        target_names=data_loader.label_encoder.classes_,
        digits=4
    )

    print("📋 Classification Report:")
    print("-" * 80)
    print(report)
    print("-" * 80)

    # Save report to file
    with open('classification_report.txt', 'w') as f:
        f.write("HUMAN ACTION RECOGNITION - CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Validation Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
        f.write(report)
    print("💾 Classification report saved: classification_report.txt")

    # ==========================================
    # PER-CLASS ACCURACY
    # ==========================================

    print("\n📊 Analyzing per-class accuracy...")
    class_accuracies = []
    for i, class_name in enumerate(data_loader.label_encoder.classes_):
        class_mask = y_true == i
        if class_mask.sum() > 0:
            class_acc = (y_pred[class_mask] == y_true[class_mask]).mean()
            class_accuracies.append((class_name, class_acc))
            status = "✅" if class_acc >= 0.8 else "⚠️" if class_acc >= 0.6 else "❌"
            print(f"   {status} {class_name:20s}: {class_acc:.4f} ({class_acc*100:.2f}%)")

    # Plot per-class accuracy
    class_names, accuracies = zip(*class_accuracies)
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_names, accuracies, color='skyblue', edgecolor='black')
    plt.axhline(y=accuracy, color='red', linestyle='--', label=f'Overall Acc: {accuracy:.3f}')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Action Class', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('per_class_accuracy.png', dpi=300, bbox_inches='tight')
    print("\n💾 Per-class accuracy plot saved: per_class_accuracy.png")

    eval_time = time.time() - eval_start
    print(f"\n✅ Evaluation completed in {eval_time:.1f} seconds!")

    # Performance summary
    print("\n📊 Performance Summary:")
    print(f"   🎯 Overall Accuracy: {accuracy*100:.2f}%")
    print(f"   📈 Best Class: {max(class_accuracies, key=lambda x: x[1])[0]} ({max(class_accuracies, key=lambda x: x[1])[1]*100:.1f}%)")
    print(f"   📉 Worst Class: {min(class_accuracies, key=lambda x: x[1])[0]} ({min(class_accuracies, key=lambda x: x[1])[1]*100:.1f}%)")

    if accuracy >= 0.85:
        print("   🏆 Excellent performance! Model is ready for deployment.")
    elif accuracy >= 0.75:
        print("   ✅ Good performance! Model is acceptable for use.")
    else:
        print("   ⚠️  Model needs improvement. Consider more training or data augmentation.")


def generate_final_report(total_time=None):
    """Generate comprehensive final report"""

    if total_time is None:
        total_time = 0

    report = f"""
╔════════════════════════════════════════════════════════════════╗
║                   TRAINING COMPLETED                           ║
╚════════════════════════════════════════════════════════════════╝

✅ Model trained successfully!

⏰ Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)

📁 Generated Files:
   ├── models/har_cnn_lstm.h5          (Trained model)
   ├── models/label_encoder.pkl         (Label encoder)
   ├── training_history.png             (Training curves)
   ├── confusion_matrix.png             (Confusion matrix)
   ├── classification_report.txt        (Detailed metrics)
   └── per_class_accuracy.png           (Per-class performance)

🎯 Next Steps:
   1. Test the model using backend API
   2. Integrate with frontend for predictions
   3. Deploy for production use

💡 Model Usage:
   - Load model: keras.models.load_model('models/har_cnn_lstm.h5')
   - Load encoder: pickle.load(open('models/label_encoder.pkl', 'rb'))
   - Preprocess image: Resize to 224x224, normalize to [0,1]
   - Predict: model.predict(preprocessed_image)

📚 Model Architecture:
   - CNN: MobileNetV2 (pretrained on ImageNet)
   - Temporal: LSTM layers (if sequence_length > 1)
   - Classification: Dense layers with Softmax
   - Loss: Categorical Crossentropy
   - Optimizer: Adam

🔍 Why This Architecture?
   - MobileNetV2: Efficient, proven features, fast inference
   - LSTM: Captures temporal patterns in sequences
   - Dropout: Prevents overfitting
   - BatchNorm: Stabilizes training

"""
    print(report)


def main():
    """Main execution"""

    print("\n" + "🎯 "*30)
    print("HUMAN ACTION RECOGNITION - TRAINING PIPELINE")
    print("🎯 "*30)

    total_start_time = time.time()
    print(f"🚀 Training pipeline started at: {time.strftime('%H:%M:%S', time.localtime(total_start_time))}")
    print(f"📊 Dataset: {DATA_ROOT}")
    print(f"🤖 Model: CNN-LSTM (MobileNetV2 + LSTM)")
    print(f"📈 Classes: 15 action categories")
    print(f"🎯 Target: >85% validation accuracy")

    try:
        # Train model
        print("\n" + "🔄 "*30)
        print("STARTING TRAINING PROCESS")
        print("🔄 "*30)

        model, history, data_loader, (X_val, y_val) = train_model()

        # Plot training history
        print("\n" + "📊 "*30)
        print("GENERATING VISUALIZATIONS")
        print("📊 "*30)
        plot_training_history(history)

        # Evaluate model
        print("\n" + "🔍 "*30)
        print("EVALUATING MODEL PERFORMANCE")
        print("🔍 "*30)
        evaluate_model(model, X_val, y_val, data_loader)

        # Calculate total time
        total_time = time.time() - total_start_time

        # Update final report with timing
        print("\n" + "📋 "*30)
        print("GENERATING FINAL REPORT")
        print("📋 "*30)

        # Generate final report
        generate_final_report(total_time)

        print("\n" + "🎉 "*30)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("🎉 "*30)

        print(f"⏰ Total execution time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(f"🏁 Completed at: {time.strftime('%H:%M:%S', time.localtime(time.time()))}")

        # Success message
        final_accuracy = max(history.history['val_accuracy']) * 100
        if final_accuracy >= 85:
            print(f"🏆 EXCELLENT RESULT: {final_accuracy:.2f}% validation accuracy!")
            print("   Model is ready for deployment!")
        elif final_accuracy >= 75:
            print(f"✅ GOOD RESULT: {final_accuracy:.2f}% validation accuracy!")
            print("   Model is acceptable for use!")
        else:
            print(f"⚠️  MODERATE RESULT: {final_accuracy:.2f}% validation accuracy!")
            print("   Consider additional training or improvements!")

        print("\n" + "📁 "*30)
        print("NEXT STEPS:")
        print("   1. Start API: cd ../backend && python app.py")
        print("   2. Start Frontend: cd ../frontend && python -m http.server 8080")
        print("   3. Test System: Open http://localhost:8080")
        print("📁 "*30)

    except Exception as e:
        print(f"\n❌ ERROR: Training failed with exception: {str(e)}")
        print("🔍 Check the error message above for details.")
        print("💡 Common solutions:")
        print("   - Ensure all dependencies are installed")
        print("   - Check dataset path exists")
        print("   - Verify sufficient disk space")
        print("   - Try reducing batch size if memory issues")
        raise

    print("\n" + "✅ "*30)
    print("ALL PHASES COMPLETED SUCCESSFULLY!")
    print("✅ "*30 + "\n")


if __name__ == "__main__":
    main()
