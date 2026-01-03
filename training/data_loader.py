"""
PHASE 2: Data Loading & Preprocessing Pipeline

This module provides data loaders for the Human Action Recognition dataset.
It handles image loading, preprocessing, augmentation, and batch generation.

Key Features:
- Image resizing and normalization
- Label encoding
- Data augmentation
- Sequence generation for LSTM
- Batch generation

Author: Deep Learning Assignment  
Date: January 2026
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


class HARDataLoader:
    """
    Human Action Recognition Data Loader
    
    This class handles all data loading and preprocessing operations including:
    - Reading CSV labels
    - Loading and preprocessing images
    - Creating sequences for temporal modeling
    - Data augmentation
    - Batch generation for training
    """
    
    def __init__(self, data_root, img_size=(224, 224), sequence_length=1, 
                 normalize=True):
        """
        Initialize the data loader
        
        Args:
            data_root (str): Root directory containing the dataset
            img_size (tuple): Target size for images (height, width)
            sequence_length (int): Number of frames in a sequence for LSTM
            normalize (bool): Whether to normalize pixel values to [0, 1]
        
        Note:
            For static images, sequence_length=1 means single-frame classification
            For temporal modeling, we can create pseudo-sequences through augmentation
        """
        self.data_root = data_root
        self.img_size = img_size
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Paths
        self.train_csv = os.path.join(data_root, "Training_set.csv")
        self.test_csv = os.path.join(data_root, "Testing_set.csv")
        self.train_dir = os.path.join(data_root, "train")
        self.test_dir = os.path.join(data_root, "test")
        
        # Label encoder
        self.label_encoder = LabelEncoder()
        self.num_classes = None
        
        print(f"📊 DataLoader Initialized:")
        print(f"   - Image Size: {img_size}")
        print(f"   - Sequence Length: {sequence_length}")
        print(f"   - Normalization: {normalize}")
    
    
    def load_and_preprocess_image(self, img_path, augment=False):
        """
        Load and preprocess a single image
        
        Args:
            img_path (str): Path to the image file
            augment (bool): Whether to apply data augmentation
        
        Returns:
            np.array: Preprocessed image
            
        Preprocessing Steps:
            1. Load image using PIL
            2. Resize to target size
            3. Convert to numpy array
            4. Normalize if enabled
            5. Apply augmentation if training
        """
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Resize
            img = img.resize(self.img_size)
            
            # Convert to array
            img_array = np.array(img, dtype=np.float32)
            
            # Normalize to [0, 1]
            if self.normalize:
                img_array = img_array / 255.0
            
            # Data augmentation (if training)
            if augment:
                img_array = self.augment_image(img_array)
            
            return img_array
            
        except Exception as e:
            print(f"❌ Error loading image {img_path}: {e}")
            # Return black image as fallback
            return np.zeros((*self.img_size, 3), dtype=np.float32)
    
    
    def augment_image(self, img_array):
        """
        Apply random augmentation to an image
        
        Args:
            img_array (np.array): Input image
        
        Returns:
            np.array: Augmented image
        
        Augmentation techniques:
            - Random horizontal flip
            - Random brightness adjustment
            - Random rotation (small angles)
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            img_array = np.fliplr(img_array)
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img_array = np.clip(img_array * factor, 0, 1)
        
        return img_array
    
    
    def load_data(self, split='train', validation_split=0.2, augment=False):
        """
        Load and prepare dataset
        
        Args:
            split (str): 'train', 'val', or 'test'
            validation_split (float): Proportion of training data for validation
            augment (bool): Apply data augmentation
        
        Returns:
            tuple: (images, labels) or (images, None) for test set
        """
        print(f"\n📂 Loading {split.upper()} data...")
        
        if split in ['train', 'val']:
            # Load training CSV
            df = pd.read_csv(self.train_csv)
            
            # Fit label encoder on all training labels
            if not hasattr(self, 'classes_'):
                self.label_encoder.fit(df['label'])
                self.classes_ = self.label_encoder.classes_
                self.num_classes = len(self.classes_)
                print(f"   🎯 Number of classes: {self.num_classes}")
            
            # Split train/val
            train_df, val_df = train_test_split(
                df, test_size=validation_split, 
                stratify=df['label'], random_state=42
            )
            
            df = train_df if split == 'train' else val_df
            
        else:  # test
            df = pd.read_csv(self.test_csv)
        
        print(f"   📊 {split.upper()} samples: {len(df)}")
        
        # Load images
        images = []
        labels = []
        
        img_dir = self.train_dir if split in ['train', 'val'] else self.test_dir
        
        for idx, row in df.iterrows():
            filename = row['filename']
            img_path = os.path.join(img_dir, filename)
            
            # Load image
            img = self.load_and_preprocess_image(img_path, augment=(augment and split=='train'))
            images.append(img)
            
            # Load label (if available)
            if 'label' in row:
                label_encoded = self.label_encoder.transform([row['label']])[0]
                labels.append(label_encoded)
            
            # Progress
            if (idx + 1) % 1000 == 0:
                print(f"   Loaded {idx + 1}/{len(df)} images...")
        
        images = np.array(images)
        
        # Convert labels to categorical if available
        if labels:
            labels = np.array(labels)
            labels = to_categorical(labels, num_classes=self.num_classes)
            print(f"   ✅ Loaded {len(images)} images with labels")
            return images, labels
        else:
            print(f"   ✅ Loaded {len(images)} test images (no labels)")
            return images, None
    
    
    def create_sequences(self, images, labels=None):
        """
        Create sequences for LSTM from static images
        
        For static images, we can create pseudo-sequences by:
        - Using the same image multiple times
        - Applying different augmentations
        - Or just use sequence_length=1 for single-frame classification
        
        Args:
            images (np.array): Array of images
            labels (np.array): Array of labels (optional)
        
        Returns:
            tuple: (sequences, labels) where sequences have shape 
                   (num_samples, sequence_length, height, width, channels)
        """
        if self.sequence_length == 1:
            # Single frame per sample
            # Reshape to add sequence dimension
            sequences = np.expand_dims(images, axis=1)
            return sequences, labels
        else:
            # Create pseudo-sequences through augmentation
            sequences = []
            for img in images:
                # Create sequence by applying different augmentations
                seq = []
                for _ in range(self.sequence_length):
                    aug_img = self.augment_image(img.copy())
                    seq.append(aug_img)
                sequences.append(seq)
            
            sequences = np.array(sequences)
            return sequences, labels
    
    
    def get_data_generators(self, batch_size=32, augment=True):
        """
        Create data generators for training
        
        Args:
            batch_size (int): Batch size for training
            augment (bool): Apply augmentation to training data
        
        Returns:
            tuple: (train_gen, val_gen, test_data)
        """
        # Load data
        X_train, y_train = self.load_data('train', augment=augment)
        X_val, y_val = self.load_data('val', augment=False)
        X_test, _ = self.load_data('test', augment=False)
        
        # Create sequences if needed
        if self.sequence_length > 1:
            print(f"\n🔄 Creating sequences (length={self.sequence_length})...")
            X_train, y_train = self.create_sequences(X_train, y_train)
            X_val, y_val = self.create_sequences(X_val, y_val)
            X_test, _ = self.create_sequences(X_test)
        else:
            # Add sequence dimension for consistency
            X_train = np.expand_dims(X_train, axis=1)
            X_val = np.expand_dims(X_val, axis=1)
            X_test = np.expand_dims(X_test, axis=1)
        
        print(f"\n✅ Data shapes:")
        print(f"   Train: {X_train.shape}, {y_train.shape}")
        print(f"   Val: {X_val.shape}, {y_val.shape}")
        print(f"   Test: {X_test.shape}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, None)
    
    
    def save_label_encoder(self, save_path='models/label_encoder.pkl'):
        """Save label encoder for later use"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"💾 Saved label encoder to {save_path}")
    
    
    def load_label_encoder(self, load_path='models/label_encoder.pkl'):
        """Load saved label encoder"""
        with open(load_path, 'rb') as f:
            self.label_encoder = pickle.load(f)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"📂 Loaded label encoder from {load_path}")


def main():
    """Test data loader"""
    print("🧪 Testing Data Loader...\n")
    
    data_root = "/Users/muhammadawais/CS-7B/Deep Learning Th/Assignment/Human Action Recognition"
    
    # Initialize loader
    loader = HARDataLoader(
        data_root=data_root,
        img_size=(224, 224),
        sequence_length=1,  # Single frame classification
        normalize=True
    )
    
    # Test loading a small batch
    print("\n🔍 Loading a small sample...")
    df = pd.read_csv(loader.train_csv).head(10)
    
    images = []
    for _, row in df.iterrows():
        img_path = os.path.join(loader.train_dir, row['filename'])
        img = loader.load_and_preprocess_image(img_path)
        images.append(img)
    
    images = np.array(images)
    print(f"✅ Loaded {len(images)} images with shape: {images.shape}")
    print(f"   Min pixel value: {images.min():.3f}")
    print(f"   Max pixel value: {images.max():.3f}")
    print(f"   Mean pixel value: {images.mean():.3f}")


if __name__ == "__main__":
    main()
