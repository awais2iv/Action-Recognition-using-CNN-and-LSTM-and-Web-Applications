"""
PHASE 1: Data Understanding & Exploration

This script analyzes the Human Action Recognition dataset to understand:
- CSV structure and format
- Image-label mappings
- Class distribution
- Data quality (missing/corrupt images)

Author: Deep Learning Assignment
Date: January 2026
"""

import pandas as pd
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Dataset paths (already downloaded locally)
DATA_ROOT = "/Users/muhammadawais/CS-7B/Deep Learning Th/Assignment/Human Action Recognition"
TRAIN_CSV = os.path.join(DATA_ROOT, "Training_set.csv")
TEST_CSV = os.path.join(DATA_ROOT, "Testing_set.csv")
TRAIN_IMAGES = os.path.join(DATA_ROOT, "train")
TEST_IMAGES = os.path.join(DATA_ROOT, "test")


def explore_csv_structure():
    """
    STEP 1: Understand CSV file structure
    - Column names
    - Data types
    - Sample rows
    - Total records
    """
    print("="*80)
    print("STEP 1: Exploring CSV Files")
    print("="*80)
    
    # Load training CSV
    train_df = pd.read_csv(TRAIN_CSV)
    print(f"\n📁 Training CSV Path: {TRAIN_CSV}")
    print(f"📊 Total Training Samples: {len(train_df)}")
    print(f"\n🔍 Column Names: {train_df.columns.tolist()}")
    print(f"\n📋 Data Types:\n{train_df.dtypes}")
    print(f"\n📝 First 10 Rows:\n{train_df.head(10)}")
    
    # Load testing CSV
    test_df = pd.read_csv(TEST_CSV)
    print(f"\n📁 Testing CSV Path: {TEST_CSV}")
    print(f"📊 Total Testing Samples: {len(test_df)}")
    print(f"\n🔍 Column Names: {test_df.columns.tolist()}")
    print(f"\n📝 First 10 Rows:\n{test_df.head(10)}")
    
    return train_df, test_df


def analyze_label_distribution(train_df):
    """
    STEP 2: Analyze action class distribution
    - Unique classes
    - Class counts
    - Balance check
    """
    print("\n" + "="*80)
    print("STEP 2: Analyzing Label Distribution")
    print("="*80)
    
    # Get unique labels
    unique_labels = train_df['label'].unique()
    print(f"\n🎯 Total Action Classes: {len(unique_labels)}")
    print(f"\n📌 All Action Classes:\n{sorted(unique_labels)}")
    
    # Count distribution
    label_counts = train_df['label'].value_counts()
    print(f"\n📊 Class Distribution:\n{label_counts}")
    
    # Check balance
    max_count = label_counts.max()
    min_count = label_counts.min()
    imbalance_ratio = max_count / min_count
    print(f"\n⚖️  Dataset Balance:")
    print(f"   - Max samples per class: {max_count}")
    print(f"   - Min samples per class: {min_count}")
    print(f"   - Imbalance Ratio: {imbalance_ratio:.2f}x")
    
    if imbalance_ratio > 2:
        print("   ⚠️  WARNING: Significant class imbalance detected!")
        print("   💡 Consider using class weights during training")
    else:
        print("   ✅ Dataset is reasonably balanced")
    
    # Visualize distribution
    plt.figure(figsize=(14, 6))
    label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Action Class Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('Action Classes', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    print("\n💾 Saved visualization: class_distribution.png")
    
    return unique_labels, label_counts


def verify_image_paths(train_df, test_df):
    """
    STEP 3: Verify image file existence
    - Check if all images exist
    - Identify missing files
    - Validate image integrity
    """
    print("\n" + "="*80)
    print("STEP 3: Verifying Image Files")
    print("="*80)
    
    # Check training images
    print(f"\n📂 Training Images Folder: {TRAIN_IMAGES}")
    train_files = set(os.listdir(TRAIN_IMAGES))
    print(f"   Total files in folder: {len(train_files)}")
    
    csv_train_files = set(train_df['filename'].values)
    print(f"   Files listed in CSV: {len(csv_train_files)}")
    
    # Find mismatches
    missing_in_folder = csv_train_files - train_files
    missing_in_csv = train_files - csv_train_files
    
    if missing_in_folder:
        print(f"\n   ⚠️  Files in CSV but missing in folder: {len(missing_in_folder)}")
        print(f"   First 5 missing: {list(missing_in_folder)[:5]}")
    else:
        print("   ✅ All CSV files exist in folder")
    
    if missing_in_csv:
        print(f"\n   ℹ️  Extra files in folder not in CSV: {len(missing_in_csv)}")
    
    # Check testing images
    print(f"\n📂 Testing Images Folder: {TEST_IMAGES}")
    test_files = set(os.listdir(TEST_IMAGES))
    print(f"   Total files in folder: {len(test_files)}")
    
    csv_test_files = set(test_df['filename'].values)
    print(f"   Files listed in CSV: {len(csv_test_files)}")
    
    # Verify match
    missing_test = csv_test_files - test_files
    if missing_test:
        print(f"\n   ⚠️  Files in CSV but missing in folder: {len(missing_test)}")
    else:
        print("   ✅ All test files exist in folder")


def generate_summary_report(train_df, test_df, unique_labels):
    """
    STEP 4: Generate comprehensive summary
    """
    print("\n" + "="*80)
    print("📋 COMPREHENSIVE DATASET SUMMARY")
    print("="*80)
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║           HUMAN ACTION RECOGNITION DATASET SUMMARY             ║
╚════════════════════════════════════════════════════════════════╝

📊 Dataset Statistics:
   • Training Samples: {len(train_df):,}
   • Testing Samples: {len(test_df):,}
   • Total Samples: {len(train_df) + len(test_df):,}
   • Action Classes: {len(unique_labels)}

🎯 Action Classes:
   {', '.join(sorted(unique_labels))}

📁 Dataset Structure:
   • CSV Format: filename, label
   • Images: JPEG format
   • Labels: Available for training only
   • Storage: Flat folder structure

⚙️  Next Steps for Modeling:
   1. Image Preprocessing:
      - Resize to standard size (e.g., 224×224)
      - Normalize pixel values (0-1 range)
      - Data augmentation for robustness
   
   2. Sequence Generation:
      - For static images, consider treating as single-frame sequences
      - Alternative: Use CNN only (no LSTM) for image classification
      - OR create sequences by augmentation/transformations
   
   3. Model Architecture:
      - CNN: Extract spatial features from each image
      - LSTM: Learn temporal patterns (if sequences)
      - Dense + Softmax: {len(unique_labels)}-class classification

💡 Important Notes:
   • This is an IMAGE classification task, not video
   • LSTM can be used for sequence modeling if we create sequences
   • Alternatively, use pure CNN architecture
   • Test set has no labels (for Kaggle-style submission)

"""
    print(summary)
    
    # Save to file
    with open('dataset_summary.txt', 'w') as f:
        f.write(summary)
    print("💾 Saved summary: dataset_summary.txt")


def main():
    """Main execution function"""
    print("\n" + "🚀 "*30)
    print("PHASE 1: DATA UNDERSTANDING & EXPLORATION")
    print("🚀 "*30 + "\n")
    
    # Step 1: Explore CSV structure
    train_df, test_df = explore_csv_structure()
    
    # Step 2: Analyze label distribution
    unique_labels, label_counts = analyze_label_distribution(train_df)
    
    # Step 3: Verify image paths
    verify_image_paths(train_df, test_df)
    
    # Step 4: Generate summary
    generate_summary_report(train_df, test_df, unique_labels)
    
    print("\n" + "✅ "*30)
    print("DATA EXPLORATION COMPLETED SUCCESSFULLY!")
    print("✅ "*30 + "\n")


if __name__ == "__main__":
    main()
