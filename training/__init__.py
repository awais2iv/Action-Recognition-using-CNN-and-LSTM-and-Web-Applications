"""
Human Action Recognition Training Module

This package contains all components for training the HAR model.

Modules:
    - explore_data: Data exploration and visualization
    - data_loader: Data loading and preprocessing
    - model: CNN-LSTM model architecture
    - train: Training script with evaluation

Usage:
    from training.data_loader import HARDataLoader
    from training.model import build_cnn_lstm_model
"""

__version__ = "1.0.0"
__author__ = "Muhammad Awais"

# Make imports easier
from .data_loader import HARDataLoader
from .model import build_cnn_lstm_model, compile_model, get_callbacks
