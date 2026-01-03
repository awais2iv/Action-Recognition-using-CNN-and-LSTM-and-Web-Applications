# Models Directory

This directory will contain trained models:

- `har_cnn_lstm.h5` - Trained CNN-LSTM model (~80 MB)
- `label_encoder.pkl` - Label encoder for action classes

These files are generated after running `python train.py`

## Model Files

### har_cnn_lstm.h5
- **Type:** Keras HDF5 model
- **Size:** ~80 MB
- **Architecture:** MobileNetV2 + LSTM
- **Input:** (batch, sequence, 224, 224, 3)
- **Output:** (batch, 15) - 15 action classes

### label_encoder.pkl
- **Type:** Scikit-learn LabelEncoder (pickled)
- **Size:** ~1 KB
- **Purpose:** Maps class indices to class names

## Loading Models

```python
import pickle
from tensorflow import keras

# Load model
model = keras.models.load_model('models/har_cnn_lstm.h5')

# Load label encoder
with open('models/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Get class names
class_names = label_encoder.classes_
print(f"Classes: {class_names}")
```

## Important Notes

- **DO NOT** commit large model files to Git
- Add `*.h5` to `.gitignore` if models are too large
- For deployment, consider:
  - Model compression (quantization)
  - Cloud storage (S3, GCS)
  - Model versioning (MLflow, DVC)
