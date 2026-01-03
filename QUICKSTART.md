# 🚀 Quick Start Guide

This guide will help you get the Human Action Recognition system up and running in minutes.

## ⚡ Prerequisites

- Python 3.8+
- pip package manager
- 5-10 GB free disk space (for dataset and models)

## 📦 Installation (5 minutes)

### Step 1: Navigate to Project

```bash
cd "/Users/muhammadawais/CS-7B/Deep Learning Th/Assignment/HAR-Action-Recognition"
```

### Step 2: Install Training Dependencies

```bash
cd training
pip install -r requirements.txt
```

**For Apple Silicon Macs (M1/M2/M3):**

```bash
pip install tensorflow-macos==2.15.0 tensorflow-metal
pip install numpy pandas matplotlib seaborn scikit-learn pillow
```

### Step 3: Install Backend Dependencies

```bash
cd ../backend
pip install -r requirements.txt
```

## 🎯 Training the Model (20-30 minutes)

### Option 1: Quick Training (Recommended for Testing)

```bash
cd ../training
python train.py
```

This will:
- ✅ Load and preprocess data
- ✅ Build CNN-LSTM model
- ✅ Train for 50 epochs (with early stopping)
- ✅ Evaluate and save model
- ✅ Generate visualizations

**Expected Output:**
```
Training samples: ~10,000
Validation samples: ~2,500
Number of classes: 15

Training Progress:
Epoch 1/50 ... val_accuracy: 0.75
Epoch 2/50 ... val_accuracy: 0.82
...

Final Validation Accuracy: 0.87
```

### Option 2: Explore Data First

```bash
python explore_data.py
```

This generates:
- `class_distribution.png` - Class balance visualization
- `dataset_summary.txt` - Detailed statistics

Then train:
```bash
python train.py
```

## 🌐 Running the Application

### Terminal 1: Start Backend API

```bash
cd backend
python app.py
```

**Expected Output:**
```
🚀 STARTING API SERVER
📦 Loading model from: ../training/models/har_cnn_lstm.h5
✅ Model loaded successfully!
✅ API Ready for predictions!

INFO: Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2: Start Frontend

```bash
cd frontend
python -m http.server 8080
```

**Expected Output:**
```
Serving HTTP on 0.0.0.0 port 8080 (http://0.0.0.0:8080/) ...
```

### Open Browser

Navigate to: **http://localhost:8080**

## 🎬 Using the System

1. **Upload Image**
   - Click "Select Image" or drag & drop
   - Supported formats: JPG, PNG, JPEG

2. **Get Prediction**
   - Click "🔮 Recognize Action"
   - Wait 1-2 seconds

3. **View Results**
   - Main prediction with confidence
   - Top 5 predictions with percentages
   - Visual confidence bars

## 🧪 Quick Test

### Test 1: Check API Health

```bash
curl http://localhost:8000/health
```

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "label_encoder_loaded": true,
  "num_classes": 15
}
```

### Test 2: Get Available Classes

```bash
curl http://localhost:8000/classes
```

**Expected Response:**
```json
{
  "num_classes": 15,
  "classes": [
    "calling", "clapping", "cycling", "dancing",
    "drinking", "eating", "fighting", "hugging",
    "laughing", "listening_to_music", "running",
    "sitting", "sleeping", "texting", "using_laptop"
  ]
}
```

### Test 3: Predict from Command Line

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/image.jpg"
```

## 📊 Expected Performance

| Metric | Value |
|--------|-------|
| Training Time (GPU) | 15-20 minutes |
| Training Time (CPU) | 1-2 hours |
| Validation Accuracy | 85-90% |
| Inference Time | 100-200ms |
| Model Size | ~80 MB |

## 🔧 Troubleshooting

### Issue 1: Module Not Found

**Error:** `ModuleNotFoundError: No module named 'tensorflow'`

**Solution:**
```bash
pip install tensorflow==2.15.0
```

### Issue 2: Model File Not Found

**Error:** `Model not found at models/har_cnn_lstm.h5`

**Solution:**
1. Ensure you've run training first:
   ```bash
   cd training
   python train.py
   ```
2. Check that `models/` directory exists

### Issue 3: API Connection Error

**Error:** `Cannot connect to API server`

**Solution:**
1. Verify API is running on port 8000
2. Check no other process is using port 8000:
   ```bash
   lsof -i :8000
   ```
3. If needed, kill conflicting process:
   ```bash
   kill -9 <PID>
   ```

### Issue 4: CUDA/GPU Not Detected

**Warning:** `Could not load dynamic library 'libcudart.so.11.0'`

**Solution:**
- This is just a warning, training will use CPU
- For GPU: Install CUDA Toolkit 11.x
- Or train on CPU (slower but works)

### Issue 5: Out of Memory

**Error:** `ResourceExhaustedError: OOM when allocating tensor`

**Solution:**
Reduce batch size in `train.py`:
```python
BATCH_SIZE = 16  # Instead of 32
```

## 📁 File Locations

After training, you'll have:

```
HAR-Action-Recognition/
├── training/
│   ├── models/
│   │   ├── har_cnn_lstm.h5           ✅ Trained model
│   │   └── label_encoder.pkl          ✅ Label encoder
│   ├── training_history.png           ✅ Training curves
│   ├── confusion_matrix.png           ✅ Confusion matrix
│   ├── per_class_accuracy.png         ✅ Per-class metrics
│   └── classification_report.txt      ✅ Detailed report
```

## ⏱️ Timeline

| Task | Duration |
|------|----------|
| Installation | 5 minutes |
| Data Exploration | 2 minutes |
| Model Training | 20-30 minutes |
| Start Backend | 1 minute |
| Start Frontend | 1 minute |
| Test Predictions | 2 minutes |
| **Total** | **~30-40 minutes** |

## 🎓 What You'll Learn

1. **Deep Learning Pipeline**
   - Data preprocessing
   - Model architecture design
   - Training and evaluation
   - Hyperparameter tuning

2. **API Development**
   - REST API design
   - FastAPI framework
   - Model deployment
   - Error handling

3. **Full-Stack Integration**
   - Frontend-backend communication
   - Asynchronous requests
   - User interface design
   - Real-time predictions

## 📚 Next Steps

Once everything is running:

1. **Experiment with Images**
   - Test different action classes
   - Try edge cases
   - Evaluate confidence scores

2. **Modify Architecture**
   - Change LSTM units in `model.py`
   - Add more Dense layers
   - Try different CNN backbones

3. **Improve Accuracy**
   - Adjust hyperparameters
   - Add more augmentation
   - Train for more epochs

4. **Deploy to Production**
   - Containerize with Docker
   - Deploy to cloud (AWS, GCP)
   - Add authentication
   - Implement monitoring

## ✅ Success Checklist

- [ ] All dependencies installed
- [ ] Model trained successfully
- [ ] API server running on port 8000
- [ ] Frontend accessible on port 8080
- [ ] Can upload and predict images
- [ ] Reasonable accuracy (>80%)
- [ ] All visualizations generated

## 💡 Pro Tips

1. **Save Time:** Training on GPU is 5-10x faster
2. **Monitor Progress:** Watch training logs for overfitting
3. **Test Incrementally:** Test each component before moving to next
4. **Check Health:** Use `/health` endpoint to verify API status
5. **Read Logs:** Backend logs show prediction details

## 🎉 You're Ready!

If you've completed all steps, congratulations! You now have a fully functional Human Action Recognition system.

**Access your application:**
- **Frontend:** http://localhost:8080
- **API Docs:** http://localhost:8000/docs
- **API Health:** http://localhost:8000/health

---

**Questions?** Review the main README.md for detailed explanations.

**Errors?** Check the Troubleshooting section above.

**Want to learn more?** Read the code comments - they're designed for interview prep!
