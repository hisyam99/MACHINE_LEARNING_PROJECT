# 🧬 COVID-19 AI Detection Dashboard

<div align="center">

![Dashboard Banner](https://img.shields.io/badge/AI-Detection%20System-blue?style=for-the-badge&logo=tensorflow)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)
![Version](https://img.shields.io/badge/Version-2.0-purple?style=for-the-badge)

**A stunning, fully-animated AI-powered dashboard for COVID-19 detection from chest X-rays**

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [Demo](#-demo)

</div>

---

## ✨ Highlights

This is not just another dashboard - it's a **masterpiece of modern web design** combined with cutting-edge AI:

### 🎨 Visual Excellence
- **Glassmorphism UI** with frosted glass effects
- **Animated gradients** that shift and flow
- **Smooth transitions** on every interaction
- **Particle effects** and floating animations
- **Responsive design** optimized for all devices
- **Dark theme** with vibrant accent colors

### 🤖 AI Power
- **9 Advanced Models** running in parallel
- **Deep Learning**: Custom CNN, Vision Transformers, DenseNet121
- **Classic ML**: SVM, Random Forest, K-NN
- **LoRA Fine-tuning** for efficient adaptation
- **Real-time inference** with confidence scores

### 📊 Advanced Analytics
- Multi-model consensus analysis
- Interactive probability visualizations
- Training metrics and learning curves
- Detailed classification reports
- Statistical summaries and insights

---

## 🚀 Quick Start

### Installation

1. **Clone and Navigate**
   ```bash
   cd results/
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch Dashboard**
   
   **Linux/Mac:**
   ```bash
   ./run_dashboard.sh
   ```
   
   **Windows:**
   ```bash
   run_dashboard.bat
   ```
   
   **Or manually:**
   ```bash
   streamlit run app.py
   ```

4. **Open Browser**
   Navigate to `http://localhost:8501`

---

## 🎯 Features

### 1. Detection & Analysis Mode 🖼️

Upload a chest X-ray and get instant AI-powered analysis:

- **Single Model Analysis** - Test with specific model
- **Category-wise Testing** - Run all classic ML or all deep learning models
- **Comprehensive Analysis** - Execute all 9 models simultaneously

**Output includes:**
- 🏆 Best consensus prediction
- 📊 Probability distribution chart
- 📈 Statistical metrics (avg confidence, std dev, consensus)
- 📋 Detailed results table with color coding

### 2. Training Metrics Mode 📈

Visualize model training and evaluation:

- **Training Curves** - Loss and accuracy plots with beautiful styling
- **Evaluation Reports** - Precision, recall, F1-score metrics
- **Summary Statistics** - Best epochs, final accuracies
- **Interactive Selection** - Compare different models

---

## 🎨 Design Features

### Animations & Effects

| Feature | Description |
|---------|-------------|
| **Background** | Animated gradient with floating bubbles |
| **Cards** | Float animation with shimmer effects |
| **Buttons** | Pulse animation with ripple on hover |
| **Progress** | Gradient shine animation |
| **Predictions** | Pop-in animation with category-specific pulse |
| **Images** | Zoom-in animation with hover scale |
| **Transitions** | Smooth cubic-bezier curves throughout |

### Color Scheme

```css
COVID-19:   #eb3349 → #f45c43 (Red gradient)
Non-COVID:  #f093fb → #f5576c (Pink gradient)  
Normal:     #4facfe → #00f2fe (Blue gradient)
Primary:    #667eea → #764ba2 (Purple gradient)
```

### Responsive Breakpoints

- **Desktop** (1920px+): Full layout with maximum spacing
- **Laptop** (1024px - 1919px): Optimized for medium screens
- **Tablet** (768px - 1023px): Adjusted font sizes and card layouts
- **Mobile** (< 768px): Single-column, touch-optimized interface

---

## 🤖 AI Models

### Deep Learning (6 Models)

1. **Pure Custom CNN**
   - Baseline architecture
   - 3 conv blocks + dense layers
   - No augmentation

2. **Custom CNN + LoRA (No Aug)**
   - LoRA adaptation layers
   - Efficient fine-tuning
   - Rank-8 decomposition

3. **Custom CNN + LoRA (Augmented)**
   - Data augmentation pipeline
   - LoRA fine-tuning
   - Best generalization

4. **Vision Transformer (Manual)**
   - Patch-based encoding
   - Self-attention mechanism
   - Custom implementation

5. **HuggingFace ViT (Pretrained)**
   - google/vit-base-patch16-224
   - Transfer learning
   - PyTorch backend

6. **DenseNet121 + LoRA**
   - Dense connections
   - ImageNet pretrained
   - LoRA adaptation

### Classic ML (3 Models)

7. **SVM (RBF Kernel)**
   - HOG features
   - Radial basis function
   - Optimized hyperparameters

8. **Random Forest**
   - Ensemble decision trees
   - Feature importance analysis
   - 100+ estimators

9. **K-Nearest Neighbors**
   - Distance-based classification
   - Optimized K value
   - Fast inference

---

## 📊 Technical Details

### Image Preprocessing Pipeline

```python
1. Load image → Grayscale
2. Resize to 224×224
3. CLAHE enhancement
4. Heuristic lung cropping
5. Normalization [0, 1]
6. Channel replication (for RGB models)
```

### Model Input Formats

- **CNN/ViT Keras**: `(1, 224, 224, 1)` grayscale
- **DenseNet/HF ViT**: `(1, 224, 224, 3)` RGB
- **Classic ML**: HOG features `(1, 1764)`

### Performance Optimization

- `@st.cache_resource` for model loading
- Batch processing for multiple models
- Efficient memory management
- Progressive rendering

---

## 📁 File Structure

```
results/
├── app.py                          # 🎨 Main Streamlit dashboard
├── run_dashboard.sh                # 🚀 Linux/Mac launcher
├── run_dashboard.bat               # 🪟 Windows launcher
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📖 This file
├── DASHBOARD_GUIDE.md              # 📚 Detailed user guide
├── streamlit_logs.txt              # 📝 Application logs
│
├── artifacts/                       # 🤖 AI Models & Artifacts
│   ├── best_custom_pure_noaug.h5
│   ├── best_custom_lora_noaug.h5
│   ├── best_custom_lora_aug.h5
│   ├── best_vit_model.h5
│   ├── best_lora_densenet.h5
│   ├── hf_vit_pretrained_best.pt
│   ├── feature_scaler_classic.joblib
│   ├── feature_selector_classic.joblib
│   ├── history_*.pkl               # Training histories
│   └── classic_models/
│       ├── svm_rbf.joblib
│       ├── random_forest.joblib
│       └── knn.joblib
│
└── __results___files/              # 📊 Generated visualizations
```

---

## 🎓 Usage Guide

### Step-by-Step

1. **Launch the dashboard** using one of the methods above
2. **Choose a mode** from the sidebar:
   - 🖼️ Detection & Analysis
   - 📈 Training Metrics
3. **Upload an X-ray** (JPG, PNG, or JPEG)
4. **Select execution mode**:
   - Single model for quick test
   - Category-wise for comparison
   - Run everything for comprehensive analysis
5. **View results** with beautiful visualizations
6. **Analyze metrics** including confidence and consensus

### Best Practices

✅ **DO:**
- Use high-quality chest X-rays
- Run multiple models for reliability
- Check consensus count
- Consider all probability scores

❌ **DON'T:**
- Use as sole diagnostic tool
- Ignore medical professional advice
- Upload non-chest X-rays
- Rely on low-confidence predictions

---

## 🛠️ Customization

### Theme Configuration

Edit in `run_dashboard.sh` or `run_dashboard.bat`:

```bash
--theme.primaryColor "#667eea"
--theme.backgroundColor "#1e1e1e"
--theme.secondaryBackgroundColor "#2d2d2d"
--theme.textColor "#ffffff"
```

### Model Configuration

Edit `ARTIFACTS_PATH` in `app.py`:

```python
ARTIFACTS_PATH = "./artifacts"  # Change to your path
```

### Custom Classes

Modify `CLASSES` dictionary:

```python
CLASSES = {0: "COVID-19", 1: "Non-COVID", 2: "Normal"}
```

---

## 📈 Performance

### Model Loading
- **First launch**: ~10-30 seconds (depends on models)
- **Subsequent loads**: Cached (instant)

### Inference Time
- **Single model**: 0.1 - 2 seconds
- **All 9 models**: 5 - 15 seconds
- **GPU acceleration**: 2-5x faster

### Resource Requirements
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: ~500MB for models
- **GPU**: Optional but recommended

---

## 🐛 Troubleshooting

### Issue: Models not loading
**Solution:** Ensure `artifacts/` folder contains all model files

### Issue: Slow performance  
**Solution:** 
- Close other applications
- Use single model mode
- Enable GPU if available

### Issue: Animations laggy
**Solution:**
- Refresh browser (Ctrl+R)
- Clear browser cache
- Disable browser extensions

### Issue: Import errors
**Solution:**
```bash
pip install -r requirements.txt --upgrade
```

---

## 🌟 Screenshots

### Main Dashboard
Beautiful glassmorphism design with animated gradients

### Detection Results
Multi-model consensus with interactive visualizations

### Training Curves
Dark-themed matplotlib plots with gradient styling

---

## 🏆 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Streamlit** | Web framework |
| **TensorFlow** | Deep learning (Keras models) |
| **PyTorch** | Deep learning (HF models) |
| **Transformers** | Vision Transformers |
| **Scikit-learn** | Classic ML algorithms |
| **OpenCV** | Image preprocessing |
| **Matplotlib** | Visualizations |
| **Seaborn** | Statistical plots |

---

## 📜 License

This project is part of an academic machine learning course.

---

## 🙏 Acknowledgments

- **Dataset**: COVID-19 Radiography Database
- **Pretrained Models**: ImageNet, HuggingFace
- **Inspiration**: Modern web design trends
- **Icons**: Icons8, Emoji

---

## 📞 Contact

For questions or feedback:
- Check `DASHBOARD_GUIDE.md` for detailed documentation
- Review code comments in `app.py`
- Consult course materials

---

<div align="center">

### Made with ❤️ and lots of ☕

**Combining cutting-edge AI with world-class design**

---

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange?style=flat&logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=flat&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red?style=flat&logo=streamlit)

**© 2025 COVID-19 Detection System | Advanced Medical AI**

</div>

