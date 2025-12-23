# 🚀 Streamlit Cloud Deployment Guide

## ✅ Masalah Sudah Diperbaiki!

Error `ImportError: libGL.so.1: cannot open shared object file` sudah diselesaikan!

---

## 🔧 Solusi yang Diterapkan

### 1. **packages.txt** ✓
File baru untuk install system dependencies di Streamlit Cloud:
```
freeglut3-dev
libgtk2.0-dev
libgl1-mesa-glx
```

### 2. **requirements.txt** ✓
Ganti `opencv-python` dengan `opencv-python-headless`:
```
opencv-python-headless>=4.10.0.84  # Headless version untuk server
```

### 3. **.streamlit/config.toml** ✓
Konfigurasi optimal untuk deployment:
```toml
[theme]
primaryColor = "#007aff"
backgroundColor = "#f5f5f7"

[server]
maxUploadSize = 50
```

---

## 📁 File Structure untuk Deployment

```
results/
├── app.py                      # Main application
├── requirements.txt            # Python dependencies (UPDATED)
├── packages.txt                # System dependencies (NEW)
├── .streamlit/
│   └── config.toml            # Streamlit config (NEW)
├── artifacts/                  # Model files
│   ├── *.h5                   # Keras models
│   ├── *.pt                   # PyTorch models
│   └── *.joblib               # Sklearn models (optional)
└── README.md
```

---

## 🚀 Cara Deploy ke Streamlit Cloud

### Step 1: Persiapan Repository

1. **Push ke GitHub**
```bash
git add .
git commit -m "Fix OpenCV for Streamlit Cloud deployment"
git push origin main
```

2. **Pastikan File Ada**
   - ✅ `app.py`
   - ✅ `requirements.txt` (dengan opencv-python-headless)
   - ✅ `packages.txt` (baru dibuat)
   - ✅ `.streamlit/config.toml` (baru dibuat)
   - ✅ `artifacts/` folder dengan model files

### Step 2: Deploy di Streamlit Cloud

1. **Buka Streamlit Cloud**
   - Go to: https://share.streamlit.io/
   - Sign in dengan GitHub account

2. **New App**
   - Click "New app"
   - Select repository: `your-username/machine_learning_project`
   - Branch: `main`
   - Main file path: `results/app.py`

3. **Advanced Settings**
   - Python version: `3.11` (recommended)
   - Requirements file: Auto-detected
   - Packages file: Auto-detected

4. **Deploy**
   - Click "Deploy!"
   - Wait 5-10 minutes for first deployment

---

## ⚙️ Konfigurasi Penting

### requirements.txt (Updated)
```txt
# IMPORTANT: Use opencv-python-headless for Streamlit Cloud
opencv-python-headless>=4.10.0.84

# Deep Learning
tensorflow>=2.20.0
torch>=2.9.1
transformers>=4.57.3

# ML & Data
scikit-learn>=1.8.0
numpy>=2.0.0,<2.3.0
pandas>=2.3.3

# Others
streamlit>=1.52.2
matplotlib>=3.10.8
seaborn>=0.13.2
Pillow>=10.0.0
joblib>=1.5.3
```

### packages.txt (New)
```txt
freeglut3-dev
libgtk2.0-dev
libgl1-mesa-glx
```

Ini install system libraries yang dibutuhkan OpenCV.

---

## 🎯 Troubleshooting

### Issue 1: Still Getting OpenCV Error

**Solution:**
1. Pastikan `opencv-python-headless` di requirements.txt (bukan `opencv-python`)
2. Pastikan `packages.txt` ada di root folder `results/`
3. Redeploy app dari Streamlit Cloud dashboard

### Issue 2: Models Not Loading

**Solution:**
1. Pastikan folder `artifacts/` ada di repository
2. File size limit GitHub: 100MB per file
3. Jika models > 100MB, gunakan Git LFS:
```bash
git lfs install
git lfs track "artifacts/*.h5"
git lfs track "artifacts/*.pt"
git add .gitattributes
git commit -m "Add Git LFS for large files"
git push
```

### Issue 3: Out of Memory

**Solution:**
1. Streamlit Cloud free tier: 1GB RAM
2. Reduce loaded models (load on-demand)
3. Optimize model caching
4. Consider Streamlit Cloud Pro (16GB RAM)

### Issue 4: Slow First Load

**Normal behavior:**
- First deployment: 5-10 minutes
- Installing dependencies + downloading models
- Subsequent loads: Much faster (cached)

---

## 📊 Optimization Tips

### 1. Lazy Loading Models
```python
@st.cache_resource
def load_model_on_demand(model_name):
    # Load only when needed
    return tf.keras.models.load_model(f"artifacts/{model_name}.h5")
```

### 2. Reduce Memory Usage
```python
# Clear unused models
import gc
del unused_model
gc.collect()
```

### 3. Progress Indicators
```python
with st.spinner("Loading AI models..."):
    models = load_models()
```

---

## 🔐 Environment Variables

Jika butuh secrets (API keys, etc):

1. **Streamlit Cloud Settings**
   - Go to app settings
   - Secrets section
   - Add:
```toml
[secrets]
API_KEY = "your-key-here"
```

2. **Access in Code**
```python
api_key = st.secrets["API_KEY"]
```

---

## 📈 Monitoring

### Logs
- Click "Manage app" di Streamlit Cloud
- View logs untuk debug
- Real-time error tracking

### Resource Usage
- CPU usage
- Memory usage
- Request count

### Analytics
- Visitor count
- Geographic distribution
- Usage patterns

---

## 🎨 Custom Domain (Optional)

1. **Streamlit Cloud Pro**
   - Custom domain support
   - Remove "streamlit.io" branding

2. **Setup**
   - Add CNAME record in DNS
   - Point to Streamlit Cloud
   - Configure in dashboard

---

## ✅ Pre-Deployment Checklist

- [ ] `opencv-python-headless` in requirements.txt
- [ ] `packages.txt` created with system dependencies
- [ ] `.streamlit/config.toml` configured
- [ ] All model files in `artifacts/`
- [ ] Models < 100MB (or using Git LFS)
- [ ] Test locally first: `streamlit run app.py`
- [ ] Git repository pushed to GitHub
- [ ] No sensitive data in code (use secrets)
- [ ] README.md updated with instructions

---

## 🚀 Deployment Commands

```bash
# 1. Test locally
streamlit run app.py

# 2. Commit changes
git add .
git commit -m "Ready for Streamlit Cloud deployment"
git push origin main

# 3. Deploy on Streamlit Cloud
# Go to https://share.streamlit.io/
# Click "New app" and follow wizard
```

---

## 📝 Important Notes

### File Size Limits
- Single file: 100MB (use Git LFS for larger)
- Total repository: 1GB
- Upload limit: 200MB (configurable in config.toml)

### Resource Limits (Free Tier)
- RAM: 1GB
- CPU: Shared
- Apps: 3 public apps
- Sleeping: After 7 days inactivity

### Upgrade to Pro
- 16GB RAM
- Dedicated resources
- Custom domain
- Private apps
- Priority support

---

## 🎯 Performance Tips

### 1. Cache Everything
```python
@st.cache_resource
def load_models():
    return models

@st.cache_data
def load_data():
    return data
```

### 2. Lazy Imports
```python
# Import heavy libraries only when needed
if analysis_mode == "Deep Learning":
    import tensorflow as tf
```

### 3. Optimize Images
```python
# Resize before processing
img = cv2.resize(img, (224, 224))
```

---

## 📧 Support

### Streamlit Community
- Forum: https://discuss.streamlit.io/
- Discord: Streamlit community
- GitHub: Issues & discussions

### Documentation
- Streamlit: https://docs.streamlit.io/
- Deployment: https://docs.streamlit.io/streamlit-community-cloud

---

## 🎉 Success!

Setelah deploy, app Anda akan available di:
```
https://share.streamlit.io/username/repo-name/main/results/app.py
```

atau custom URL yang lebih pendek.

---

<div align="center">

## Ready to Deploy! 🚀

**Semua file sudah siap!**

**OpenCV error sudah fixed!**

**Just push to GitHub and deploy!**

</div>

---

## 📋 Quick Reference

### requirements.txt Key Changes
```diff
- opencv-python>=4.10.0.84
+ opencv-python-headless>=4.10.0.84
```

### New Files Created
- ✅ `packages.txt` - System dependencies
- ✅ `.streamlit/config.toml` - App configuration

### Deployment URL Format
```
https://<username>-<repo>-<branch>-app-<hash>.streamlit.app
```

### Expected Deploy Time
- First time: 5-10 minutes
- Updates: 2-5 minutes
- Wake from sleep: 10-30 seconds

---

## 🔄 Update Deployed App

```bash
# Make changes locally
# Test: streamlit run app.py

# Push to GitHub
git add .
git commit -m "Update dashboard"
git push

# Auto-deploys in 2-5 minutes
```

---

<div align="center">

**Happy Deploying! 🎉**

*Your beautiful AI dashboard is ready for the world!*

</div>

