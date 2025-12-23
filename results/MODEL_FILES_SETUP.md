# 📦 Model Files Setup Guide

## 🚨 Error: Model Files Not Found

Jika Anda mendapat error:
```
Unable to open file: name = './artifacts/best_custom_pure_noaug.h5'
No such file or directory
```

**Artinya:** Folder `artifacts/` tidak ada atau tidak ter-upload ke GitHub.

---

## ✅ Solusi

### Problem: Model Files Terlalu Besar

GitHub memiliki limit:
- **Single file:** 100MB maximum
- **Push size:** 100MB per file

Model Deep Learning biasanya > 100MB, jadi perlu solusi khusus.

---

## 🎯 3 Solusi yang Bisa Dipilih

### Solusi 1: Git LFS (Recommended untuk file besar)

**Git Large File Storage** - untuk file > 100MB

#### Step 1: Install Git LFS
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# macOS
brew install git-lfs

# Windows
# Download dari: https://git-lfs.github.com/
```

#### Step 2: Setup Git LFS di Repository
```bash
cd "/path/to/MACHINE_LEARNING/results"

# Initialize Git LFS
git lfs install

# Track model files
git lfs track "artifacts/*.h5"
git lfs track "artifacts/*.pt"
git lfs track "artifacts/*.pkl"
git lfs track "artifacts/*.joblib"

# Add .gitattributes
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

#### Step 3: Push Model Files
```bash
# Add artifacts folder
git add artifacts/

# Commit
git commit -m "Add model files via Git LFS"

# Push (Git LFS akan handle file besar)
git push origin main
```

#### Step 4: Verify
```bash
# Check LFS files
git lfs ls-files

# Should show your model files
```

---

### Solusi 2: Streamlit Cloud Secrets + Download URL

Jika model terlalu besar atau tidak mau pakai Git LFS.

#### Step 1: Upload Models ke Cloud Storage

**Option A: Google Drive**
1. Upload file ke Google Drive
2. Share file (Anyone with link)
3. Get direct download link:
```
https://drive.google.com/uc?id=FILE_ID&export=download
```

**Option B: Dropbox**
1. Upload ke Dropbox
2. Get share link
3. Change `?dl=0` to `?dl=1`

**Option C: Hugging Face Hub** (Recommended)
```bash
pip install huggingface_hub

# Upload
huggingface-cli upload your-username/covid-models ./artifacts
```

#### Step 2: Modify app.py to Download Models

```python
import os
import requests
from pathlib import Path

def download_model(url, filename):
    """Download model if not exists"""
    filepath = os.path.join(ARTIFACTS_PATH, filename)
    if not os.path.exists(filepath):
        st.info(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"Downloaded {filename}")
    return filepath

# Download models on first run
MODEL_URLS = {
    "best_custom_pure_noaug.h5": "YOUR_GOOGLE_DRIVE_LINK",
    "best_custom_lora_noaug.h5": "YOUR_GOOGLE_DRIVE_LINK",
    # ... other models
}

# Create artifacts directory
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

# Download models
for filename, url in MODEL_URLS.items():
    download_model(url, filename)
```

---

### Solusi 3: Gunakan Model Lebih Kecil

Jika tidak mau kompleksitas di atas.

#### Option A: Model Compression
```python
# Quantize models to reduce size
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('model.h5')

# Convert to TensorFlow Lite (much smaller)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save compressed model
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

#### Option B: Only Essential Models
Hanya upload model terbaik (1-2 models) instead of semua 9 models.

---

## 📋 Checklist: Verify Artifacts Folder

### Local Check
```bash
cd results

# List artifacts
ls -lh artifacts/

# Should show:
# best_custom_pure_noaug.h5
# best_custom_lora_noaug.h5
# best_custom_lora_aug.h5
# best_vit_model.h5
# best_lora_densenet.h5
# hf_vit_pretrained_best.pt
# *.joblib files
# *.pkl files
```

### GitHub Check
1. Go to your repository on GitHub
2. Navigate to `results/artifacts/`
3. Verify all model files are there
4. Check file sizes

### Streamlit Cloud Check
Setelah deploy, app akan:
- Check if `artifacts/` exists ✓
- Show error message if not found ✓
- List what files are missing ✓

---

## 🎯 Recommended Approach

**Untuk file < 100MB:**
```bash
# Just git add normally
git add artifacts/
git commit -m "Add model files"
git push
```

**Untuk file > 100MB:**
```bash
# Use Git LFS
git lfs track "artifacts/*.h5"
git lfs track "artifacts/*.pt"
git add .gitattributes
git add artifacts/
git commit -m "Add large model files via Git LFS"
git push
```

**Untuk file SANGAT besar (>1GB):**
```
Use Hugging Face Hub atau download URL approach
```

---

## 🔍 Debugging

### Check Current Directory on Streamlit Cloud
Add this to app.py temporarily:
```python
import os
st.write("Current directory:", os.getcwd())
st.write("App directory:", APP_DIR)
st.write("Artifacts path:", ARTIFACTS_PATH)
st.write("Artifacts exists:", os.path.exists(ARTIFACTS_PATH))
if os.path.exists(ARTIFACTS_PATH):
    st.write("Files in artifacts:", os.listdir(ARTIFACTS_PATH))
```

### Check Git LFS Status
```bash
# Local
git lfs ls-files

# Should show tracked files
```

### Check File Sizes
```bash
# List large files in repo
find . -type f -size +50M -exec ls -lh {} \;
```

---

## 📝 .gitattributes for Git LFS

Jika pakai Git LFS, file ini akan otomatis dibuat:

```
artifacts/*.h5 filter=lfs diff=lfs merge=lfs -text
artifacts/*.pt filter=lfs diff=lfs merge=lfs -text
artifacts/*.pkl filter=lfs diff=lfs merge=lfs -text
```

---

## ⚠️ Important Notes

### GitHub Limits
- Free account: 1GB storage, 1GB bandwidth/month untuk LFS
- File must be < 2GB individual
- Total repo < 5GB recommended

### Streamlit Cloud
- No storage for large files
- Must download or use Git LFS
- Models load on every cold start

### Best Practice
1. Keep models optimized and compressed
2. Use Git LFS for files > 100MB
3. Consider cloud storage for very large models
4. Cache model loading with `@st.cache_resource`

---

## 🚀 Quick Fix Now

**If you want app to work NOW without models:**

1. Comment out model loading temporarily:
```python
# deep_models = load_deep_models()
deep_models = {}  # Empty for now
```

2. Add placeholder message:
```python
st.warning("Models will be added soon. Demo mode active.")
```

3. Deploy to test other features

4. Then add models properly with Git LFS

---

## ✅ Final Checklist

- [ ] Artifacts folder exists locally
- [ ] All model files present
- [ ] Files tracked with Git LFS (if >100MB)
- [ ] .gitattributes committed
- [ ] Pushed to GitHub
- [ ] Verified on GitHub web interface
- [ ] Redeployed on Streamlit Cloud
- [ ] App loads without errors

---

<div align="center">

## Need Help?

**File too large?** → Use Git LFS

**Don't want Git LFS?** → Use download URL

**Just testing?** → Use smaller models

**Ready for production?** → Git LFS + optimization

</div>

