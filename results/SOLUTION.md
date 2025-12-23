# 🎯 Quick Solution - Model Files Not Found

## ❌ Error
```
Unable to open file: name = './artifacts/best_custom_pure_noaug.h5'
No such file or directory
```

## ✅ Solution

Folder `artifacts/` dengan model files tidak ter-upload ke GitHub.

---

## 🚀 3 Steps to Fix

### Step 1: Setup Git LFS (untuk file besar)

**Linux/Mac:**
```bash
cd results
./setup_git_lfs.sh
```

**Windows:**
```bash
cd results
setup_git_lfs.bat
```

**Manual:**
```bash
git lfs install
git lfs track "artifacts/*.h5"
git lfs track "artifacts/*.pt"
git add .gitattributes
```

### Step 2: Add Model Files

```bash
git add artifacts/
git commit -m "Add model files via Git LFS"
```

### Step 3: Push to GitHub

```bash
git push origin main
```

Tunggu 2-3 menit, lalu Streamlit Cloud akan auto-redeploy.

---

## 📊 Verify

### Check Locally
```bash
cd results
ls -lh artifacts/

# Should show all *.h5, *.pt, *.pkl files
```

### Check GitHub
1. Go to repository on GitHub
2. Navigate to `results/artifacts/`
3. See if files are there
4. Large files will show "Stored with Git LFS" badge

### Check Streamlit Cloud
After push, wait 2-5 minutes for redeploy:
- App should load without errors
- Models should load successfully
- "Loading AI models..." spinner should complete

---

## 💡 If Files Too Large

### Option 1: Git LFS (Recommended)
```bash
# Free: 1GB storage, 1GB bandwidth/month
git lfs track "artifacts/*.h5"
```

### Option 2: Model Compression
```python
# Reduce model size by 50-70%
# Use TensorFlow Lite or quantization
```

### Option 3: Cloud Storage
```python
# Upload to Google Drive/Hugging Face
# Download on app startup
```

---

## ✅ Fixes Applied

1. ✓ Path fixed - Menggunakan `os.path.abspath(__file__)`
2. ✓ Warning fixed - Radio label tidak kosong
3. ✓ Error check - Cek artifacts folder exists
4. ✓ Scripts created - setup_git_lfs.sh dan .bat

---

## 🎯 What to Do Now

### If Models < 100MB Each:
```bash
git add artifacts/
git commit -m "Add models"
git push
```

### If Models > 100MB:
```bash
./setup_git_lfs.sh     # or .bat on Windows
git add .gitattributes artifacts/
git commit -m "Add models via LFS"
git push
```

### If You Don't Have Models Yet:
- App will show error message
- You can still test with sample models
- Or deploy without models (show placeholder)

---

## 📁 Required Structure

```
results/
├── app.py                          ← Updated with absolute path
├── artifacts/                       ← MUST EXIST
│   ├── best_custom_pure_noaug.h5   ← Model files
│   ├── best_custom_lora_noaug.h5
│   ├── best_custom_lora_aug.h5
│   ├── best_vit_model.h5
│   ├── best_lora_densenet.h5
│   ├── hf_vit_pretrained_best.pt
│   ├── *.pkl
│   └── classic_models/
│       ├── svm_rbf.joblib
│       ├── random_forest.joblib
│       └── knn.joblib
├── packages.txt                     ← For OpenCV
├── requirements.txt                 ← opencv-python-headless
└── .gitattributes                  ← If using Git LFS
```

---

## 🎉 After Fix

App akan show:
```
✅ Loading AI models...
✅ 6 Deep Learning models loaded
✅ Ready to use!
```

Instead of:
```
❌ Unable to open file
❌ No such file or directory
```

---

<div align="center">

## Ready to Fix! 🔧

**Run setup script → Add files → Push → Done!**

</div>

