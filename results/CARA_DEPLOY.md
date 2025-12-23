# 🚀 Cara Deploy Dashboard ke Streamlit Cloud

## ✅ Error Sudah Diperbaiki

Semua error deployment sudah diselesaikan:
- ✓ OpenCV import error → Fixed
- ✓ Model path error → Fixed
- ✓ Empty label warning → Fixed

Tinggal **upload model files** ke GitHub!

---

## 📦 Problem Utama

**Error yang Anda alami:**
```
Unable to open file: './artifacts/best_custom_pure_noaug.h5'
No such file or directory
```

**Artinya:** Folder `artifacts/` dengan model files **tidak ada di GitHub**.

---

## 🎯 Solusi Simple - 4 Langkah

### Langkah 1: Cek Ukuran File Model

```bash
cd results
ls -lh artifacts/*.h5 artifacts/*.pt

# Lihat ukuran setiap file
```

**Jika ada file > 100MB:** Perlu Git LFS (lanjut ke langkah 2a)  
**Jika semua < 100MB:** Langsung ke langkah 3

---

### Langkah 2a: Setup Git LFS (File Besar > 100MB)

**Linux/Mac:**
```bash
./setup_git_lfs.sh
```

**Windows:**
```bash
setup_git_lfs.bat
```

Script akan otomatis:
- ✓ Check Git LFS installed
- ✓ Initialize Git LFS
- ✓ Track *.h5, *.pt, *.pkl files
- ✓ Create .gitattributes

---

### Langkah 2b: Manual Git LFS (Jika Script Gagal)

```bash
# Install Git LFS dulu (sekali saja)
# Ubuntu: sudo apt-get install git-lfs
# Mac: brew install git-lfs
# Windows: Download dari https://git-lfs.github.com/

# Initialize
git lfs install

# Track large files
git lfs track "artifacts/*.h5"
git lfs track "artifacts/*.pt"
git lfs track "artifacts/*.pkl"

# Lihat yang ditrack
git lfs track
```

---

### Langkah 3: Add & Commit Files

**Jika pakai Git LFS:**
```bash
git add .gitattributes
git add artifacts/
git add app.py packages.txt requirements.txt
git commit -m "Add model files and fix deployment issues"
```

**Jika tidak pakai Git LFS (file kecil):**
```bash
git add artifacts/
git add app.py packages.txt requirements.txt
git commit -m "Add model files and fix deployment issues"
```

---

### Langkah 4: Push ke GitHub

```bash
git push origin main

# Atau jika branch lain:
git push origin <branch-name>
```

**Tunggu push selesai** (bisa 1-10 menit tergantung ukuran files)

---

### Langkah 5: Deploy/Redeploy di Streamlit Cloud

**Jika belum deploy:**
1. Buka https://share.streamlit.io/
2. Sign in dengan GitHub
3. Click "New app"
4. Pilih repository
5. Main file: `results/app.py`
6. Deploy!

**Jika sudah deploy:**
1. Streamlit Cloud akan **auto-redeploy** setelah Anda push
2. Tunggu 5-10 menit
3. Refresh halaman app Anda
4. Done!

---

## ✅ Verification

Setelah deployment selesai, cek:

### 1. App Loads
```
✓ Dashboard terbuka tanpa error
✓ Tidak ada "Unable to open file"
✓ Loading spinner selesai
```

### 2. Models Load
```
ℹ️ Classic ML models unavailable (sklearn version)
   Deep Learning models (6 models) ready

atau

✓ All 9 models loaded successfully
```

### 3. Features Work
```
✓ Bisa upload gambar
✓ Bisa pilih mode analysis
✓ Run analysis berhasil
✓ Results ditampilkan
✓ Charts muncul
```

---

## 📁 Struktur yang Benar

Pastikan struktur folder seperti ini di GitHub:

```
your-repository/
└── results/
    ├── app.py                      ✓ Main app
    ├── requirements.txt            ✓ opencv-python-headless
    ├── packages.txt                ✓ System deps
    ├── setup_git_lfs.sh            ✓ Helper script
    ├── setup_git_lfs.bat           ✓ Helper script
    │
    └── artifacts/                  ← WAJIB ADA!
        ├── best_custom_pure_noaug.h5
        ├── best_custom_lora_noaug.h5
        ├── best_custom_lora_aug.h5
        ├── best_vit_model.h5
        ├── best_lora_densenet.h5
        ├── hf_vit_pretrained_best.pt
        ├── history_*.pkl
        └── classic_models/
            ├── svm_rbf.joblib
            ├── random_forest.joblib
            └── knn.joblib
```

---

## 🔍 Cara Cek di GitHub

1. Buka repository di GitHub web
2. Navigate ke `results/artifacts/`
3. Pastikan semua file model ada
4. File besar akan punya badge "Stored with Git LFS"

---

## 💡 Tips

### Jika Git LFS Terlalu Kompleks

Gunakan salah satu alternatif:

**Option 1: Compress Models**
```python
# Reduce model size dengan quantization
# TensorFlow Lite atau ONNX
```

**Option 2: Upload ke Cloud Storage**
```python
# Google Drive, Dropbox, Hugging Face
# Download di app.py saat startup
```

**Option 3: Hanya Deploy 1-2 Model**
```python
# Pilih model terbaik saja
# Edit load_deep_models() untuk load fewer models
```

---

## 🎯 Yang Paling Penting

### Untuk Deployment Berhasil:

1. ✅ **opencv-python-headless** ← Fixed
2. ✅ **packages.txt exists** ← Fixed
3. ✅ **Path menggunakan absolute** ← Fixed
4. ✅ **Radio label tidak kosong** ← Fixed
5. 📦 **artifacts/ folder di GitHub** ← **ANDA PERLU PUSH INI**

---

## 🚀 Perintah Lengkap (Copy-Paste)

```bash
# Masuk ke folder results
cd "/home/hisyam99/Documents/TUGAS KULIAH/SEMESTER 7/MACHINE_LEARNING/MACHINE_LEARNING/results"

# Setup Git LFS (sekali saja)
chmod +x setup_git_lfs.sh
./setup_git_lfs.sh

# Add semua files
git add .gitattributes
git add artifacts/
git add app.py packages.txt requirements.txt
git add *.md

# Commit
git commit -m "Fix all deployment errors + add model files"

# Push
git push origin main

# Selesai! Tunggu 5-10 menit untuk redeploy otomatis
```

---

## ✨ Expected Result

Setelah push dan redeploy:

```
🎉 Deployment Successful!
✓ App accessible at: https://your-app.streamlit.app
✓ All models loaded
✓ Dark mode working
✓ Light mode working
✓ Ready to use!
```

---

## 📞 Jika Masih Error

1. **Cek log** di Streamlit Cloud dashboard
2. **Verify artifacts/** ada di GitHub
3. **Check Git LFS** dengan `git lfs ls-files`
4. **Force reboot** app di Streamlit Cloud
5. **Baca** `MODEL_FILES_SETUP.md` untuk detail

---

<div align="center">

## 🎊 Siap Deploy!

**Jalankan script → Push → Done!**

**Your app will be live in 10 minutes! 🚀**

</div>

