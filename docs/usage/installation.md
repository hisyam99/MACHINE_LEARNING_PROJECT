# Instalasi

## 📋 Prerequisites

### Python Version

- **Python 3.8+** (disarankan 3.9 atau 3.10)

### System Requirements

- **RAM:** Minimal 8GB (disarankan 16GB untuk training)
- **Storage:** Minimal 10GB untuk dataset dan model
- **GPU:** Opsional, tetapi disarankan untuk training deep learning (CUDA 11.0+)

## 🔧 Instalasi Dependencies

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/MACHINE_LEARNING.git
cd MACHINE_LEARNING
```

### 2. Setup Virtual Environment (Disarankan)

```bash
# Menggunakan venv
python -m venv venv

# Aktifkan virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

#### Option A: Install Semua (Recommended)

```bash
pip install -r requirements.txt
```

#### Option B: Install Manual

```bash
# Core dependencies
pip install numpy pandas matplotlib seaborn
pip install scikit-learn
pip install tensorflow>=2.10
pip install opencv-python
pip install tqdm
pip install jupyter

# Optional: Untuk Vision Transformer
pip install transformers
pip install torch torchvision

# Optional: Untuk Kaggle API
pip install kaggle
```

### 4. Verify Installation

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import sklearn; print(sklearn.__version__)"
python -c "import cv2; print(cv2.__version__)"
```

## 📦 Dataset Setup

### Download Dataset

1. **Dari Kaggle:**
   ```bash
   # Install Kaggle API
   pip install kaggle
   
   # Setup credentials (dapatkan dari Kaggle Account Settings)
   # Letakkan kaggle.json di ~/.kaggle/
   
   # Download dataset
   kaggle datasets download -d anasmohammedtahir/covidqu
   
   # Extract
   unzip covidqu.zip -d datasets/
   ```

2. **Manual Download:**
   - Kunjungi: https://www.kaggle.com/datasets/anasmohammedtahir/covidqu
   - Download dataset
   - Extract ke folder `datasets/`

### Struktur Folder

Setelah extract, struktur folder harus seperti ini:

```
datasets/
└── Infection Segmentation Data/
    └── Infection Segmentation Data/
        ├── Train/
        │   ├── COVID-19/
        │   ├── Non-COVID/
        │   └── Normal/
        ├── Val/
        │   ├── COVID-19/
        │   ├── Non-COVID/
        │   └── Normal/
        └── Test/
            ├── COVID-19/
            ├── Non-COVID/
            └── Normal/
```

## 🐛 Troubleshooting

### TensorFlow GPU Issues

Jika menggunakan GPU:

```bash
# Install CUDA dan cuDNN
# Lihat: https://www.tensorflow.org/install/gpu

# Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### OpenCV Issues

```bash
# Jika ada masalah dengan OpenCV
pip uninstall opencv-python
pip install opencv-python-headless
```

### Memory Issues

Jika mengalami masalah memory:

1. Kurangi batch size di notebook
2. Gunakan data generator dengan `yield`
3. Gunakan mixed precision training

## ✅ Verifikasi Setup

Jalankan script verifikasi:

```python
import sys
import importlib

required_packages = [
    'numpy', 'pandas', 'matplotlib', 'seaborn',
    'sklearn', 'tensorflow', 'cv2', 'tqdm', 'jupyter'
]

missing_packages = []
for package in required_packages:
    try:
        if package == 'cv2':
            importlib.import_module('cv2')
        elif package == 'sklearn':
            importlib.import_module('sklearn')
        else:
            importlib.import_module(package)
        print(f"✅ {package}")
    except ImportError:
        missing_packages.append(package)
        print(f"❌ {package}")

if missing_packages:
    print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
    print("Install dengan: pip install " + " ".join(missing_packages))
else:
    print("\n✅ Semua package terinstall dengan benar!")
```

## 📚 Next Steps

Setelah instalasi selesai:

1. [Quick Start Guide](quickstart.md) - Mulai menggunakan proyek
2. [Notebooks Guide](notebooks.md) - Menjalankan eksperimen
3. [Dataset Documentation](../dataset/introduction.md) - Informasi dataset

