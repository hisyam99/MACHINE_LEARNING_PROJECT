# Transfer Learning

## 🎯 Konsep

Transfer Learning memanfaatkan pengetahuan dari model yang sudah dilatih pada dataset besar (seperti ImageNet) dan mengadaptasikannya untuk tugas spesifik kita.

## 🔬 Model yang Digunakan

### 1. DenseNet121 + LoRA

**Base Model:** DenseNet121 pretrained pada ImageNet

**Arsitektur:**
- **Input:** (224, 224, 3) - RGB (dikonversi dari grayscale)
- **Base:** DenseNet121 (frozen)
- **Head:** LoRA Dense Layer
- **Output:** 3 classes

**Konfigurasi:**
- Base model **frozen** (tidak di-train)
- Hanya head yang di-train dengan LoRA
- Data augmentation diterapkan

**Hasil:**
- **Akurasi:** 82.04%
- **Macro F1:** 0.8003
- **F1 COVID-19:** 0.8743
- **F1 Non-COVID:** 0.7241
- **F1 Normal:** 0.8025

### 2. Vision Transformer (ViT)

#### A. Manual Implementation

ViT dari scratch menggunakan Keras.

**Arsitektur:**
- **Patch Size:** 16×16
- **Embedding Dimension:** 768
- **Number of Heads:** 12
- **Number of Layers:** 12

**Hasil:**
- **Akurasi:** 68.54%
- **Macro F1:** 0.6645

**Kesimpulan:** Training from scratch memerlukan dataset yang lebih besar.

#### B. HuggingFace Pretrained

ViT-Base/16 pretrained dari HuggingFace Transformers.

**Arsitektur:**
- **Model:** `google/vit-base-patch16-224`
- **Patch Size:** 16×16
- **Image Size:** 224×224
- **Embedding Dimension:** 768

**Konfigurasi:**
- Pretrained weights dari ImageNet
- Fine-tuning pada dataset COVID-19
- Data augmentation diterapkan

**Hasil:**
- **Akurasi:** **91.65%** (terbaik)
- **Macro F1:** **0.9017**
- **F1 COVID-19:** **0.9601**
- **F1 Non-COVID:** **0.8677**
- **F1 Normal:** **0.8773**

## 📊 Perbandingan

| Model | Akurasi | Macro F1 | Catatan |
|:------|:-------:|:--------:|:--------|
| **HF ViT Pretrained** | **91.65%** | **0.9017** | Best overall |
| **DenseNet121 + LoRA** | 82.04% | 0.8003 | Keseimbangan baik |
| **ViT Manual** | 68.54% | 0.6645 | Perlu dataset lebih besar |

## 🔍 Analisis

### Kelebihan Transfer Learning

1. **Performa Superior:**
   - HF ViT mencapai 91.65% akurasi
   - Mengungguli semua model lain

2. **Efisiensi Training:**
   - Tidak perlu train dari scratch
   - Konvergensi lebih cepat

3. **Generalization:**
   - Pretrained weights sudah belajar representasi umum
   - Lebih robust terhadap variasi data

### Tantangan

1. **Resource Requirements:**
   - Memerlukan lebih banyak memory
   - Training lebih lama (meskipun lebih efisien)

2. **Domain Gap:**
   - ImageNet (natural images) vs X-Ray (medical images)
   - Perlu fine-tuning yang tepat

## 💡 Kesimpulan

1. **Transfer Learning memberikan keunggulan signifikan** pada dataset terbatas
2. **Pretrained ViT adalah pilihan terbaik** untuk akurasi maksimal
3. **DenseNet121 + LoRA** memberikan keseimbangan baik antara performa dan efisiensi
4. **Training from scratch** memerlukan dataset yang lebih besar

[📖 Pelajari lebih lanjut tentang LoRA →](lora.md)

