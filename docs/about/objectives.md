# Tujuan Penelitian

## 🎯 Tujuan Utama

### 1. Komparasi Metode

Membandingkan efektivitas antara:
- **Machine Learning Klasik** dengan feature engineering manual (HOG + SVM)
- **Deep Learning** dengan berbagai arsitektur (CNN, Transfer Learning, ViT)

### 2. Evaluasi Trade-off

Menganalisis trade-off antara:
- **Akurasi** vs **Efisiensi Komputasi**
- **Ukuran Model** vs **Performa**
- **Waktu Training** vs **Akurasi**

### 3. Pengembangan Model Lightweight

Mengembangkan model yang:
- **Ringan** untuk deployment di perangkat mobile/edge
- **Efisien** dalam penggunaan parameter (menggunakan LoRA)
- **Akurat** untuk deteksi COVID-19

### 4. Evaluasi Teknik

Mengevaluasi dampak:
- **Data Augmentation** pada performa model
- **Transfer Learning** vs training from scratch
- **LoRA** untuk efisiensi parameter

## 📊 Metrik Evaluasi

### Metrik Utama

1. **Akurasi (Accuracy)**
   - Proporsi prediksi yang benar dari total prediksi

2. **Macro F1-Score**
   - Rata-rata F1-score untuk semua kelas (tanpa mempertimbangkan jumlah sampel)

3. **Weighted F1-Score**
   - Rata-rata F1-score yang dihitung berdasarkan jumlah sampel per kelas

4. **Per-Class F1-Score**
   - F1-score untuk setiap kelas (COVID-19, Non-COVID, Normal)

### Metrik Tambahan

- **Precision** per kelas
- **Recall** per kelas
- **Confusion Matrix**
- **Training Time**
- **Model Size**

## 🎯 Hipotesis

1. **Transfer Learning** akan memberikan performa lebih baik dibandingkan training from scratch pada dataset terbatas
2. **Data Augmentation** akan meningkatkan performa model, terutama untuk model kecil
3. **LoRA** dapat mengurangi jumlah parameter tanpa mengorbankan performa secara signifikan
4. **Vision Transformer** dengan pretrained weights akan memberikan performa terbaik
5. **SVM dengan HOG features** masih relevan sebagai baseline yang kuat

## 📈 Ekspektasi Hasil

1. Model dengan **transfer learning** mencapai akurasi >80%
2. Model **lightweight** dengan LoRA mencapai akurasi >75% dengan ukuran <5MB
3. **Data augmentation** meningkatkan akurasi minimal 5% untuk model from scratch
4. **HF ViT Pretrained** mencapai performa terbaik di antara semua model

## 🔬 Kontribusi Penelitian

1. **Komparasi komprehensif** antara classic ML dan deep learning untuk deteksi COVID-19
2. **Implementasi LoRA** untuk efisiensi parameter pada CNN
3. **Evaluasi data augmentation** pada dataset terbatas
4. **Benchmark** berbagai arsitektur pada dataset COVID-QU-Ex

