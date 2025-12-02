# Preprocessing

## 🔄 Pipeline Preprocessing

Pipeline preprocessing yang diterapkan pada semua citra dalam proyek ini:

1. **Grayscale Conversion** - Konversi citra ke skala abu-abu
2. **Resize** - Standarisasi ukuran menjadi `224×224` piksel
3. **CLAHE** - Contrast Limited Adaptive Histogram Equalization
4. **Heuristic Lung Cropping** - Otomatis crop area paru-paru
5. **Normalisasi** - Skala piksel ke rentang `[0, 1]`

[📖 Detail lengkap preprocessing →](../dataset/preprocessing.md)

## 🎯 Tujuan Preprocessing

1. **Standarisasi Input:** Memastikan semua citra memiliki format dan ukuran yang konsisten
2. **Peningkatan Kualitas:** Meningkatkan kontras dan menonjolkan struktur penting
3. **Optimasi Area:** Fokus pada area paru-paru yang relevan
4. **Stabilitas Numerik:** Normalisasi untuk training yang stabil

## 📊 Dampak pada Model

Preprocessing yang tepat sangat penting untuk:
- ✅ Meningkatkan akurasi model
- ✅ Mempercepat konvergensi training
- ✅ Meningkatkan generalisasi model
- ✅ Mengurangi overfitting

