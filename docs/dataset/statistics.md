# Statistik Dataset

## 📊 Distribusi Kelas

### Dataset Lengkap (COVID-QU-Ex)

| Kelas | Jumlah | Persentase |
|:------|:------:|:----------:|
| **COVID-19** | 11,956 | 35.3% |
| **Non-COVID** | 11,263 | 33.2% |
| **Normal** | 10,701 | 31.5% |
| **Total** | **33,920** | **100%** |

### Dataset yang Digunakan (Infection Segmentation Subset)

| Kelas | Jumlah | Persentase |
|:------|:------:|:----------:|
| **COVID-19** | 2,913 | 50.0% |
| **Non-COVID** | 1,457 | 25.0% |
| **Normal** | 1,456 | 25.0% |
| **Total** | **5,826** | **100%** |

## 📈 Pembagian Data

### Stratified Split

Dataset dibagi menggunakan **stratified split** untuk mempertahankan proporsi kelas di setiap subset:

| Split | COVID-19 | Non-COVID | Normal | Total | Persentase |
|:------|:--------:|:---------:|:------:|:-----:|:----------:|
| **Train** | 2,039 | 1,020 | 1,019 | 4,078 | 70% |
| **Validation** | 437 | 219 | 218 | 874 | 15% |
| **Test** | 437 | 218 | 219 | 874 | 15% |

### Distribusi per Split

```python
Train Set:
  COVID-19:  2,039 (50.0%)
  Non-COVID: 1,020 (25.0%)
  Normal:    1,019 (25.0%)
  Total:     4,078

Validation Set:
  COVID-19:  437 (50.0%)
  Non-COVID: 219 (25.0%)
  Normal:    218 (25.0%)
  Total:     874

Test Set:
  COVID-19:  437 (50.0%)
  Non-COVID: 218 (25.0%)
  Normal:    219 (25.0%)
  Total:     874
```

## 📐 Spesifikasi Citra

### Format Citra

- **Format:** PNG, JPG, JPEG
- **Mode:** Grayscale (setelah preprocessing)
- **Ukuran Standar:** 224×224 piksel
- **Bit Depth:** 8-bit (setelah normalisasi ke [0, 1])

### Preprocessing Pipeline

1. **Grayscale Conversion** - Konversi ke skala abu-abu
2. **Resize** - Standarisasi ke 224×224
3. **CLAHE** - Contrast Limited Adaptive Histogram Equalization
4. **Lung Cropping** - Heuristic cropping area paru-paru
5. **Normalisasi** - Skala ke [0, 1]

[📖 Detail preprocessing →](../methodology/preprocessing.md)

## 📊 Sumber Data

Citra X-Ray dalam COVID-QU-Ex dikumpulkan dari berbagai repository dan studi:

### COVID-19 Samples

1. QaTa-COV19 Database
2. Covid-19-image-repository
3. Eurorad
4. Covid-chestxray-dataset
5. COVID-19 DATABASE (SIRM)
6. COVID-19 Radiography Database (Kaggle)
7. COVID-CXNet (GitHub)

### Non-COVID & Normal Samples

8. RSNA Pneumonia Detection Challenge
9. Chest X-Ray Images (Pneumonia)
10. PadChest: Medical Imaging Databank of the Valencia Region

## 🔍 Karakteristik Dataset

### Kelebihan

- ✅ Dataset besar dan terstruktur dengan baik
- ✅ Memiliki ground-truth masks untuk lung segmentation
- ✅ Distribusi kelas relatif seimbang (untuk subset yang digunakan)
- ✅ Sumber data beragam meningkatkan generalisasi

### Tantangan

- ⚠️ Dataset terbatas untuk deep learning (5,826 citra)
- ⚠️ Imbalance kelas pada dataset lengkap (COVID-19 lebih banyak)
- ⚠️ Variasi kualitas citra dari berbagai sumber
- ⚠️ Perlu preprocessing yang tepat untuk optimalisasi

## 📈 Statistik Penggunaan

### Kaggle Dataset Stats (per November 2025)

- **Total Views:** 47.5K+
- **Total Downloads:** 10.4K+
- **Engagement Rate:** 0.21845 (downloads per view)
- **Usability Score:** 6.88/10

### Metadata Teknis

- **File Count:** 85.3k files
- **Dataset Size:** ~1.21 GB
- **License:** CC BY-SA 4.0
- **Update Frequency:** Not specified

