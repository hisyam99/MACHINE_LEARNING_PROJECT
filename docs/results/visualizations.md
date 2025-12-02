# Visualisasi Hasil

## 📊 Grafik Training

### Custom CNN + LoRA (No Augmentation)

![Training Curves - Custom CNN No Aug](../output_images/acc_loss_custom_cnn_lora_no_augmentation.png)

**Observasi:**
- Overfitting mulai terjadi di Epoch 8
- Training accuracy mencapai ~85%
- Validation accuracy plateau di ~72%

### Custom CNN + LoRA (With Augmentation)

![Training Curves - Custom CNN With Aug](../output_images/acc_loss_custom_cnn_lora_with_augmentation.png)

**Observasi:**
- Training lebih stabil
- Training dan validation lebih seimbang
- Overfitting berkurang signifikan

### DenseNet121 + LoRA

![Training Curves - DenseNet121](../output_images/acc_loss_densenet121_lora.png)

**Observasi:**
- Konvergensi cepat
- Training dan validation seimbang
- Stabil dan konsisten

### Vision Transformer (Manual)

![Training Curves - ViT Manual](../output_images/acc_loss_VisionTransformer_manual.png)

**Observasi:**
- Training lebih lambat
- Perlu lebih banyak epochs
- Struggles dengan dataset terbatas

### HuggingFace ViT Pretrained

![Training Curves - HF ViT](../output_images/acc_loss_HF_VisionTransformer_Pretrained.png)

**Observasi:**
- Konvergensi sangat cepat
- Training dan validation sangat seimbang
- Performa terbaik

## 📈 Benchmark Performa

### Accuracy per Model

![Benchmark Accuracy](../output_images/benchmark_accuracy_per_model.png)

**Highlights:**
- HF ViT Pretrained: 91.65% (tertinggi)
- SVM (RBF): 86.27% (terbaik kedua)
- DenseNet121 + LoRA: 82.04%

### Macro F1 per Model

![Benchmark Macro F1](../output_images/benchmark_macro_f1_per_model.png)

**Highlights:**
- HF ViT Pretrained: 0.9017 (tertinggi)
- SVM (RBF): 0.843 (terbaik kedua)
- DenseNet121 + LoRA: 0.8003

### Accuracy vs Macro F1

![Accuracy vs Macro F1](../output_images/accuracy_vs_macro_f1_per_model.png)

**Observasi:**
- Korelasi positif antara accuracy dan macro F1
- HF ViT Pretrained unggul di kedua metrik
- SVM menunjukkan keseimbangan baik

## 📊 Analisis Per Kelas

### Per-Class F1-Score

![Per-Class F1](../output_images/per_class_f1_per_model.png)

**Observasi:**
- F1 COVID-19: Semua model >0.75, HF ViT mencapai 0.9601
- F1 Non-COVID: Hanya HF ViT dan DenseNet121 >0.70
- F1 Normal: Semua model >0.64

### Per-Class Error Rate (Recall)

![Per-Class Error Rate](../output_images/per_class_error_rate_per_model_recall.png)

**Observasi:**
- COVID-19: Error rate rendah untuk semua model
- Non-COVID: Error rate lebih tinggi (kelas paling sulit)
- Normal: Error rate menengah

## 🎯 Confusion Matrix

### Custom CNN + LoRA (No Augmentation)

![Confusion Matrix - Custom CNN No Aug](../output_images/conf_matrix_custom_cnn_lora_no_augmentation.png)

**Observasi:**
- Banyak false positives untuk COVID-19
- Non-COVID sering salah klasifikasi
- Normal sering diprediksi sebagai COVID-19

### Custom CNN + LoRA (With Augmentation)

![Confusion Matrix - Custom CNN With Aug](../output_images/conf_matrix_custom_cnn_lora_with_augmentation.png)

**Observasi:**
- False positives berkurang
- Non-COVID lebih akurat
- Keseimbangan lebih baik

### DenseNet121 + LoRA

![Confusion Matrix - DenseNet121](../output_images/conf_matrix_densenet121_lora.png)

**Observasi:**
- Keseimbangan baik untuk semua kelas
- False positives dan negatives seimbang
- Performa konsisten

### Vision Transformer (Manual)

![Confusion Matrix - ViT Manual](../output_images/conf_matrix_VisionTransformer_manual.png)

**Observasi:**
- Struggles dengan semua kelas
- Banyak kesalahan klasifikasi
- Perlu dataset lebih besar

### HuggingFace ViT Pretrained

![Confusion Matrix - HF ViT](../output_images/conf_matrix_HF_VisionTransformer_Pretrained.png)

**Observasi:**
- Keseimbangan sangat baik
- False positives dan negatives minimal
- Performa terbaik

## 🖼️ Visualisasi Prediksi

### Custom CNN + LoRA (No Augmentation)

![Predictions - Custom CNN No Aug](../output_images/custom_cnn_lora_no_augmentation_5_predict_true_false.png)

**Observasi:**
- Prediksi benar: Confidence tinggi untuk COVID-19
- Prediksi salah: Normal sering diprediksi sebagai COVID-19
- False positives dengan confidence tinggi

### Custom CNN + LoRA (With Augmentation)

![Predictions - Custom CNN With Aug](../output_images/custom_cnn_lora_with_augmentation_5_predict_true_false.png)

**Observasi:**
- Prediksi benar: Confidence lebih seimbang
- Prediksi salah: Masih ada false positives, tapi lebih sedikit
- Perbaikan signifikan dengan augmentation

### DenseNet121 + LoRA

![Predictions - DenseNet121](../output_images/densenet121_lora_5_predict_true_false.png)

**Observasi:**
- Prediksi benar: Confidence tinggi dan konsisten
- Prediksi salah: Lebih sedikit false positives
- Keseimbangan baik

### Vision Transformer (Manual)

![Predictions - ViT Manual](../output_images/VisionTransformer_manual_5_predict_true_false.png)

**Observasi:**
- Prediksi benar: Confidence bervariasi
- Prediksi salah: Banyak kesalahan klasifikasi
- Struggles dengan dataset terbatas

### HuggingFace ViT Pretrained

![Predictions - HF ViT](../output_images/HF_VisionTransformer_Pretrained_5_predict_true_false.png)

**Observasi:**
- Prediksi benar: Confidence sangat tinggi dan konsisten
- Prediksi salah: Sangat sedikit false positives
- Performa terbaik

## 📝 Catatan

Semua visualisasi tersedia di folder `../output_images/` dengan format:
- `acc_loss_*.png` - Training curves
- `conf_matrix_*.png` - Confusion matrices
- `*_5_predict_true_false.png` - Prediction visualizations
- `benchmark_*.png` - Benchmark comparisons
- `per_class_*.png` - Per-class analysis

[📊 Lihat analisis lengkap →](analysis.md)

