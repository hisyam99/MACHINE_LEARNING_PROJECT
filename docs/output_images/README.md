# Output Images

Folder ini berisi gambar visualisasi hasil eksperimen.

## 📋 Catatan

Gambar-gambar di folder ini akan otomatis di-copy dari folder `output_images/` di root repository saat build dokumentasi.

## 🖼️ Gambar yang Diperlukan

Folder ini harus berisi file-file berikut untuk dokumentasi lengkap:

### Training Curves
- `acc_loss_custom_cnn_lora_no_augmentation.png`
- `acc_loss_custom_cnn_lora_with_augmentation.png`
- `acc_loss_densenet121_lora.png`
- `acc_loss_VisionTransformer_manual.png`
- `acc_loss_HF_VisionTransformer_Pretrained.png`

### Benchmark
- `benchmark_accuracy_per_model.png`
- `benchmark_macro_f1_per_model.png`
- `accuracy_vs_macro_f1_per_model.png`

### Per-Class Analysis
- `per_class_f1_per_model.png`
- `per_class_error_rate_per_model_recall.png`

### Confusion Matrices
- `conf_matrix_custom_cnn_lora_no_augmentation.png`
- `conf_matrix_custom_cnn_lora_with_augmentation.png`
- `conf_matrix_densenet121_lora.png`
- `conf_matrix_VisionTransformer_manual.png`
- `conf_matrix_HF_VisionTransformer_Pretrained.png`

### Predictions
- `custom_cnn_lora_no_augmentation_5_predict_true_false.png`
- `custom_cnn_lora_with_augmentation_5_predict_true_false.png`
- `densenet121_lora_5_predict_true_false.png`
- `VisionTransformer_manual_5_predict_true_false.png`
- `HF_VisionTransformer_Pretrained_5_predict_true_false.png`

## ⚠️ Penting

Jika gambar tidak ada, dokumentasi akan tetap build tetapi link ke gambar akan broken. Pastikan semua gambar ada di folder `output_images/` di root repository sebelum push.

