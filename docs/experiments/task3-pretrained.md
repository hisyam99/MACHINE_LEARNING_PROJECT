# TASK 3: Pretrained Models + ViT

## 📋 Deskripsi

Task ini mengimplementasikan Transfer Learning menggunakan pretrained models (DenseNet121) dan Vision Transformer (ViT) dengan berbagai konfigurasi.

## 🔬 Model yang Diuji

### 1. DenseNet121 + LoRA

#### Arsitektur

```
Input (224×224×3) - RGB
    ↓
DenseNet121 Base (Frozen)
    ↓
Global Average Pooling
    ↓
LoRA Dense (256 units, rank=8)
    ↓
LoRA Dense (3 units, rank=8) → Softmax
    ↓
Output (3 classes)
```

#### Konfigurasi

```python
# Base Model (Frozen)
base_model = tf.keras.applications.DenseNet121(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)
base_model.trainable = False

# Head dengan LoRA
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = LoRADense(256, rank=8, alpha=64)(x)
x = LoRADense(3, rank=8, alpha=64, activation='softmax')(x)
```

#### Data Augmentation

```python
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1),
    layers.RandomBrightness(0.2),
    layers.RandomFlip("horizontal")
])
```

#### Hasil

- **Akurasi:** 82.04%
- **Macro F1:** 0.8003
- **F1 COVID-19:** 0.8743
- **F1 Non-COVID:** 0.7241
- **F1 Normal:** 0.8025

### 2. Vision Transformer (Manual)

#### Arsitektur

- **Patch Size:** 16×16
- **Embedding Dimension:** 768
- **Number of Heads:** 12
- **Number of Layers:** 12
- **MLP Dimension:** 3072

#### Implementasi

```python
# Patch Embedding
patches = layers.Conv2D(
    embed_dim,
    kernel_size=patch_size,
    strides=patch_size,
    padding='valid'
)(images)

# Transformer Blocks
for _ in range(num_layers):
    x = MultiHeadSelfAttention(num_heads, embed_dim)(x)
    x = layers.LayerNormalization()(x)
    x = MLP(mlp_dim)(x)
    x = layers.LayerNormalization()(x)
```

#### Hasil

- **Akurasi:** 68.54%
- **Macro F1:** 0.6645
- **F1 COVID-19:** 0.7569
- **F1 Non-COVID:** 0.5921
- **F1 Normal:** 0.6446

**Kesimpulan:** Training from scratch memerlukan dataset yang lebih besar.

### 3. HuggingFace ViT Pretrained

#### Arsitektur

- **Model:** `google/vit-base-patch16-224`
- **Patch Size:** 16×16
- **Image Size:** 224×224
- **Embedding Dimension:** 768
- **Number of Heads:** 12
- **Number of Layers:** 12

#### Konfigurasi

```python
from transformers import ViTForImageClassification, ViTImageProcessor

# Load pretrained model
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=3
)

# Image processor
processor = ViTImageProcessor.from_pretrained(
    'google/vit-base-patch16-224'
)
```

#### Fine-tuning

```python
# Training configuration
training_args = TrainingArguments(
    output_dir='./vit-covid',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_dir='./logs',
)
```

#### Hasil

- **Akurasi:** **91.65%** (terbaik)
- **Macro F1:** **0.9017**
- **F1 COVID-19:** **0.9601**
- **F1 Non-COVID:** **0.8677**
- **F1 Normal:** **0.8773**

## 📊 Perbandingan

| Model | Akurasi | Macro F1 | F1 COVID-19 | F1 Non-COVID | F1 Normal |
|:------|:-------:|:--------:|:-----------:|:------------:|:---------:|
| **HF ViT Pretrained** | **91.65%** | **0.9017** | **0.9601** | **0.8677** | **0.8773** |
| **DenseNet121 + LoRA** | 82.04% | 0.8003 | 0.8743 | 0.7241 | 0.8025 |
| **ViT Manual** | 68.54% | 0.6645 | 0.7569 | 0.5921 | 0.6446 |

## 🔍 Analisis

### HuggingFace ViT Pretrained

**Kelebihan:**
- ✅ **Performa terbaik:** 91.65% akurasi
- ✅ **F1 COVID-19 sangat tinggi:** 0.9601
- ✅ **Keseimbangan kelas baik:** Semua F1 >0.87
- ✅ **Robust:** Generalisasi baik

**Keterbatasan:**
- ⚠️ **Resource intensive:** Memerlukan lebih banyak memory
- ⚠️ **Training time:** Lebih lama dibandingkan CNN

### DenseNet121 + LoRA

**Kelebihan:**
- ✅ **Keseimbangan baik:** 82.04% akurasi dengan efisiensi
- ✅ **Efisien parameter:** LoRA mengurangi parameter
- ✅ **Stabil:** Training lebih stabil

**Keterbatasan:**
- ⚠️ **Performa lebih rendah:** Dibandingkan HF ViT
- ⚠️ **F1 Non-COVID rendah:** 0.7241

### ViT Manual

**Kelebihan:**
- ✅ **Dari scratch:** Tidak perlu pretrained weights
- ✅ **Fleksibel:** Dapat dikustomisasi

**Keterbatasan:**
- ⚠️ **Performa rendah:** 68.54% akurasi
- ⚠️ **Perlu dataset besar:** Training from scratch memerlukan lebih banyak data

## 💡 Kesimpulan

1. **Transfer Learning memberikan keunggulan signifikan** pada dataset terbatas
2. **HF ViT Pretrained adalah pilihan terbaik** untuk akurasi maksimal
3. **DenseNet121 + LoRA** memberikan keseimbangan baik antara performa dan efisiensi
4. **Training from scratch** memerlukan dataset yang lebih besar

## 📈 Visualisasi

Lihat visualisasi lengkap di:
- Training Curves: `output_images/acc_loss_*.png`
- Confusion Matrix: `output_images/conf_matrix_*.png`
- Predictions: `output_images/*_5_predict_true_false.png`

## 🔗 Referensi

- [Notebook: TASK_3:PretrainedDenseNet+ViT+Augmentation+LoRA.ipynb](../../TASK_3:PretrainedDenseNet+ViT+Augmentation+LoRA.ipynb)
- [Transfer Learning](../methodology/transfer-learning.md)
- [LoRA Implementation](../methodology/lora.md)

