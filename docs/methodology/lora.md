# LoRA (Low-Rank Adaptation)

## 🎯 Konsep

**LoRA (Low-Rank Adaptation)** adalah teknik untuk fine-tuning model besar dengan cara yang efisien dengan menambahkan matriks low-rank ke weight yang sudah ada, tanpa mengubah weight asli.

## 🔬 Prinsip Kerja

### Traditional Fine-tuning

```python
# Semua parameter di-update
model.trainable = True
```

**Masalah:**
- Memerlukan update semua parameter
- Memory intensive
- Risk of overfitting

### LoRA Approach

```python
# Hanya menambahkan low-rank matrices
W_new = W_original + BA
```

Dimana:
- **B** dan **A** adalah matriks low-rank
- **r** (rank) << dimensi asli
- Hanya **B** dan **A** yang di-train

## 📐 Implementasi

### LoRA Dense Layer

```python
class LoRADense(tf.keras.layers.Layer):
    def __init__(self, units, rank=4, alpha=32, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.rank = rank
        self.alpha = alpha
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Low-rank matrices
        self.A = self.add_weight(
            name='lora_A',
            shape=(input_dim, self.rank),
            initializer='he_uniform',
            trainable=True
        )
        
        self.B = self.add_weight(
            name='lora_B',
            shape=(self.rank, self.units),
            initializer='zeros',
            trainable=True
        )
        
        # Scaling factor
        self.scaling = self.alpha / self.rank
    
    def call(self, inputs):
        # LoRA: W = W_original + (B @ A) * scaling
        lora_output = tf.matmul(
            tf.matmul(inputs, self.A),
            self.B
        ) * self.scaling
        
        # Dense layer (optional, bisa juga hanya LoRA)
        dense_output = tf.matmul(inputs, self.dense_kernel)
        
        return dense_output + lora_output
```

## 🔧 Parameter LoRA

### Rank (r)

- **Nilai:** Biasanya 4, 8, atau 16
- **Trade-off:** 
  - Rank lebih kecil = lebih efisien, tapi kurang fleksibel
  - Rank lebih besar = lebih fleksibel, tapi kurang efisien

### Alpha (α)

- **Nilai:** Biasanya 32 atau 64
- **Fungsi:** Scaling factor untuk LoRA weights
- **Rasio:** α/r menentukan seberapa besar pengaruh LoRA

## 📊 Efisiensi Parameter

### Perbandingan

| Metode | Parameters | Memory |
|:-------|:----------:|:------:|
| **Full Fine-tuning** | ~25M | ~100 MB |
| **LoRA (r=4)** | ~400K | ~1.8 MB |
| **Reduction** | **98.4%** | **98.2%** |

### Contoh: Custom CNN

- **Tanpa LoRA:** ~500K parameters
- **Dengan LoRA:** ~405K parameters
- **Pengurangan:** ~19%

## 🎯 Penggunaan dalam Proyek

### 1. Custom CNN

```python
# Dense layer diganti dengan LoRA
model.add(LoRADense(units=128, rank=4, alpha=32))
model.add(LoRADense(units=3, rank=4, alpha=32))  # Output layer
```

**Hasil:**
- Model size: ~1.8 MB
- Akurasi: 81.35% (dengan augmentation)

### 2. DenseNet121 + LoRA

```python
# Base model frozen
base_model = DenseNet121(weights='imagenet', include_top=False)
base_model.trainable = False

# LoRA head
x = base_model.output
x = LoRADense(units=256, rank=8, alpha=64)(x)
x = LoRADense(units=3, rank=8, alpha=64)(x)
```

**Hasil:**
- Akurasi: 82.04%
- Efisiensi parameter tinggi

## 💡 Kelebihan LoRA

1. **Efisiensi Parameter:**
   - Mengurangi jumlah parameter yang di-train
   - Mengurangi memory footprint

2. **Fleksibilitas:**
   - Dapat diterapkan pada berbagai layer
   - Mudah di-disable atau di-enable

3. **Stabilitas:**
   - Mengurangi risk of overfitting
   - Lebih stabil untuk dataset kecil

4. **Modularity:**
   - Dapat digunakan dengan berbagai base model
   - Mudah di-swap dengan dense layer biasa

## ⚠️ Keterbatasan

1. **Rank Selection:**
   - Perlu tuning untuk rank optimal
   - Terlalu kecil bisa mengurangi kapasitas model

2. **Layer Selection:**
   - Perlu memilih layer yang tepat untuk LoRA
   - Tidak semua layer cocok untuk LoRA

## 📚 Referensi

- **Paper:** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- **Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, et al.

## 🔗 Implementasi

Lihat implementasi lengkap di:
- [TASK 2: Custom CNN + LoRA](../experiments/task2-cnn-lora.md)
- [TASK 3: Transfer Learning + LoRA](../experiments/task3-pretrained.md)

