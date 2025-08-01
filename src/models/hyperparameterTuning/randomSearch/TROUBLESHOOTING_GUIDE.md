# 🔧 TROUBLESHOOTING GUIDE - RandomSearch Notebook

## 🚨 **MASALAH YANG SUDAH DIPERBAIKI:**

### 1. **ValueError: Unrecognized arguments ['max_epochs', 'factor']**
- **Masalah**: Parameter `max_epochs` dan `factor` tidak valid untuk `RandomSearch`
- **Solusi**: ✅ **SUDAH DIPERBAIKI** - Diganti dengan `max_trials=10`

### 2. **Missing max_trials parameter**
- **Masalah**: `RandomSearch` membutuhkan parameter `max_trials`
- **Solusi**: ✅ **SUDAH DIPERBAIKI** - Ditambahkan `max_trials=10`

### 3. **Missing reduce_lr callback**
- **Masalah**: Callback `reduce_lr` tidak dimasukkan dalam search
- **Solusi**: ✅ **SUDAH DIPERBAIKI** - Ditambahkan ke callbacks list

## 🎯 **CARA MENJALANKAN NOTEBOOK:**

### **Langkah 1: Restart Kernel**
```bash
# Di Jupyter, klik Kernel -> Restart & Clear Output
```

### **Langkah 2: Jalankan Sel Secara Berurutan**
1. **Cell 0**: Markdown (skip)
2. **Cell 1**: Time tracking imports ✅
3. **Cell 2**: Main imports ✅
4. **Cell 3**: Path setup ✅
5. **Cell 4**: Configuration ✅
6. **Cell 5**: Data augmentation ✅
7. **Cell 6**: Data generators ✅
8. **Cell 7**: Model building function ✅
9. **Cell 8**: RandomSearch tuner ✅
10. **Cell 9**: Callbacks dan search ✅
11. **Cell 10**: Final training ✅
12. **Cell 11**: Evaluation ✅
13. **Cell 12**: Timing summary ✅

## ⚠️ **JIKA KERNEL CRASH LAGI:**

### **Solusi 1: Restart WSL**
```bash
# Di Windows PowerShell (Run as Administrator)
wsl --shutdown
wsl
```

### **Solusi 2: Clean Memory**
```bash
# Di terminal WSL
sudo sync
echo 3 | sudo tee /proc/sys/vm/drop_caches
```

### **Solusi 3: Reduce Batch Size**
```python
# Ubah di Cell 4
BATCH_SIZE = 2  # Kurangi dari 4 ke 2
```

### **Solusi 4: Reduce Max Trials**
```python
# Ubah di Cell 8
max_trials=5  # Kurangi dari 10 ke 5
```

## 📊 **MONITORING RESOURCES:**

### **Check Memory Usage**
```bash
free -h
htop
```

### **Check Disk Usage**
```bash
df -h
```

### **Check CPU Usage**
```bash
top
```

## 🔍 **DEBUGGING TIPS:**

### **1. Check Imports**
```python
# Pastikan semua import berhasil
import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")

import kerastuner
print(f"Keras Tuner version: {kerastuner.__version__}")
```

### **2. Check Data Paths**
```python
import os
print(f"Train dir exists: {os.path.exists('../../../../dataset/train')}")
print(f"Val dir exists: {os.path.exists('../../../../dataset/val')}")
print(f"Test dir exists: {os.path.exists('../../../../dataset/test')}")
```

### **3. Check GPU Memory**
```python
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))
```

## 🎯 **EXPECTED OUTPUT:**

### **Successful Run Should Show:**
```
⏱️  Mulai tracking waktu komputasi: Random Search
📅 Waktu mulai: 2025-07-31 13:08:41

Search: Running Trial #1
Value             |Best Value So Far |Hyperparameter
0.0001            |0.0001            |learning_rate
adam              |adam              |optimizer
0.3               |0.3               |dropout_rate
128               |128               |dense_units
False             |False             |add_conv_layer

Epoch 1/30
...
```

## 🚨 **JIKA MASIH ERROR:**

### **Error: CUDA out of memory**
```python
# Tambahkan di awal notebook
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

### **Error: Import kerastuner**
```bash
pip install keras-tuner
```

### **Error: File not found**
```bash
# Pastikan berada di direktori yang benar
cd /home/nvidia/Batik-Final/src/models/hyperparameterTuning/randomSearch
pwd
ls -la
```

## 📞 **SUPPORT:**

Jika masih mengalami masalah, cek:
1. **Log Jupyter**: View -> Log
2. **System Resources**: Monitor memory dan disk
3. **WSL Status**: Pastikan WSL berjalan dengan baik

---

**✅ NOTEBOOK SUDAH DIPERBAIKI DAN SIAP DIJALANKAN!** 