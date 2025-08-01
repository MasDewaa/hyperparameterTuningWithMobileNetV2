# =========== PROJECT OPTIMIZATION GUIDE ===========

## 🚨 OPTIMASI UNTUK DISK USAGE TINGGI

### Current Issues:
- Disk usage 100% menghambat training
- VmmemWSL menggunakan 2.9GB RAM
- Slow I/O untuk dataset loading

## 🔧 OPTIMASI PROJECT BATIK:

### 1. **Reduce Batch Size:**
```python
# Ubah dari BATCH_SIZE = 8 menjadi:
BATCH_SIZE = 4  # Kurangi memory usage
```

### 2. **Optimasi Data Loading:**
```python
# Tambahkan prefetch dan caching
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True,
    # Tambahkan optimasi:
    workers=2,  # Kurangi worker threads
    use_multiprocessing=False  # Disable multiprocessing
)
```

### 3. **Memory Management:**
```python
# Tambahkan garbage collection
import gc

# Setelah setiap epoch:
gc.collect()
tf.keras.backend.clear_session()
```

### 4. **Checkpoint Strategy:**
```python
# Simpan model lebih jarang
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    save_weights_only=True,  # Hemat disk space
    verbose=1
)
```

### 5. **Reduce Logging:**
```python
# Kurangi verbose output
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint, reduce_lr],
    verbose=1  # Kurangi dari 2 ke 1
)
```

## 📊 MONITORING SCRIPT:
```python
import psutil
import time

def monitor_resources():
    """Monitor resource usage selama training"""
    while True:
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        print(f"CPU: {cpu_percent}% | RAM: {memory_percent}% | Disk: {disk_percent}%")
        
        if disk_percent > 90:
            print("⚠️ WARNING: Disk usage tinggi!")
            break
            
        time.sleep(30)  # Check setiap 30 detik
```

## 🎯 IMMEDIATE ACTIONS:

### **Langkah 1: Restart WSL**
```bash
# Di Windows PowerShell (Admin):
wsl --shutdown
wsl
```

### **Langkah 2: Cleanup**
```bash
cd /home/nvidia/Batik-Final
sudo apt clean
sudo apt autoremove
```

### **Langkah 3: Monitor**
```bash
htop  # Monitor resource usage
```

### **Langkah 4: Optimasi Project**
- Kurangi batch size ke 4
- Disable multiprocessing
- Tambahkan garbage collection

## ⚠️ WARNING:
- Backup semua work sebelum restart
- Monitor disk usage setelah optimasi
- Jika masih tinggi, pertimbangkan training di cloud 