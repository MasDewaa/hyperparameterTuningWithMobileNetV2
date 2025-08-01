# 🎯 PANDUAN PENYERAGAMAN PARAMETER HYPERPARAMETER

## 📋 **TUJUAN**
Menyeragamkan parameter hyperparameter di semua metode tuning (Random Search, Grid Search, Genetic Algorithm) untuk memastikan perbandingan yang adil dan konsisten.

## 🔧 **PARAMETER YANG DISERAGAMKAN**

### **1. Learning Rate**
```python
# SEMUA METODE MENGGUNAKAN:
LEARNING_RATE_CHOICES = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
LEARNING_RATE_MIN = 1e-4
LEARNING_RATE_MAX = 1e-2
```

### **2. Optimizer**
```python
# SEMUA METODE MENGGUNAKAN:
OPTIMIZER_CHOICES = ['adam', 'rmsprop', 'sgd']
```

### **3. Dense Units**
```python
# SEMUA METODE MENGGUNAKAN:
DENSE_UNITS_CHOICES = [64, 128, 256, 384, 512]
DENSE_UNITS_MIN = 64
DENSE_UNITS_MAX = 512
DENSE_UNITS_STEP = 64
```

### **4. Dropout Rate**
```python
# SEMUA METODE MENGGUNAKAN:
DROPOUT_RATE_CHOICES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
DROPOUT_RATE_MIN = 0.0
DROPOUT_RATE_MAX = 0.5
DROPOUT_RATE_STEP = 0.1
```

### **5. Conv Filters**
```python
# SEMUA METODE MENGGUNAKAN:
CONV_FILTERS_CHOICES = [32, 64, 96, 128]
CONV_FILTERS_MIN = 32
CONV_FILTERS_MAX = 128
CONV_FILTERS_STEP = 32
```

### **6. Add Conv Layer**
```python
# SEMUA METODE MENGGUNAKAN:
ADD_CONV_LAYER_CHOICES = [True, False]
```

## 📝 **LANGKAH PENYERAGAMAN**

### **Langkah 1: Import Konfigurasi Uniform**

Tambahkan di bagian atas setiap notebook (setelah import TensorFlow):

```python
# Import konfigurasi hyperparameter uniform
import sys
sys.path.append('..')
from hyperparameter_config import *
```

### **Langkah 2: Update Random Search**

**File:** `src/models/hyperparameterTuning/randomSearch/RandomSearch.ipynb`

**Ganti bagian build_model dengan:**

```python
def build_model(hp):
    # Base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    
    # Add conv layer (optional)
    if hp.Boolean('add_conv_layer', default=True):
        filters = hp.Int('conv_filters', 
                        min_value=CONV_FILTERS_MIN, 
                        max_value=CONV_FILTERS_MAX, 
                        step=CONV_FILTERS_STEP, 
                        default=64)
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(2, 2))
    
    # Dense layer
    dense_units = hp.Int('dense_units', 
                         min_value=DENSE_UNITS_MIN, 
                         max_value=DENSE_UNITS_MAX, 
                         step=DENSE_UNITS_STEP, 
                         default=128)
    model.add(Dense(dense_units, activation='relu'))
    
    # Dropout
    dropout_rate = hp.Float('dropout_rate', 
                           min_value=DROPOUT_RATE_MIN, 
                           max_value=DROPOUT_RATE_MAX, 
                           step=DROPOUT_RATE_STEP, 
                           default=0.2)
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(60, activation='softmax'))
    
    # Compile
    learning_rate = hp.Float('learning_rate', 
                            min_value=LEARNING_RATE_MIN, 
                            max_value=LEARNING_RATE_MAX, 
                            sampling='log', 
                            default=1e-3)
    
    optimizer_choice = hp.Choice('optimizer', values=OPTIMIZER_CHOICES)
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Update tuner configuration:**

```python
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=RANDOM_SEARCH_TRIALS,
    executions_per_trial=RANDOM_SEARCH_EXECUTIONS_PER_TRIAL,
    directory=project_dir,
    project_name=f'mobilenetv2_randomsearch_{int(time.time())}'
)
```

### **Langkah 3: Update Grid Search**

**File:** `src/models/hyperparameterTuning/gridSearch/GridSearch.ipynb`

**Ganti bagian build_model dengan:**

```python
def build_model(hp):
    # Base model
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    
    # Add conv layer (optional)
    if hp.Boolean('add_conv_layer', default=True):
        filters = hp.Choice('conv_filters', values=CONV_FILTERS_CHOICES, default=64)
        model.add(Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(2, 2))
    
    # Dense layer
    dense_units = hp.Choice('dense_units', values=DENSE_UNITS_CHOICES, default=128)
    model.add(Dense(dense_units, activation='relu'))
    
    # Dropout
    dropout_rate = hp.Choice('dropout_rate', values=DROPOUT_RATE_CHOICES, default=0.2)
    if dropout_rate > 0:
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(60, activation='softmax'))
    
    # Compile
    learning_rate = hp.Choice('learning_rate', values=LEARNING_RATE_CHOICES, default=1e-3)
    
    optimizer_choice = hp.Choice('optimizer', values=OPTIMIZER_CHOICES)
    if optimizer_choice == 'adam':
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_choice == 'rmsprop':
        optimizer = RMSprop(learning_rate=learning_rate)
    else:
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
```

**Update tuner configuration:**

```python
tuner = GridSearch(
    build_model,
    objective='val_accuracy',
    max_epochs=MAX_EPOCHS,
    directory=project_dir,
    project_name=f'mobilenetv2_gridsearch_{int(time.time())}'
)
```

### **Langkah 4: Update Genetic Algorithm**

**File:** `src/models/hyperparameterTuning/geneticAlgorithm/GeneticAlgorithm.ipynb`

**Ganti bagian create_random_individual dengan:**

```python
def create_random_individual(self):
    """Create a random individual with uniform hyperparameters"""
    return {
        'learning_rate': random.choice(LEARNING_RATE_CHOICES),
        'optimizer': random.choice(OPTIMIZER_CHOICES),
        'dense_units': random.choice(DENSE_UNITS_CHOICES),
        'dropout_rate': random.choice(DROPOUT_RATE_CHOICES),
        'add_conv_layer': random.choice(ADD_CONV_LAYER_CHOICES),
        'conv_filters': random.choice(CONV_FILTERS_CHOICES)
    }
```

**Update GA configuration:**

```python
# Genetic Algorithm parameters
population_size = GA_POPULATION_SIZE
generations = GA_GENERATIONS
mutation_rate = GA_MUTATION_RATE
crossover_rate = GA_CROSSOVER_RATE
```

### **Langkah 5: Update Training Parameters**

**Semua notebook menggunakan:**

```python
# Training parameters
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
SEED = 42
MAX_EPOCHS = 15
PATIENCE = 5

# Callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOPPING_PATIENCE,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=REDUCE_LR_FACTOR,
    patience=REDUCE_LR_PATIENCE,
    verbose=1,
    min_lr=REDUCE_LR_MIN_LR
)
```

## ✅ **VERIFIKASI PENYERAGAMAN**

### **Test Konfigurasi:**

```python
# Jalankan di setiap notebook untuk memverifikasi
from hyperparameter_config import validate_hyperparameter_config, print_hyperparameter_summary

validate_hyperparameter_config()
print_hyperparameter_summary()
```

### **Expected Output:**

```
✅ Konfigurasi hyperparameter valid

============================================================
📊 KONFIGURASI HYPERPARAMETER UNIFORM
============================================================

🔹 LEARNING_RATE:
   choices: [0.0001, 0.0003, 0.001, 0.003, 0.01]
   range: 0.0001 - 0.01

🔹 OPTIMIZER:
   choices: ['adam', 'rmsprop', 'sgd']

🔹 DENSE_UNITS:
   choices: [64, 128, 256, 384, 512]
   range: 64 - 512 (step: 64)

🔹 DROPOUT_RATE:
   choices: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
   range: 0.0 - 0.5 (step: 0.1)

🔹 CONV_FILTERS:
   choices: [32, 64, 96, 128]
   range: 32 - 128 (step: 32)

🔹 ADD_CONV_LAYER:
   choices: [True, False]

🔹 TRAINING:
   max_epochs: 15
   batch_size: 32
   patience: 5

🔹 TUNING:
   random_search_trials: 30
   ga_population_size: 8
   ga_generations: 10

============================================================
```

## 🎯 **MANFAAT PENYERAGAMAN**

1. **Perbandingan Adil**: Semua metode menggunakan search space yang sama
2. **Reproducibility**: Hasil dapat direproduksi dengan parameter yang konsisten
3. **Konsistensi**: Tidak ada bias karena perbedaan parameter
4. **Validitas Penelitian**: Memenuhi standar penelitian yang baik

## ⚠️ **CATATAN PENTING**

1. **Backup**: Backup notebook lama sebelum melakukan perubahan
2. **Test**: Jalankan test konfigurasi di setiap notebook
3. **Verifikasi**: Pastikan semua parameter sudah seragam
4. **Documentation**: Update dokumentasi setelah perubahan

## 📊 **TOTAL KOMBINASI**

- **Random Search**: 30 trials (sampling dari search space)
- **Grid Search**: 3600 kombinasi (exhaustive search)
- **Genetic Algorithm**: 8 individuals × 10 generations = 80 evaluations

Semua metode sekarang menggunakan search space yang identik untuk perbandingan yang adil! 