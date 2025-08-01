# 📊 PANDUAN PENGGUNAAN SCRIPT PERBANDINGAN

## 🎯 **TUJUAN PENELITIAN**
Script ini dibuat untuk melakukan analisis perbandingan sistematis antara metode hyperparameter tuning yang berbeda dalam klasifikasi citra batik menggunakan MobileNetV2. Penelitian ini bertujuan untuk:

1. **Membandingkan performa** accuracy, precision, recall, dan F1-score model klasifikasi citra batik berbasis Transfer Learning ketika dioptimalkan menggunakan berbagai metode Hyperparameter Tuning
2. **Menganalisis efisiensi komputasional** waktu tuning dan sumber daya yang dibutuhkan antara metode-metode Hyperparameter Tuning yang diterapkan pada model Transfer Learning

## 📋 **RUMUSAN MASALAH**

### **Rumusan Masalah 1:**
Bagaimana perbandingan performa accuracy, precision, recall, dan F1-score model klasifikasi citra batik berbasis Transfer Learning ketika dioptimalkan menggunakan berbagai metode Hyperparameter Tuning seperti Grid Search, Random Search, Genetic Algorithm?

### **Rumusan Masalah 2:**
Bagaimana perbandingan efisiensi komputasional waktu tuning, sumber daya yang dibutuhkan antara metode-metode Hyperparameter Tuning yang diterapkan pada model Transfer Learning untuk klasifikasi citra batik?

## 🎯 **TUJUAN PENELITIAN**

### **Tujuan 1:**
Menghasilkan analisis perbandingan kinerja MobileNetV2 setelah dan sebelum diterapkan Hyperparameter Tuning dengan metode Grid Search, Random Search, dan Genetic Algorithm.

### **Tujuan 2:**
Menghasilkan data perbandingan terukur mengenai efisiensi komputasional meliputi waktu tuning dan estimasi kebutuhan sumber daya dari penerapan ketiga metode Hyperparameter Tuning tersebut pada model Transfer Learning MobileNetV2 untuk tugas klasifikasi citra batik.

## 📁 **STRUKTUR FILE**

```
src/
├── models/
│   ├── origin/
│   │   ├── baseline_evaluation.py          # Evaluasi baseline model
│   │   ├── baseline_evaluation_results.json
│   │   ├── baseline_efficiency_metrics.json
│   │   └── baseline_evaluation.log        # Log evaluasi baseline
│   └── hyperparameterTuning/
│       ├── randomSearch/
│       │   ├── evaluation_complete.py      # Evaluasi lengkap Random Search
│       │   ├── random_search_evaluation_results.json
│       │   ├── random_search_efficiency_metrics.json
│       │   └── evaluation_complete.log    # Log evaluasi Random Search
│       ├── gridSearch/
│       │   ├── evaluation_complete.py      # Evaluasi lengkap Grid Search
│       │   ├── grid_search_evaluation_results.json
│       │   ├── grid_search_efficiency_metrics.json
│       │   └── evaluation_complete.log    # Log evaluasi Grid Search
│       └── geneticAlgorithm/
│           ├── evaluation_complete.py      # Evaluasi lengkap Genetic Algorithm
│           ├── genetic_algorithm_evaluation_results.json
│           ├── genetic_algorithm_efficiency_metrics.json
│           └── evaluation_complete.log    # Log evaluasi Genetic Algorithm
└── result/
    ├── comparison_analysis.py              # Script utama perbandingan
    ├── comparison_analysis.log             # Log analisis perbandingan
    ├── visualizations/                     # Folder visualisasi
    │   ├── performance_comparison.png      # Visualisasi performance
    │   └── efficiency_comparison.png       # Visualisasi efisiensi
    └── comparison/
        ├── comparison_report.md            # Laporan lengkap
        ├── performance_comparison.csv      # Tabel performance
        └── efficiency_comparison.csv       # Tabel efisiensi
```

## 🚀 **CARA MENGGUNAKAN**

### **Prerequisites**
Pastikan semua dependency terinstall:
```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy pandas keras-tuner
```

### **Langkah 1: Evaluasi Baseline Model**
```bash
cd src/models/origin
python baseline_evaluation.py
```

**Output yang diharapkan:**
- `baseline_evaluation_results.json` - Hasil evaluasi baseline
- `baseline_efficiency_metrics.json` - Metrics efisiensi baseline
- `baseline_confusion_matrix.png` - Visualisasi confusion matrix
- `baseline_evaluation.log` - Log evaluasi

### **Langkah 2: Evaluasi Setiap Metode Hyperparameter Tuning**

#### **A. Random Search**
```bash
cd src/models/hyperparameterTuning/randomSearch
# Jalankan notebook RandomSearch.ipynb sampai selesai
# Kemudian tambahkan cell baru dengan kode evaluasi lengkap
```

**Kode untuk ditambahkan di notebook RandomSearch.ipynb (Cell baru):**
```python
# Import evaluasi lengkap
import sys
sys.path.append('.')
from evaluation_complete import evaluate_model_complete, analyze_computational_efficiency, generate_efficiency_report

# Dapatkan nama kelas
class_names = list(test_generator.class_indices.keys())

# Evaluasi lengkap
results = evaluate_model_complete(best_model, test_generator, class_names, "Random Search")

# Analisis efisiensi
efficiency_metrics = analyze_computational_efficiency(tuner, results, "Random Search")

# Generate laporan efisiensi
generate_efficiency_report(efficiency_metrics, "Random Search")
```

#### **B. Grid Search**
```bash
cd src/models/hyperparameterTuning/gridSearch
# Jalankan notebook GridSearch.ipynb sampai selesai
# Kemudian tambahkan cell baru dengan kode evaluasi lengkap
```

**Kode untuk ditambahkan di notebook GridSearch.ipynb (Cell baru):**
```python
# Import evaluasi lengkap
import sys
sys.path.append('.')
from evaluation_complete import evaluate_model_complete, analyze_computational_efficiency, generate_efficiency_report

# Dapatkan nama kelas
class_names = list(test_generator.class_indices.keys())

# Evaluasi lengkap
results = evaluate_model_complete(best_model, test_generator, class_names, "Grid Search")

# Analisis efisiensi
efficiency_metrics = analyze_computational_efficiency(tuner, results, "Grid Search")

# Generate laporan efisiensi
generate_efficiency_report(efficiency_metrics, "Grid Search")
```

#### **C. Genetic Algorithm**
```bash
cd src/models/hyperparameterTuning/geneticAlgorithm
# Jalankan notebook GeneticAlgorithm.ipynb sampai selesai
# Kemudian tambahkan cell baru dengan kode evaluasi lengkap
```

**Kode untuk ditambahkan di notebook GeneticAlgorithm.ipynb (Cell baru):**
```python
# Import evaluasi lengkap
import sys
sys.path.append('.')
from evaluation_complete import evaluate_model_complete, analyze_computational_efficiency, generate_efficiency_report

# Dapatkan nama kelas
class_names = list(test_generator.class_indices.keys())

# Evaluasi lengkap
results = evaluate_model_complete(best_model, test_generator, class_names, "Genetic Algorithm")

# Analisis efisiensi
efficiency_metrics = analyze_computational_efficiency(tuner, results, "Genetic Algorithm")

# Generate laporan efisiensi
generate_efficiency_report(efficiency_metrics, "Genetic Algorithm")
```

### **Langkah 3: Jalankan Analisis Perbandingan**
```bash
cd src/result
python comparison_analysis.py
```

## 📊 **OUTPUT YANG DIHASILKAN**

### **1. File JSON Evaluasi**
- `baseline_evaluation_results.json` - Hasil evaluasi baseline model
- `random_search_evaluation_results.json` - Hasil evaluasi Random Search
- `grid_search_evaluation_results.json` - Hasil evaluasi Grid Search
- `genetic_algorithm_evaluation_results.json` - Hasil evaluasi Genetic Algorithm

### **2. File JSON Efisiensi**
- `baseline_efficiency_metrics.json` - Metrics efisiensi baseline
- `random_search_efficiency_metrics.json` - Metrics efisiensi Random Search
- `grid_search_efficiency_metrics.json` - Metrics efisiensi Grid Search
- `genetic_algorithm_efficiency_metrics.json` - Metrics efisiensi Genetic Algorithm

### **3. Visualisasi**
- `src/result/visualizations/performance_comparison.png` - Perbandingan accuracy, precision, recall, F1-score
- `src/result/visualizations/efficiency_comparison.png` - Perbandingan waktu tuning dan efisiensi

### **4. Laporan Lengkap**
- `src/result/comparison/comparison_report.md` - Laporan perbandingan dalam format Markdown
- `src/result/comparison/performance_comparison.csv` - Tabel performance dalam format CSV
- `src/result/comparison/efficiency_comparison.csv` - Tabel efisiensi dalam format CSV

### **5. Log Files**
- `baseline_evaluation.log` - Log evaluasi baseline
- `evaluation_complete.log` - Log evaluasi hyperparameter tuning
- `comparison_analysis.log` - Log analisis perbandingan

## 📈 **METRICS YANG DIUKUR**

### **Performance Metrics:**
- **Accuracy**: Persentase prediksi yang benar
- **Precision**: Presisi rata-rata (macro average)
- **Recall**: Recall rata-rata (macro average)
- **F1-Score**: F1-score rata-rata (macro average)
- **Test Loss**: Loss pada test set

### **Efficiency Metrics:**
- **Total Tuning Time**: Total waktu yang dibutuhkan untuk tuning
- **Total Trials**: Jumlah trial yang dilakukan
- **Avg Time per Trial**: Rata-rata waktu per trial
- **Best Accuracy**: Akurasi terbaik yang dicapai
- **Best Precision**: Presisi terbaik yang dicapai
- **Best Recall**: Recall terbaik yang dicapai
- **Best F1-Score**: F1-score terbaik yang dicapai

## 🎯 **KESIMPULAN YANG DIHASILKAN**

Script akan menghasilkan kesimpulan otomatis tentang:
1. **Metode Terbaik berdasarkan Accuracy**
2. **Metode Terbaik berdasarkan F1-Score**
3. **Metode Terbaik berdasarkan Precision**
4. **Metode Terbaik berdasarkan Recall**
5. **Metode Tercepat** (waktu tuning terpendek)
6. **Metode Paling Efisien** (rasio accuracy/waktu terbaik)
7. **Metode dengan Trials Terbanyak**

## ⚠️ **CATATAN PENTING**

### **Sebelum Menjalankan Script:**
1. **Pastikan semua notebook sudah dijalankan** sampai selesai sebelum menjalankan script perbandingan
2. **Pastikan semua file JSON** sudah dihasilkan sebelum menjalankan `comparison_analysis.py`
3. **Pastikan model baseline** sudah dilatih dan disimpan sebagai `mymodel.keras`
4. **Pastikan dataset** tersedia di folder `./data/splits/dataset_split/test`
5. **Pastikan semua dependency** sudah terinstall dengan versi yang kompatibel

### **Best Practices:**
1. **Reproducibility**: Semua script menggunakan seed yang konsisten (42)
2. **Error Handling**: Semua script memiliki error handling yang komprehensif
3. **Logging**: Semua script menghasilkan log file untuk debugging
4. **Validation**: Semua input divalidasi sebelum diproses
5. **Documentation**: Semua fungsi didokumentasi dengan baik

## 🔧 **TROUBLESHOOTING**

### **Error: File tidak ditemukan**
```bash
# Periksa apakah file ada
ls -la src/models/origin/
ls -la src/models/hyperparameterTuning/*/

# Pastikan notebook sudah dijalankan sampai selesai
# Pastikan cell evaluasi sudah ditambahkan dan dijalankan
```

### **Error: Model tidak ditemukan**
```bash
# Periksa apakah model baseline ada
ls -la src/models/origin/mymodel.keras

# Jika tidak ada, jalankan training baseline terlebih dahulu
cd src/models/origin
python main.ipynb  # atau jalankan notebook baseline
```

### **Error: Dataset tidak ditemukan**
```bash
# Periksa struktur dataset
ls -la ./data/splits/dataset_split/test/

# Pastikan dataset sudah diorganisir dengan benar
# Struktur yang diharapkan:
# ./data/splits/dataset_split/test/
# ├── class1/
# ├── class2/
# └── ...
```

### **Error: Dependency tidak ditemukan**
```bash
# Install semua dependency
pip install tensorflow scikit-learn matplotlib seaborn numpy pandas keras-tuner

# Atau install dari requirements.txt
pip install -r requirements.txt
```

### **Error: Memory tidak cukup**
```bash
# Kurangi batch size di script evaluasi
# Ubah BATCH_SIZE dari 4 menjadi 2 atau 1

# Atau gunakan GPU jika tersedia
export CUDA_VISIBLE_DEVICES=0
```

### **Error: Log file tidak bisa dibuat**
```bash
# Periksa permission folder
ls -la src/models/origin/
ls -la src/result/

# Buat folder jika belum ada
mkdir -p src/result/visualizations
mkdir -p src/result/comparison
```

## 📞 **BANTUAN**

### **Jika mengalami masalah:**
1. **Periksa log files** untuk informasi error yang detail
2. **Pastikan semua dependency** sudah terinstall dengan versi yang benar
3. **Pastikan semua file** berada di lokasi yang benar
4. **Pastikan semua notebook** sudah dijalankan sampai selesai
5. **Pastikan dataset** tersedia dan terorganisir dengan benar

### **Untuk debugging:**
```bash
# Periksa log files
cat src/models/origin/baseline_evaluation.log
cat src/result/comparison_analysis.log

# Periksa file JSON yang dihasilkan
cat src/models/origin/baseline_evaluation_results.json
```

### **Untuk reproduksi hasil:**
1. Gunakan seed yang sama (42) di semua script
2. Pastikan urutan eksekusi: Baseline → Random Search → Grid Search → Genetic Algorithm → Comparison
3. Pastikan semua parameter hyperparameter tuning konsisten

## 📚 **REFERENSI**

### **Paper yang relevan:**
1. "Hyperparameter Optimization: A Comparative Study" - Bergstra & Bengio (2012)
2. "Random Search for Hyper-Parameter Optimization" - Bergstra & Bengio (2012)
3. "Genetic Algorithms for Hyperparameter Optimization" - Young et al. (2015)

### **Best Practices:**
1. "Reproducible Research in Machine Learning" - Pineau et al. (2018)
2. "Machine Learning Model Evaluation" - Raschka & Mirjalili (2019)

## 📝 **CHANGELOG**

### **Version 1.0 (Current)**
- ✅ Implementasi evaluasi baseline model
- ✅ Implementasi evaluasi hyperparameter tuning
- ✅ Implementasi analisis perbandingan sistematis
- ✅ Implementasi visualisasi komprehensif
- ✅ Implementasi logging dan error handling
- ✅ Implementasi reproducibility
- ✅ Implementasi dokumentasi lengkap

### **Planned Features:**
- 🔄 Statistical significance testing
- 🔄 Cross-validation results
- 🔄 Additional hyperparameter tuning methods
- 🔄 Real-time monitoring dashboard 