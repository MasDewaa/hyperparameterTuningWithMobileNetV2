# =========== SCRIPT PERBANDINGAN SISTEMATIS ===========
"""
Script untuk analisis perbandingan sistematis metode hyperparameter tuning.

Author: [Nama Anda]
Date: [Tanggal]
Version: 1.0

Description:
    Script ini melakukan analisis perbandingan komprehensif antara berbagai
    metode hyperparameter tuning (Grid Search, Random Search, Genetic Algorithm)
    dalam klasifikasi citra batik menggunakan MobileNetV2. Analisis mencakup
    performance metrics dan computational efficiency.

Requirements:
    - pandas
    - matplotlib
    - seaborn
    - numpy
    - json
    - os
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import numpy as np
import logging
import warnings
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('comparison_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_reproducibility(seed=42):
    """Setup reproducibility untuk hasil yang konsisten"""
    np.random.seed(seed)
    logger.info(f"Reproducibility setup dengan seed: {seed}")

def validate_file_paths():
    """Validasi path file yang diperlukan"""
    required_files = [
        'src/models/origin/baseline_evaluation_results.json',
        'src/models/hyperparameterTuning/gridSearch/grid_search_evaluation_results.json',
        'src/models/hyperparameterTuning/randomSearch/random_search_evaluation_results.json',
        'src/models/hyperparameterTuning/geneticAlgorithm/genetic_algorithm_evaluation_results.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"⚠️ File yang tidak ditemukan: {missing_files}")
        return False
    
    logger.info("✅ Semua file evaluasi ditemukan")
    return True

def load_evaluation_results():
    """Load semua hasil evaluasi dari file JSON dengan error handling"""
    results = {}
    
    # Path ke file hasil evaluasi
    evaluation_files = [
        'src/models/origin/baseline_evaluation_results.json',
        'src/models/hyperparameterTuning/gridSearch/grid_search_evaluation_results.json',
        'src/models/hyperparameterTuning/randomSearch/random_search_evaluation_results.json',
        'src/models/hyperparameterTuning/geneticAlgorithm/genetic_algorithm_evaluation_results.json'
    ]
    
    for file_path in evaluation_files:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Extract method name from file path
                    method_name = file_path.split('/')[-1].replace('_evaluation_results.json', '').replace('_', ' ').title()
                    
                    # Handle different JSON structures
                    if 'evaluation_metrics' in data:
                        # New structure with metadata
                        results[method_name] = data['evaluation_metrics']
                    else:
                        # Old structure
                        results[method_name] = data
                    
                    logger.info(f"✅ Berhasil load: {method_name}")
            else:
                logger.warning(f"⚠️ File tidak ditemukan: {file_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {str(e)}")
    
    return results

def load_efficiency_metrics():
    """Load semua metrics efisiensi dari file JSON dengan error handling"""
    metrics = {}
    
    # Path ke file metrics efisiensi
    efficiency_files = [
        'src/models/origin/baseline_efficiency_metrics.json',
        'src/models/hyperparameterTuning/gridSearch/grid_search_efficiency_metrics.json',
        'src/models/hyperparameterTuning/randomSearch/random_search_efficiency_metrics.json',
        'src/models/hyperparameterTuning/geneticAlgorithm/genetic_algorithm_efficiency_metrics.json'
    ]
    
    for file_path in efficiency_files:
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Extract method name from file path
                    method_name = file_path.split('/')[-1].replace('_efficiency_metrics.json', '').replace('_', ' ').title()
                    metrics[method_name] = data
                    
                    logger.info(f"✅ Berhasil load efficiency metrics: {method_name}")
            else:
                logger.warning(f"⚠️ File efficiency metrics tidak ditemukan: {file_path}")
                
        except Exception as e:
            logger.error(f"❌ Error loading efficiency metrics {file_path}: {str(e)}")
    
    return metrics

def create_performance_comparison_table(evaluation_results):
    """Buat tabel perbandingan performa dengan validasi data"""
    comparison_data = []
    
    for method, results in evaluation_results.items():
        try:
            # Validate required fields
            required_fields = ['test_accuracy', 'precision', 'recall', 'f1_score', 'test_loss']
            missing_fields = [field for field in required_fields if field not in results]
            
            if missing_fields:
                logger.warning(f"⚠️ Missing fields in {method}: {missing_fields}")
                continue
            
            comparison_data.append({
                'Method': method,
                'Accuracy': results['test_accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score'],
                'Test Loss': results['test_loss']
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing {method}: {str(e)}")
    
    if not comparison_data:
        logger.error("❌ Tidak ada data yang valid untuk perbandingan")
        return None
    
    df = pd.DataFrame(comparison_data)
    logger.info(f"✅ Tabel performance berhasil dibuat dengan {len(df)} metode")
    return df

def create_efficiency_comparison_table(efficiency_metrics):
    """Buat tabel perbandingan efisiensi dengan validasi data"""
    comparison_data = []
    
    for method, metrics in efficiency_metrics.items():
        try:
            # Validate required fields
            required_fields = ['total_tuning_time', 'total_trials', 'avg_time_per_trial', 
                             'best_accuracy', 'best_precision', 'best_recall', 'best_f1_score']
            missing_fields = [field for field in required_fields if field not in metrics]
            
            if missing_fields:
                logger.warning(f"⚠️ Missing fields in {method}: {missing_fields}")
                continue
            
            comparison_data.append({
                'Method': method,
                'Total Tuning Time (s)': metrics['total_tuning_time'],
                'Total Trials': metrics['total_trials'],
                'Avg Time per Trial (s)': metrics['avg_time_per_trial'],
                'Best Accuracy': metrics['best_accuracy'],
                'Best Precision': metrics['best_precision'],
                'Best Recall': metrics['best_recall'],
                'Best F1-Score': metrics['best_f1_score']
            })
            
        except Exception as e:
            logger.error(f"❌ Error processing {method}: {str(e)}")
    
    if not comparison_data:
        logger.error("❌ Tidak ada data yang valid untuk perbandingan efisiensi")
        return None
    
    df = pd.DataFrame(comparison_data)
    logger.info(f"✅ Tabel efisiensi berhasil dibuat dengan {len(df)} metode")
    return df

def create_comparison_visualizations(performance_df, efficiency_df):
    """Buat visualisasi perbandingan yang informatif"""
    try:
        # Create output directory
        os.makedirs('src/result/visualizations', exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Performance Metrics Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            row, col = i // 2, i % 2
            bars = axes[row, col].bar(performance_df['Method'], performance_df[metric], color=color, alpha=0.8)
            axes[row, col].set_title(f'{metric} Comparison', fontweight='bold')
            axes[row, col].set_ylabel(metric)
            axes[row, col].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                  f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('src/result/visualizations/performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Efficiency Metrics Comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Computational Efficiency Comparison', fontsize=16, fontweight='bold')
        
        # Total Tuning Time
        bars = axes[0, 0].bar(efficiency_df['Method'], efficiency_df['Total Tuning Time (s)'], 
                              color='#2E86AB', alpha=0.8)
        axes[0, 0].set_title('Total Tuning Time Comparison', fontweight='bold')
        axes[0, 0].set_ylabel('Time (seconds)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
        
        # Total Trials
        bars = axes[0, 1].bar(efficiency_df['Method'], efficiency_df['Total Trials'], 
                              color='#A23B72', alpha=0.8)
        axes[0, 1].set_title('Total Trials Comparison', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Trials')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{int(height)}', ha='center', va='bottom', fontsize=8)
        
        # Avg Time per Trial
        bars = axes[1, 0].bar(efficiency_df['Method'], efficiency_df['Avg Time per Trial (s)'], 
                              color='#F18F01', alpha=0.8)
        axes[1, 0].set_title('Average Time per Trial Comparison', fontweight='bold')
        axes[1, 0].set_ylabel('Time (seconds)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}s', ha='center', va='bottom', fontsize=8)
        
        # Best Accuracy vs Time (Scatter plot)
        scatter = axes[1, 1].scatter(efficiency_df['Total Tuning Time (s)'], efficiency_df['Best Accuracy'], 
                                    s=100, alpha=0.7, c=range(len(efficiency_df)), cmap='viridis')
        for i, method in enumerate(efficiency_df['Method']):
            axes[1, 1].annotate(method, (efficiency_df['Total Tuning Time (s)'].iloc[i], 
                                       efficiency_df['Best Accuracy'].iloc[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_title('Accuracy vs Tuning Time', fontweight='bold')
        axes[1, 1].set_xlabel('Total Tuning Time (s)')
        axes[1, 1].set_ylabel('Best Accuracy')
        
        plt.tight_layout()
        plt.savefig('src/result/visualizations/efficiency_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Visualisasi perbandingan berhasil dibuat")
        
    except Exception as e:
        logger.error(f"❌ Error dalam membuat visualisasi: {str(e)}")

def generate_comparison_report(performance_df, efficiency_df):
    """Generate laporan perbandingan lengkap dengan analisis mendalam"""
    try:
        # Buat direktori untuk hasil
        os.makedirs('src/result/comparison', exist_ok=True)
        
        # Simpan tabel perbandingan
        performance_df.to_csv('src/result/comparison/performance_comparison.csv', index=False)
        efficiency_df.to_csv('src/result/comparison/efficiency_comparison.csv', index=False)
        
        # Analisis mendalam
        best_accuracy_method = performance_df.loc[performance_df['Accuracy'].idxmax(), 'Method']
        fastest_method = efficiency_df.loc[efficiency_df['Total Tuning Time (s)'].idxmin(), 'Method']
        
        # Hitung efisiensi (accuracy/time)
        efficiency_df['Efficiency'] = efficiency_df['Best Accuracy'] / (efficiency_df['Total Tuning Time (s)'] + 1e-6)
        most_efficient_method = efficiency_df.loc[efficiency_df['Efficiency'].idxmax(), 'Method']
        
        # Generate laporan
        report = f"""
# LAPORAN PERBANDINGAN HYPERPARAMETER TUNING
## Klasifikasi Citra Batik dengan MobileNetV2

**Tanggal Analisis:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**Jumlah Metode:** {len(performance_df)}

### 📊 HASIL PERBANDINGAN PERFORMANCE

{performance_df.to_string(index=False)}

### ⏱️ HASIL PERBANDINGAN EFISIENSI

{efficiency_df.to_string(index=False)}

### 🏆 ANALISIS MENDALAM

#### **Performance Analysis:**
- **Metode Terbaik berdasarkan Accuracy:** {best_accuracy_method} ({performance_df.loc[performance_df['Accuracy'].idxmax(), 'Accuracy']:.4f})
- **Metode Terbaik berdasarkan F1-Score:** {performance_df.loc[performance_df['F1-Score'].idxmax(), 'Method']} ({performance_df.loc[performance_df['F1-Score'].idxmax(), 'F1-Score']:.4f})
- **Metode Terbaik berdasarkan Precision:** {performance_df.loc[performance_df['Precision'].idxmax(), 'Method']} ({performance_df.loc[performance_df['Precision'].idxmax(), 'Precision']:.4f})
- **Metode Terbaik berdasarkan Recall:** {performance_df.loc[performance_df['Recall'].idxmax(), 'Method']} ({performance_df.loc[performance_df['Recall'].idxmax(), 'Recall']:.4f})

#### **Efficiency Analysis:**
- **Metode Tercepat:** {fastest_method} ({efficiency_df.loc[efficiency_df['Total Tuning Time (s)'].idxmin(), 'Total Tuning Time (s)']:.2f} detik)
- **Metode Paling Efisien (Accuracy/Time):** {most_efficient_method} ({efficiency_df.loc[efficiency_df['Efficiency'].idxmax(), 'Efficiency']:.6f})
- **Metode dengan Trials Terbanyak:** {efficiency_df.loc[efficiency_df['Total Trials'].idxmax(), 'Method']} ({efficiency_df.loc[efficiency_df['Total Trials'].idxmax(), 'Total Trials']} trials)

### 📈 REKOMENDASI

#### **Untuk Akurasi Tertinggi:**
- **Rekomendasi:** {best_accuracy_method}
- **Alasan:** Mencapai akurasi tertinggi dengan margin yang signifikan

#### **Untuk Waktu Tercepat:**
- **Rekomendasi:** {fastest_method}
- **Alasan:** Menyelesaikan tuning dalam waktu tercepat

#### **Untuk Efisiensi Optimal:**
- **Rekomendasi:** {most_efficient_method}
- **Alasan:** Memberikan rasio accuracy/waktu terbaik

### 📋 KESIMPULAN

Berdasarkan analisis komprehensif terhadap {len(performance_df)} metode hyperparameter tuning:

1. **{best_accuracy_method}** menunjukkan performa terbaik dalam hal akurasi
2. **{fastest_method}** adalah metode tercepat dalam hal waktu tuning
3. **{most_efficient_method}** memberikan efisiensi komputasional terbaik

### 📊 METADATA ANALISIS

- **Total Metode:** {len(performance_df)}
- **Range Accuracy:** {performance_df['Accuracy'].min():.4f} - {performance_df['Accuracy'].max():.4f}
- **Range F1-Score:** {performance_df['F1-Score'].min():.4f} - {performance_df['F1-Score'].max():.4f}
- **Total Waktu Tuning:** {efficiency_df['Total Tuning Time (s)'].sum():.2f} detik
- **Total Trials:** {efficiency_df['Total Trials'].sum()}
"""
        
        with open('src/result/comparison/comparison_report.md', 'w') as f:
            f.write(report)
        
        logger.info("✅ Laporan perbandingan telah disimpan ke 'src/result/comparison/comparison_report.md'")
        logger.info("✅ Tabel perbandingan telah disimpan ke CSV files")
        
    except Exception as e:
        logger.error(f"❌ Error dalam generate laporan: {str(e)}")

def main():
    """Main function untuk menjalankan analisis perbandingan dengan best practice"""
    logger.info("🚀 Memulai analisis perbandingan sistematis...")
    
    try:
        # Setup reproducibility
        setup_reproducibility()
        
        # Validasi file paths
        validate_file_paths()
        
        # Load data
        evaluation_results = load_evaluation_results()
        efficiency_metrics = load_efficiency_metrics()
        
        if not evaluation_results:
            logger.error("❌ Tidak ada file hasil evaluasi yang ditemukan!")
            return
        
        # Buat tabel perbandingan
        performance_df = create_performance_comparison_table(evaluation_results)
        efficiency_df = create_efficiency_comparison_table(efficiency_metrics)
        
        if performance_df is None or efficiency_df is None:
            logger.error("❌ Gagal membuat tabel perbandingan!")
            return
        
        # Tampilkan hasil
        logger.info("\n📊 TABEL PERBANDINGAN PERFORMANCE:")
        print(performance_df.to_string(index=False))
        
        logger.info("\n⏱️ TABEL PERBANDINGAN EFISIENSI:")
        print(efficiency_df.to_string(index=False))
        
        # Buat visualisasi
        create_comparison_visualizations(performance_df, efficiency_df)
        
        # Generate laporan
        generate_comparison_report(performance_df, efficiency_df)
        
        logger.info("\n✅ Analisis perbandingan selesai!")
        
    except Exception as e:
        logger.error(f"❌ Error dalam analisis perbandingan: {str(e)}")
        raise

if __name__ == "__main__":
    main() 