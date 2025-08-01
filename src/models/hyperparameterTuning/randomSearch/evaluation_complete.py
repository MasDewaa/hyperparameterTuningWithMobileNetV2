# =========== EVALUASI LENGKAP MODEL ===========
"""
Script untuk evaluasi lengkap model klasifikasi citra batik dengan hyperparameter tuning.

Author: [Nama Anda]
Date: [Tanggal]
Version: 1.0

Description:
    Script ini melakukan evaluasi komprehensif pada model yang telah dioptimasi
    dengan berbagai metode hyperparameter tuning. Evaluasi mencakup accuracy,
    precision, recall, F1-score, confusion matrix, dan analisis efisiensi.

Requirements:
    - TensorFlow 2.x
    - scikit-learn
    - matplotlib
    - seaborn
    - numpy
    - pandas
    - keras-tuner (untuk analisis efisiensi)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import logging
import os
import time
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import warnings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation_complete.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

def setup_reproducibility(seed=42):
    """Setup reproducibility untuk hasil yang konsisten"""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Reproducibility setup dengan seed: {seed}")

def validate_model_and_data(model, test_generator):
    """Validasi model dan data generator"""
    if model is None:
        logger.error("❌ Model tidak valid (None)")
        return False
    
    if test_generator is None:
        logger.error("❌ Test generator tidak valid (None)")
        return False
    
    if test_generator.samples == 0:
        logger.error("❌ Test generator tidak memiliki sampel")
        return False
    
    logger.info(f"✅ Validasi model dan data berhasil - {test_generator.samples} sampel")
    return True

def evaluate_model_complete(model, test_generator, class_names, method_name="Model"):
    """Evaluasi model dengan metrics lengkap dan best practice"""
    logger.info(f"🚀 Memulai evaluasi lengkap untuk {method_name}")
    
    # Setup reproducibility
    setup_reproducibility()
    
    # Validasi input
    if not validate_model_and_data(model, test_generator):
        return None
    
    start_time = time.time()
    
    try:
        # Prediksi
        logger.info("Melakukan prediksi pada test set...")
        y_pred = model.predict(test_generator, verbose=0)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = test_generator.classes
        
        # Metrics dasar
        logger.info("Menghitung metrics dasar...")
        test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
        
        # Classification Report
        logger.info("Menghitung classification report...")
        report = classification_report(y_true, y_pred_classes, 
                                    target_names=class_names, 
                                    output_dict=True)
        
        # Confusion Matrix
        logger.info("Menghitung confusion matrix...")
        cm = confusion_matrix(y_true, y_pred_classes)
        
        # Per-class metrics dengan precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='macro')
        
        evaluation_time = time.time() - start_time
        
        # Print results
        logger.info(f"\n📊 HASIL EVALUASI LENGKAP - {method_name.upper()}:")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Macro Avg Precision: {precision:.4f}")
        logger.info(f"Macro Avg Recall: {recall:.4f}")
        logger.info(f"Macro Avg F1-Score: {f1:.4f}")
        logger.info(f"Waktu Evaluasi: {evaluation_time:.2f} detik")
        logger.info(f"Jumlah Sampel: {len(y_true)}")
        logger.info(f"Jumlah Kelas: {len(class_names)}")
        
        # Buat visualisasi confusion matrix
        create_confusion_matrix_visualization(cm, class_names, method_name)
        
        # Simpan hasil evaluasi
        save_evaluation_results({
            'test_accuracy': float(test_accuracy),
            'test_loss': float(test_loss),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'evaluation_time': evaluation_time,
            'num_samples': len(y_true),
            'num_classes': len(class_names)
        }, method_name)
        
        return {
            'test_accuracy': test_accuracy,
            'test_loss': test_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'evaluation_time': evaluation_time,
            'num_samples': len(y_true),
            'num_classes': len(class_names)
        }
        
    except Exception as e:
        logger.error(f"❌ Error dalam evaluasi model: {str(e)}")
        return None

def create_confusion_matrix_visualization(cm, class_names, method_name="Model"):
    """Buat visualisasi confusion matrix yang informatif"""
    try:
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create subplot for both raw and normalized
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Raw confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=class_names[:20], yticklabels=class_names[:20])
        ax1.set_title(f'Confusion Matrix - {method_name} Model (Raw Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        ax1.tick_params(axis='x', rotation=45)
        ax1.tick_params(axis='y', rotation=45)
        
        # Normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=ax2,
                    xticklabels=class_names[:20], yticklabels=class_names[:20])
        ax2.set_title(f'Confusion Matrix - {method_name} Model (Normalized)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        ax2.tick_params(axis='x', rotation=45)
        ax2.tick_params(axis='y', rotation=45)
        
        plt.tight_layout()
        
        # Simpan gambar
        output_file = f'confusion_matrix_{method_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"✅ Confusion matrix disimpan: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Error dalam membuat visualisasi confusion matrix: {str(e)}")

def save_evaluation_results(results, method_name="Model"):
    """Simpan hasil evaluasi dengan metadata"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Tambahkan metadata
        results_with_metadata = {
            'method': method_name,
            'timestamp': timestamp,
            'evaluation_metrics': results,
            'model_info': {
                'num_samples': results['num_samples'],
                'num_classes': results['num_classes'],
                'evaluation_time': results['evaluation_time']
            }
        }
        
        # Simpan hasil evaluasi
        output_file = f'{method_name.lower().replace(" ", "_")}_evaluation_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_with_metadata, f, indent=2)
        
        logger.info(f"✅ Hasil evaluasi disimpan: {output_file}")
        
    except Exception as e:
        logger.error(f"❌ Error dalam menyimpan hasil evaluasi: {str(e)}")

def analyze_computational_efficiency(tuner, results, method_name="Model"):
    """Analisis efisiensi komputasional dengan best practice"""
    logger.info(f"🔍 Memulai analisis efisiensi komputasional untuk {method_name}")
    
    try:
        # Ambil informasi waktu tuning dari tuner
        tuning_time = 0
        total_trials = 0
        avg_time_per_trial = 0
        
        if tuner is not None:
            try:
                # Coba ambil informasi dari tuner
                if hasattr(tuner, 'oracle') and hasattr(tuner.oracle, 'trials'):
                    total_trials = len(tuner.oracle.trials)
                    
                    # Hitung total waktu tuning
                    if hasattr(tuner.oracle, 'get_space'):
                        tuning_time = tuner.oracle.get_space().get('tuning_time', 0)
                    
                    if total_trials > 0:
                        avg_time_per_trial = tuning_time / total_trials
                        
            except Exception as e:
                logger.warning(f"⚠️ Tidak dapat mengambil informasi dari tuner: {str(e)}")
        
        logger.info(f"⏱️  Total Waktu Tuning: {tuning_time:.2f} detik")
        logger.info(f"🔢 Total Trials: {total_trials}")
        logger.info(f"⏱️  Rata-rata Waktu per Trial: {avg_time_per_trial:.2f} detik")
        
        # Simpan metrics efisiensi
        efficiency_metrics = {
            'method': method_name,
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'total_tuning_time': tuning_time,
            'total_trials': total_trials,
            'avg_time_per_trial': avg_time_per_trial,
            'best_accuracy': results['test_accuracy'],
            'best_precision': results['precision'],
            'best_recall': results['recall'],
            'best_f1_score': results['f1_score'],
            'evaluation_time': results['evaluation_time'],
            'num_samples': results['num_samples'],
            'num_classes': results['num_classes']
        }
        
        efficiency_file = f'{method_name.lower().replace(" ", "_")}_efficiency_metrics.json'
        with open(efficiency_file, 'w') as f:
            json.dump(efficiency_metrics, f, indent=2)
        
        logger.info(f"✅ Metrics efisiensi disimpan: {efficiency_file}")
        
        return efficiency_metrics
        
    except Exception as e:
        logger.error(f"❌ Error dalam analisis efisiensi: {str(e)}")
        return None

def generate_efficiency_report(efficiency_metrics, method_name):
    """Generate laporan efisiensi yang informatif"""
    try:
        report = f"""
# LAPORAN EFISIENSI KOMPUTASIONAL - {method_name.upper()}

## 📊 METRICS EFISIENSI

- **Metode:** {method_name}
- **Total Waktu Tuning:** {efficiency_metrics['total_tuning_time']:.2f} detik
- **Total Trials:** {efficiency_metrics['total_trials']}
- **Rata-rata Waktu per Trial:** {efficiency_metrics['avg_time_per_trial']:.2f} detik
- **Waktu Evaluasi:** {efficiency_metrics['evaluation_time']:.2f} detik

## 🎯 HASIL PERFORMANCE

- **Best Accuracy:** {efficiency_metrics['best_accuracy']:.4f}
- **Best Precision:** {efficiency_metrics['best_precision']:.4f}
- **Best Recall:** {efficiency_metrics['best_recall']:.4f}
- **Best F1-Score:** {efficiency_metrics['best_f1_score']:.4f}

## 📈 ANALISIS EFISIENSI

- **Efisiensi (Accuracy/Time):** {efficiency_metrics['best_accuracy']/efficiency_metrics['total_tuning_time']:.6f} (jika tuning_time > 0)
- **Jumlah Sampel:** {efficiency_metrics['num_samples']}
- **Jumlah Kelas:** {efficiency_metrics['num_classes']}

## ⏰ TIMESTAMP

- **Waktu Evaluasi:** {efficiency_metrics['timestamp']}
"""
        
        report_file = f'{method_name.lower().replace(" ", "_")}_efficiency_report.md'
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"✅ Laporan efisiensi disimpan: {report_file}")
        
    except Exception as e:
        logger.error(f"❌ Error dalam generate laporan efisiensi: {str(e)}")

def main():
    """Main function untuk testing"""
    logger.info("🚀 Testing evaluation_complete module...")
    
    # Test reproducibility
    setup_reproducibility()
    logger.info("✅ Reproducibility test berhasil")
    
    logger.info("✅ Module evaluation_complete siap digunakan")

if __name__ == "__main__":
    main() 