# =========== EVALUASI LENGKAP BASELINE MODEL ===========
"""
Script untuk evaluasi lengkap baseline model klasifikasi citra batik.

Author: [Nama Anda]
Date: [Tanggal]
Version: 1.0

Description:
    Script ini melakukan evaluasi komprehensif pada baseline model
    klasifikasi citra batik menggunakan MobileNetV2. Evaluasi mencakup
    accuracy, precision, recall, F1-score, dan confusion matrix.

Requirements:
    - TensorFlow 2.x
    - scikit-learn
    - matplotlib
    - seaborn
    - numpy
    - pandas
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
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('baseline_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def setup_reproducibility(seed=42):
    """Setup reproducibility untuk hasil yang konsisten"""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"Reproducibility setup dengan seed: {seed}")

def validate_paths():
    """Validasi path yang diperlukan"""
    required_paths = [
        'src/models/hyperparameterTuning/randomSearch/final_tuned_model.keras',
        'data/splits/dataset_split/test'
    ]
    
    missing_paths = []
    for path in required_paths:
        if not os.path.exists(path):
            missing_paths.append(path)
    
    if missing_paths:
        logger.error(f"Path yang diperlukan tidak ditemukan: {missing_paths}")
        return False
    
    logger.info("Semua path valid")
    return True

def load_model_safely(model_path):
    """Load model dengan error handling yang baik"""
    try:
        model = load_model(model_path)
        logger.info(f"✅ Model berhasil dimuat dari: {model_path}")
        return model
    except Exception as e:
        logger.error(f"❌ Gagal memuat model dari {model_path}: {str(e)}")
        return None

def create_test_generator(test_dir, img_size=(160, 160), batch_size=4):
    """Buat test generator dengan validasi"""
    try:
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=img_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=False,
            seed=42
        )
        
        logger.info(f"✅ Test generator berhasil dibuat dengan {test_generator.samples} sampel")
        return test_generator
    except Exception as e:
        logger.error(f"❌ Gagal membuat test generator: {str(e)}")
        return None

def evaluate_model_comprehensive(model, test_generator, class_names):
    """Evaluasi model secara komprehensif"""
    start_time = time.time()
    
    logger.info("Memulai evaluasi model...")
    
    # Prediksi
    logger.info("Melakukan prediksi pada test set...")
    y_pred = model.predict(test_generator, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = test_generator.classes
    
    # Metrics dasar
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)
    
    # Classification Report
    report = classification_report(y_true, y_pred_classes, 
                                target_names=class_names, 
                                output_dict=True)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred_classes, average='macro')
    
    evaluation_time = time.time() - start_time
    
    # Print results
    logger.info(f"\n📊 HASIL EVALUASI LENGKAP:")
    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Macro Avg Precision: {precision:.4f}")
    logger.info(f"Macro Avg Recall: {recall:.4f}")
    logger.info(f"Macro Avg F1-Score: {f1:.4f}")
    logger.info(f"Waktu Evaluasi: {evaluation_time:.2f} detik")
    
    return {
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
    }

def create_confusion_matrix_visualization(cm, class_names, method_name="Baseline"):
    """Buat visualisasi confusion matrix yang informatif"""
    plt.figure(figsize=(20, 16))
    
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
    plt.savefig(f'src/models/hyperparameterTuning/randomSearch/{method_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    logger.info(f"✅ Confusion matrix disimpan: {method_name.lower().replace(' ', '_')}_confusion_matrix.png")

def save_evaluation_results(results, method_name="Baseline"):
    """Simpan hasil evaluasi dengan metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Tambahkan metadata
    results_with_metadata = {
        'method': method_name,
        'timestamp': timestamp,
        'model_path': 'src/models/hyperparameterTuning/randomSearch/final_tuned_model.keras',
        'dataset_path': 'data/splits/dataset_split/test',
        'evaluation_metrics': results,
        'model_summary': {
            'total_params': results.get('total_params', 0),
            'trainable_params': results.get('trainable_params', 0),
            'non_trainable_params': results.get('non_trainable_params', 0)
        }
    }
    
    # Simpan hasil evaluasi
    output_file = f'src/models/hyperparameterTuning/randomSearch/{method_name.lower().replace(" ", "_")}_evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_with_metadata, f, indent=2)
    
    logger.info(f"✅ Hasil evaluasi disimpan: {output_file}")
    
    # Simpan metrics efisiensi
    efficiency_metrics = {
        'method': method_name,
        'timestamp': timestamp,
        'total_tuning_time': 0,  # Baseline tidak ada tuning
        'total_trials': 0,
        'avg_time_per_trial': 0,
        'evaluation_time': results['evaluation_time'],
        'best_accuracy': results['test_accuracy'],
        'best_precision': results['precision'],
        'best_recall': results['recall'],
        'best_f1_score': results['f1_score'],
        'num_samples': results['num_samples'],
        'num_classes': results['num_classes']
    }
    
    efficiency_file = f'src/models/origin/{method_name.lower().replace(" ", "_")}_efficiency_metrics.json'
    with open(efficiency_file, 'w') as f:
        json.dump(efficiency_metrics, f, indent=2)
    
    logger.info(f"✅ Metrics efisiensi disimpan: {efficiency_file}")
    
    return results_with_metadata

def evaluate_baseline_model():
    """Evaluasi lengkap baseline model dengan best practice"""
    logger.info("🚀 Memulai evaluasi baseline model...")
    
    # Setup reproducibility
    setup_reproducibility()
    
    # Validasi paths
    if not validate_paths():
        logger.error("❌ Validasi path gagal. Evaluasi dihentikan.")
        return None
    
    # Load model
    model = load_model_safely('src/models/hyperparameterTuning/randomSearch/final_tuned_model.keras')
    if model is None:
        return None
    
    # Buat test generator
    test_generator = create_test_generator('./data/splits/dataset_split/test')
    if test_generator is None:
        return None
    
    # Dapatkan nama kelas
    class_names = list(test_generator.class_indices.keys())
    logger.info(f"✅ Ditemukan {len(class_names)} kelas")
    
    # Evaluasi model
    results = evaluate_model_comprehensive(model, test_generator, class_names)
    
    # Buat visualisasi
    create_confusion_matrix_visualization(
        np.array(results['confusion_matrix']), 
        class_names, 
        "Baseline"
    )
    
    # Simpan hasil
    final_results = save_evaluation_results(results, "Baseline")
    
    logger.info("✅ Evaluasi baseline model selesai!")
    return final_results

if __name__ == "__main__":
    try:
        results = evaluate_baseline_model()
        if results:
            logger.info("🎉 Evaluasi baseline berhasil diselesaikan!")
        else:
            logger.error("❌ Evaluasi baseline gagal!")
    except Exception as e:
        logger.error(f"❌ Error dalam evaluasi baseline: {str(e)}")
        raise 