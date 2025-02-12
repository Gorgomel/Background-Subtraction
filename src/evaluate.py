import cv2
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score, accuracy_score

GROUND_TRUTH_DIR = "data/ground_truth"  # Diretório contendo as máscaras reais
PROCESSED_BASE_DIR = "data/processed"  # Diretório com as segmentações geradas

def load_images_from_folder(folder):
    """Carrega todas as imagens binárias de um diretório."""
    images = {}
    for filename in sorted(os.listdir(folder)):  
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                images[filename] = binary_img
    return images

def compute_metrics(gt_images, predicted_images):
    """Calcula métricas de avaliação (Acurácia, Precisão, Recall, IoU)."""
    accuracies, precisions, recalls, ious = [], [], [], []

    for filename in gt_images:
        if filename in predicted_images:
            gt = gt_images[filename]  # Ground Truth
            pred = predicted_images[filename]  # Máscara segmentada

            # Ajusta dimensões se necessário
            if gt.shape != pred.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Converte para formato binário (0 e 1) para cálculos
            gt_bin = (gt > 127).astype(np.uint8)
            pred_bin = (pred > 127).astype(np.uint8)

            # Flatten para cálculos
            gt_flat = gt_bin.flatten()
            pred_flat = pred_bin.flatten()

            # Métricas
            acc = accuracy_score(gt_flat, pred_flat)
            precision = precision_score(gt_flat, pred_flat, zero_division=1)
            recall = recall_score(gt_flat, pred_flat, zero_division=1)

            intersection = np.logical_and(gt_bin, pred_bin).sum()
            union = np.logical_or(gt_bin, pred_bin).sum()
            iou = intersection / union if union != 0 else 0

            accuracies.append(acc)
            precisions.append(precision)
            recalls.append(recall)
            ious.append(iou)

            print(f"[INFO] {filename}: Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, IoU={iou:.4f}")

    # Resultados médios
    return {
        "Accuracy": np.mean(accuracies),
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "IoU": np.mean(ious)
    }

def evaluate_segmentation(video_name):
    """Executa a avaliação comparando segmentações geradas com a ground truth."""
    processed_dir = os.path.join(PROCESSED_BASE_DIR, video_name)
    
    if not os.path.exists(GROUND_TRUTH_DIR):
        print("[ERRO] Diretório da ground truth não encontrado!")
        return
    
    if not os.path.exists(processed_dir):
        print("[ERRO] Diretório da segmentação processada não encontrado!")
        return

    print(f"[INFO] Avaliando segmentação para o vídeo: {video_name}")
    
    gt_images = load_images_from_folder(GROUND_TRUTH_DIR)
    pred_images = load_images_from_folder(processed_dir)

    metrics = compute_metrics(gt_images, pred_images)

    print("\n==== MÉTRICAS GERAIS ====")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    return metrics

if __name__ == "__main__":
    video_name = "reconstructed_video"  # Nome do vídeo sem extensão
    evaluate_segmentation(video_name)
