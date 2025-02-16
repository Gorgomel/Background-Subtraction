import cv2
import numpy as np
import os
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../data/ground_truth")
PROCESSED_BASE_DIR = os.path.join(BASE_DIR, "../data/processed")
RESULTS_DIR = os.path.join(BASE_DIR, "../data/results")  

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_images_from_folder(folder):
    """Carrega todas as imagens binárias de um diretório."""
    images = {}
    
    if not os.path.exists(folder):
        print(f"Erro. Diretório não encontrado: {folder}")
        return images

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
    per_frame_results = []

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

            per_frame_results.append(f"{filename}: Acc={acc:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, IoU={iou:.4f}")

    # Retorna as métricas médias e os resultados por frame
    return {
        "Accuracy": np.mean(accuracies) if accuracies else 0,
        "Precision": np.mean(precisions) if precisions else 0,
        "Recall": np.mean(recalls) if recalls else 0,
        "IoU": np.mean(ious) if ious else 0
    }, per_frame_results

def save_results_to_file(metrics, per_frame_results, elapsed_time):
    """Salva as métricas em um arquivo de texto com UTF-8 para evitar erros no Windows."""
    results_file = os.path.join(RESULTS_DIR, "evaluation_results.txt")

    with open(results_file, "w", encoding="utf-8") as f: 
        f.write("==== MÉTRICAS POR FRAME ====\n")
        for line in per_frame_results:
            f.write(line + "\n")

        f.write("\n==== MÉTRICAS GERAIS ====\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")

        f.write(f"\nTempo total de avaliação: {elapsed_time:.2f} segundos\n")

    print(f"\nResultados salvos em: {results_file}")

def evaluate_segmentation(video_name):
    """Executa a avaliação comparando segmentações geradas com a ground truth."""
    processed_dir = os.path.join(PROCESSED_BASE_DIR, video_name, "masks")  

    if not os.path.exists(GROUND_TRUTH_DIR):
        print("Erro. Diretório da ground truth não encontrado!")
        return
    
    if not os.path.exists(processed_dir):
        print(f"Erro. Diretório das máscaras segmentadas não encontrado: {processed_dir}")
        return

    print(f"Avaliando segmentação para o vídeo: {video_name}")
    
    gt_images = load_images_from_folder(GROUND_TRUTH_DIR)
    pred_images = load_images_from_folder(processed_dir)

    if not gt_images:
        print("Erro. Nenhuma imagem encontrada na ground truth!")
        return

    if not pred_images:
        print("Erro. Nenhuma imagem encontrada na segmentação gerada!")
        return
    
    start_time = time.time()

    metrics, per_frame_results = compute_metrics(gt_images, pred_images)

    end_time = time.time()  
    elapsed_time = end_time - start_time

    print("\n==== MÉTRICAS GERAIS ====")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print(f"\nTempo total de avaliação: {elapsed_time:.2f} segundos")

    save_results_to_file(metrics, per_frame_results, elapsed_time)

    return metrics

if __name__ == "__main__":
    video_name = "reconstructed_video" 
    evaluate_segmentation(video_name)
