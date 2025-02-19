import cv2
import os
import numpy as np
import random
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MASKS_DIR = os.path.join(BASE_DIR, "../data/processed/reconstructed_video/masks/")
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../data/ground_truth/")
DEBUG_DIR = os.path.join(BASE_DIR, "../data/debug_ground_truth/")

os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

LOG_FILE = os.path.join(BASE_DIR, "../data/ground_truth_log.txt")

def generate_ground_truth():
    print("🟢 Gerando máscaras para ground truth...")

    log_data = []

    for filename in sorted(os.listdir(MASKS_DIR)):
        if not filename.endswith(".png"):
            continue
        
        img_path = os.path.join(MASKS_DIR, filename)
        output_path = os.path.join(GROUND_TRUTH_DIR, filename)

        # Verificar se a máscara já foi gerada
        if os.path.exists(output_path):
            continue

        # Carregar a máscara original
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"⚠️ Erro ao carregar {filename}. Pulando...")
            continue

        # Aplicar threshold para binarização
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        # Contar pixels ativos
        num_active_pixels = np.sum(binary_mask == 255)
        total_pixels = binary_mask.size
        percentage_active = (num_active_pixels / total_pixels) * 100

        log_data.append(f"{filename}: {num_active_pixels} pixels brancos ({percentage_active:.2f}%)")

        # Salvar o Ground Truth
        cv2.imwrite(output_path, binary_mask)

    # Salvar log com estatísticas
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(log_data))

    print(f"\n✅ Todas as máscaras foram processadas e salvas em: {GROUND_TRUTH_DIR}")
    print(f"📄 Estatísticas salvas em {LOG_FILE}")

    # Visualizar amostras aleatórias para depuração
    visualize_random_samples()

def visualize_random_samples():
    """Gera imagens comparativas entre GT e máscara original"""
    sample_files = random.sample(os.listdir(MASKS_DIR), min(5, len(os.listdir(MASKS_DIR))))  # Escolher até 5 imagens aleatórias

    for filename in sample_files:
        mask_path = os.path.join(MASKS_DIR, filename)
        gt_path = os.path.join(GROUND_TRUTH_DIR, filename)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        # Verificar se as imagens foram carregadas corretamente
        if mask is None or gt is None:
            continue

        # Calcular diferença
        diff = np.abs(mask - gt)

        # Criar visualização
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(mask, cmap="gray")
        plt.title("Máscara Predita")

        plt.subplot(1, 3, 2)
        plt.imshow(gt, cmap="gray")
        plt.title("Ground Truth")

        plt.subplot(1, 3, 3)
        plt.imshow(diff, cmap="hot")
        plt.title("Diferença (GT - Predição)")

        output_path = os.path.join(DEBUG_DIR, f"comparison_{filename}")
        plt.savefig(output_path)
        plt.close()

    print(f"\n🖼️ Imagens comparativas salvas em {DEBUG_DIR}")

if __name__ == "__main__":
    generate_ground_truth()
