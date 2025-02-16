import cv2
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MASKS_DIR = os.path.join(BASE_DIR, "../data/processed/reconstructed_video/masks/")
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../data/ground_truth/")

os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

def generate_ground_truth():
    print("Gerando máscaras para ground truth...")

    for filename in sorted(os.listdir(MASKS_DIR)):
        if not filename.endswith(".png"):
            continue
        
        img_path = os.path.join(MASKS_DIR, filename)
        output_path = os.path.join(GROUND_TRUTH_DIR, filename)

        # Verificar se a máscara já foi gerada
        if os.path.exists(output_path):
            print(f"Máscara {filename} já existe. Pulando...")
            continue

        # Carregar a máscara original
        mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Erro. Falha ao carregar {filename}. Pulando...")
            continue

        # Aplicar threshold para binarização
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        cv2.imwrite(output_path, binary_mask)
        print(f"Ground Truth gerada para {filename}")

    print(f"Todas as máscaras foram processadas e salvas em: {GROUND_TRUTH_DIR}")

if __name__ == "__main__":
    generate_ground_truth()
