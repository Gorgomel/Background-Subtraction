import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../data/ground_truth")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "../data/processed/reconstructed_video/masks")
DEBUG_DIR = os.path.join(BASE_DIR, "../data/debug_comparison")

os.makedirs(DEBUG_DIR, exist_ok=True)

def load_images(folder):
    """Carrega todas as imagens binárias de um diretório."""
    images = {}
    
    if not os.path.exists(folder):
        print(f"⚠️ Erro: Diretório não encontrado: {folder}")
        return images

    for filename in sorted(os.listdir(folder)):  
        if filename.endswith(".png"):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                images[filename] = binary_img
            else:
                print(f"⚠️ Erro ao carregar imagem: {filename}")
    return images

# Carregar imagens da Ground Truth e das máscaras geradas
gt_images = load_images(GROUND_TRUTH_DIR)
pred_images = load_images(PREDICTIONS_DIR)

# Verificar se há imagens carregadas
if not gt_images or not pred_images:
    print("⚠️ Erro: Ground truth ou segmentações não foram carregadas corretamente!")
    exit()

# Comparar todas as imagens
for filename in gt_images:
    if filename in pred_images:
        gt = gt_images[filename]
        pred = pred_images[filename]

        # Ajustar dimensões se necessário
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Calcular diferença
        diff = np.abs(gt - pred)

        # Salvar imagens de comparação
        cv2.imwrite(os.path.join(DEBUG_DIR, f"diff_{filename}"), diff)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(gt, cmap="gray")
        plt.title("Ground Truth")

        plt.subplot(1, 3, 2)
        plt.imshow(pred, cmap="gray")
        plt.title("Predição")

        plt.subplot(1, 3, 3)
        plt.imshow(diff, cmap="hot")
        plt.title("Diferença (GT - Predição)")

        plt.savefig(os.path.join(DEBUG_DIR, f"comparison_{filename}"))
        plt.close()

        print(f"✅ Comparação salva: {filename}")
    else:
        print(f"⚠️ Máscara predita não encontrada para {filename}")

print("\n✅ Comparação concluída! Resultados salvos em '/data/debug_comparison/'")
