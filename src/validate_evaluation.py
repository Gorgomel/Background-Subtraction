import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../data/ground_truth")
PREDICTIONS_DIR = os.path.join(BASE_DIR, "../data/processed/reconstructed_video/masks")

def load_images(folder):
    """Carrega todas as imagens de um diretório."""
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

# Carregar as imagens
gt_images = load_images(GROUND_TRUTH_DIR)
pred_images = load_images(PREDICTIONS_DIR)

# Verificar se há imagens carregadas
if not gt_images or not pred_images:
    print("⚠️ Erro: Ground truth ou segmentações não foram carregadas corretamente!")
    exit()

# Escolher um frame aleatório para análise
random_frame = random.choice(list(gt_images.keys()))

gt = gt_images[random_frame]
pred = pred_images.get(random_frame, None)

if pred is None:
    print(f"⚠️ Erro: Não há predição correspondente para {random_frame}")
    exit()

# Verificar se as imagens têm o mesmo tamanho
if gt.shape != pred.shape:
    pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)

# Comparação pixel a pixel
diff = np.abs(gt - pred)
if np.sum(diff) == 0:
    print("⚠️ As máscaras preditas são idênticas à ground truth!")

# Mostrar valores únicos para verificar problemas
print(f"Valores únicos na Ground Truth: {np.unique(gt)}")
print(f"Valores únicos na Predição: {np.unique(pred)}")

# Contar pixels ativos em cada uma
print(f"Pixels ativos na Ground Truth: {np.sum(gt > 127)}")
print(f"Pixels ativos na Predição: {np.sum(pred > 127)}")

# Exibir imagens para comparação
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

plt.show()
