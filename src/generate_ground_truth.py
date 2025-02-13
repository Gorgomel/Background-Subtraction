import cv2
import os
import numpy as np

# Caminho absoluto do diretório do script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos dos diretórios
MASKS_DIR = os.path.join(BASE_DIR, "../data/processed/reconstructed_video/masks/")
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, "../data/ground_truth/")

# Criar diretório de ground truth se não existir
os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

def refine_mask(mask):
    """Refina a máscara usando operações morfológicas para melhorar a segmentação."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    # Remove pequenos ruídos
    refined = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Preenche pequenos buracos
    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    return refined

def generate_ground_truth():
    """Gera máscaras binárias refinadas para serem usadas como ground truth."""
    
    # Verifica se há máscaras segmentadas disponíveis
    if not os.path.exists(MASKS_DIR) or not os.listdir(MASKS_DIR):
        print("[ERRO] Nenhuma máscara segmentada encontrada! Execute background_subtraction.py primeiro.")
        return

    print("[INFO] Gerando máscaras para ground truth...")

    for filename in sorted(os.listdir(MASKS_DIR)):
        if filename.endswith(".png"):
            img_path = os.path.join(MASKS_DIR, filename)
            mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"[ERRO] Não foi possível carregar {filename}")
                continue

            # Binarizar a máscara (garantindo apenas 0 e 255)
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # Refinar a máscara para melhorar segmentação
            refined_mask = refine_mask(binary_mask)

            # Salvar a máscara refinada em `ground_truth`
            output_path = os.path.join(GROUND_TRUTH_DIR, filename)
            cv2.imwrite(output_path, refined_mask)
    
    print(f"[INFO] Máscaras geradas e salvas em: {GROUND_TRUTH_DIR}")

if __name__ == "__main__":
    generate_ground_truth()
