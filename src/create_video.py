import cv2
import os

# Obtém o caminho absoluto do diretório do script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/raw")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
OUTPUT_VIDEO = os.path.join(DATA_DIR, "reconstructed_video.mp4")

# Criar o diretório de frames caso não exista
os.makedirs(FRAMES_DIR, exist_ok=True)

# Pegar a lista de arquivos, filtrando apenas imagens (.jpg, .png) e ordenando
frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith((".jpg", ".png"))])

if not frames:
    print(f"[ERRO] Nenhuma imagem encontrada em {FRAMES_DIR}")
    exit()

# Carregar um frame para obter as dimensões do vídeo
frame_sample = cv2.imread(os.path.join(FRAMES_DIR, frames[0]))

if frame_sample is None:
    print("[ERRO] Não foi possível carregar a imagem de referência.")
    exit()

height, width, _ = frame_sample.shape

# Criar o objeto de escrita de vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (width, height))  # FPS = 30

print(f"[INFO] Criando vídeo a partir de {len(frames)} frames...")

for frame in frames:
    img_path = os.path.join(FRAMES_DIR, frame)
    img = cv2.imread(img_path)

    if img is None:
        print(f"[ERRO] Falha ao carregar imagem: {img_path}")
        continue

    video.write(img)

video.release()
print(f"[INFO] Vídeo salvo em {OUTPUT_VIDEO}")
