import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/raw")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
OUTPUT_VIDEO = os.path.join(DATA_DIR, "reconstructed_video.mp4")

os.makedirs(FRAMES_DIR, exist_ok=True)

frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith((".jpg", ".png"))])

if not frames:
    print(f"Erro. Nenhuma imagem encontrada em {FRAMES_DIR}")
    exit()

frame_sample = cv2.imread(os.path.join(FRAMES_DIR, frames[0]))

if frame_sample is None:
    print("Erro. Não foi possível carregar a imagem de referência.")
    exit()

height, width, _ = frame_sample.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para MP4
video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30, (width, height))  # FPS = 30

print(f"Criando vídeo a partir de {len(frames)} frames...")

for frame in frames:
    img_path = os.path.join(FRAMES_DIR, frame)
    img = cv2.imread(img_path)

    if img is None:
        print(f"Erro. Falha ao carregar imagem: {img_path}")
        continue

    video.write(img)

video.release()
print(f"Vídeo salvo em {OUTPUT_VIDEO}")
