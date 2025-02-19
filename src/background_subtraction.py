import cv2
import numpy as np
import os
import time
import matplotlib.pyplot as plt

# Diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, '../data/raw/reconstructed_video.mp4')

RAW_FRAMES_DIR = os.path.join(BASE_DIR, '../data/raw/frames')
MASKS_DIR = os.path.join(BASE_DIR, '../data/processed/reconstructed_video/masks')
TRACKED_DIR = os.path.join(BASE_DIR, '../data/processed/reconstructed_video/tracked')
DEBUG_DIR = os.path.join(BASE_DIR, '../data/debug')
LOG_FILE = os.path.join(DEBUG_DIR, "debug_log.txt")
GRAY_DIR = os.path.join(BASE_DIR, '../data/processed/reconstructed_video/gray_frames')
FILTERED_DIR = os.path.join(BASE_DIR, '../data/processed/reconstructed_video/filtered_frames')

# Criar diretórios se não existirem
os.makedirs(RAW_FRAMES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(TRACKED_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(GRAY_DIR, exist_ok=True)
os.makedirs(FILTERED_DIR, exist_ok=True)

# Verificar se a pasta DEBUG foi criada
if not os.path.exists(DEBUG_DIR):
    print("❌ ERRO: A pasta DEBUG_DIR não foi criada corretamente!")

# Inicializar o Background Subtractor (MOG2)
fgbg = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40, detectShadows=False)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_num = 0
start_time = time.time()

video_original = cv2.VideoWriter(os.path.join(BASE_DIR, '../data/processed/reconstructed_video/original.avi'),
                                 cv2.VideoWriter_fourcc(*'XVID'), 20, 
                                 (int(cap.get(3)), int(cap.get(4))))

video_tracking = cv2.VideoWriter(os.path.join(BASE_DIR, '../data/processed/reconstructed_video/tracking.avi'),
                                 cv2.VideoWriter_fourcc(*'XVID'), 20, 
                                 (int(cap.get(3)), int(cap.get(4))))

video_filtered = cv2.VideoWriter(os.path.join(BASE_DIR, '../data/processed/reconstructed_video/filtered.avi'),
                                 cv2.VideoWriter_fourcc(*'XVID'), 20, 
                                 (int(cap.get(3)), int(cap.get(4))))

video_mog2 = cv2.VideoWriter(os.path.join(BASE_DIR, '../data/processed/reconstructed_video/mog2.avi'),
                             cv2.VideoWriter_fourcc(*'XVID'), 20, 
                             (int(cap.get(3)), int(cap.get(4))))

# Função de melhorias na segmentação
def apply_morphology(mask, kernel_size=5):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Preenche buracos
    mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)  # Remove pequenos ruídos
    return mask_opened

def apply_filter(mask, method='median', kernel_size=5):
    if method == 'median':
        return cv2.medianBlur(mask, kernel_size)
    elif method == 'gaussian':
        return cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    return mask

def remove_small_regions(mask, min_size=500):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    mask_filtered = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask_filtered[labels == i] = 255
    return mask_filtered

# Criar log de depuração
with open(LOG_FILE, "w") as log_file:
    log_file.write("Frame, Pixels Ativos (Antes), Pixels Ativos (Depois)\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"📌 Processando frame {frame_num}...")

    # Converter para tons de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(GRAY_DIR, f"gray_{frame_num:04d}.png"), gray)

    # Aplicar subtração de fundo (MOG2)
    fgmask = fgbg.apply(gray)

    cv2.imwrite(os.path.join(FILTERED_DIR, f"mog2_{frame_num:04d}.png"), fgmask)

    # Contagem de pixels ativos antes das melhorias
    pixels_antes = np.sum(fgmask > 0)

    # Aplicar melhorias
    fgmask = cv2.medianBlur(fgmask, 5)
    kernel = np.ones((5, 5), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

    fgmask = apply_morphology(fgmask)
    fgmask = apply_filter(fgmask, method='median')
    fgmask = remove_small_regions(fgmask)

    # Contagem de pixels ativos depois das melhorias
    pixels_depois = np.sum(fgmask > 0)

    # Salvar informações no log
    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"{frame_num}, {pixels_antes}, {pixels_depois}\n")

    # Salvar máscara final
    mask_filename = os.path.join(MASKS_DIR, f"mask_{frame_num:04d}.png")
    saved_mask = cv2.imwrite(mask_filename, fgmask)

    # Salvar debug das máscaras intermediárias
    debug_filename = os.path.join(DEBUG_DIR, f"mask_before_{frame_num:04d}.png")
    saved_debug = cv2.imwrite(debug_filename, fgmask)

    # Teste: salvar a máscara em outro diretório
    test_filename = os.path.join(MASKS_DIR, f"test_mask_{frame_num:04d}.png")
    saved_test = cv2.imwrite(test_filename, fgmask)

    # Aplicar rastreamento de objetos
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tracked_frame = frame.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 300:  # Evitar falsos positivos
            cv2.rectangle(tracked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Salvar rastreamento
    tracked_filename = os.path.join(TRACKED_DIR, f"tracked_{frame_num}.png")
    saved_tracked = cv2.imwrite(tracked_filename, tracked_frame)

    # Salvar imagem de depuração do rastreamento
    debug_tracked_filename = os.path.join(DEBUG_DIR, f"tracked_debug_{frame_num}.png")
    saved_debug_tracked = cv2.imwrite(debug_tracked_filename, tracked_frame)

    # Criar uma cópia do frame filtrado antes de exibir
    filtered = fgmask.copy()

    # Exibir os diferentes estágios do processamento
    cv2.imshow("Frame Original", frame)
    cv2.imshow("Escala de Cinza", gray)
    cv2.imshow("Filtro Bruto", filtered)
    cv2.imshow("Máscara MOG2", fgmask)

    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


    video_original.write(frame)
    video_tracking.write(tracked_frame)
    video_filtered.write(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))
    video_mog2.write(cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR))

cap.release()
video_original.release()
video_tracking.release()
video_filtered.release()
video_mog2.release()
cv2.destroyAllWindows()

elapsed_time = time.time() - start_time
print(f"\n⏳ Processamento concluído! Tempo total: {elapsed_time:.2f} segundos.")
print(f"📂 Máscaras salvas em: {MASKS_DIR}")
print(f"📂 Rastreamento salvo em: {TRACKED_DIR}")
print(f"📂 Filtros brutos salvos em: {FILTERED_DIR}")
print(f"📂 Escala de cinza salva em: {GRAY_DIR}")
print(f"📂 Vídeos gerados em: {os.path.join(BASE_DIR, '../data/processed/reconstructed_video/')}")
