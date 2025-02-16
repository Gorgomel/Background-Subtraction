import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, '../data/raw/reconstructed_video.mp4')

RAW_FRAMES_DIR = os.path.join(BASE_DIR, '../data/raw/frames')  
PREPROCESSED_DIR = os.path.join(BASE_DIR, '../data/preprocessed')  
MASKS_DIR = os.path.join(BASE_DIR, '../data/processed/reconstructed_video/masks')  
TRACKED_DIR = os.path.join(BASE_DIR, '../data/processed/reconstructed_video/tracked')  
SEGMENTATION_MOG2_DIR = os.path.join(BASE_DIR, '../data/segmentation/mog2')  
SEGMENTATION_MORPH_DIR = os.path.join(BASE_DIR, '../data/segmentation/morphology')  

os.makedirs(RAW_FRAMES_DIR, exist_ok=True)
os.makedirs(PREPROCESSED_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)
os.makedirs(TRACKED_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_MOG2_DIR, exist_ok=True)
os.makedirs(SEGMENTATION_MORPH_DIR, exist_ok=True)

# Background Subtractor (MOG2)
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Processando frame {frame_num}...")

    cv2.imwrite(os.path.join(RAW_FRAMES_DIR, f"frame_{frame_num}.png"), frame)
    cv2.imshow("Frame Original", frame)

    # Converter para tons de cinza (Pré-processamento)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(PREPROCESSED_DIR, f"preprocessed_{frame_num}.png"), gray)
    cv2.imshow("Frame Pré-processado", gray)

    # Aplicar subtração de fundo (MOG2)
    fgmask = fgbg.apply(gray)
    cv2.imshow("Máscara Bruta", fgmask)

    # Aplicar operações morfológicas na máscara segmentada
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    refined_mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel)
    
    # Salvar segmentação MOG2
    cv2.imwrite(os.path.join(SEGMENTATION_MOG2_DIR, f"segmentation_mog2_{frame_num}.png"), refined_mask)
    cv2.imshow("Segmentação MOG2", refined_mask)

    # Aplicar segmentação baseada apenas em morfologia
    morphology_only = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    morphology_only = cv2.morphologyEx(morphology_only, cv2.MORPH_OPEN, kernel)

    cv2.imwrite(os.path.join(SEGMENTATION_MORPH_DIR, f"segmentation_morphology_{frame_num}.png"), morphology_only)
    cv2.imshow("Segmentação Morfológica", morphology_only)

    # Aplicar rastreamento e exibir
    contours, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    tracked_frame = frame.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 500:
            cv2.rectangle(tracked_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Objetos Rastreando", tracked_frame)
    cv2.imwrite(os.path.join(TRACKED_DIR, f"tracked_{frame_num}.png"), tracked_frame)

    # Pressionar q para encerrar o processamento manualmente
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Processamento concluído!")
