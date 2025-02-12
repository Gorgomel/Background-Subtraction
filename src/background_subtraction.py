import cv2
import numpy as np
import os
import time

# Diretório base onde as imagens processadas serão armazenadas
BASE_PROCESSED_DIR = "data/processed"

def create_output_directory(video_name):
    """
    Cria um diretório único para armazenar os frames processados,
    com base no nome do vídeo. Se já existir, cria versões numeradas.
    """
    video_name = os.path.splitext(os.path.basename(video_name))[0]  # Remove extensão do vídeo
    output_folder = os.path.join(BASE_PROCESSED_DIR, video_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        return output_folder

    # Se a pasta já existe, cria versões numeradas (_1, _2, etc.)
    version = 1
    while True:
        new_folder = f"{output_folder}_{version}"
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
            return new_folder
        version += 1

def background_subtraction(video_path):
    print(f"[INFO] Iniciando processamento do vídeo: {video_path}")

    output_folder = create_output_directory(video_path)
    print(f"[INFO] Frames serão salvos em: {output_folder}")

    start_time = time.time()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("[ERRO] Não foi possível abrir o vídeo! Verifique o caminho.")
        return

    # Criar subtrator de fundo
    backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=35, detectShadows=True)

    frame_count = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] Fim do vídeo ou erro na leitura do frame.")
            break

        frame_count += 1
        print(f"[INFO] Processando frame {frame_count}...")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)  # Equalização do histograma para melhorar o contraste

        # Aplicação de um desfoque leve para suavizar sem remover detalhes importantes
        blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)

        # Aplicar subtração de fundo
        fg_mask = backSub.apply(blurred)

        # Subtração de quadros consecutivos para evitar perda de detecção
        if prev_frame is not None:
            frame_diff = cv2.absdiff(prev_frame, gray)
            _, frame_diff = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
            fg_mask = cv2.bitwise_or(fg_mask, frame_diff)

        prev_frame = gray  # Atualiza o frame anterior

        # Aplicação de um limiar binário para refinar a máscara
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        # Aplicação de operações morfológicas para reduzir ruído e preencher contornos
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)  # Fecha buracos pequenos
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)   # Remove pequenos ruídos

        # Encontrar contornos
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Ajuste do tamanho mínimo da área para evitar perda de pequenos objetos
        min_area = 500  
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

        # Criar uma cópia do frame original e desenhar os contornos detectados
        frame_contours = frame.copy()
        cv2.drawContours(frame_contours, filtered_contours, -1, (0, 255, 0), 2)

        # Exibir resultados
        cv2.imshow("Original", frame)
        cv2.imshow("Máscara de Fundo", fg_mask)
        cv2.imshow("Contornos Detectados", frame_contours)

        # Salvar frame processado
        frame_name = os.path.join(output_folder, f"frame_{frame_count}.png")
        cv2.imwrite(frame_name, fg_mask)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            print("[INFO] Interrompido pelo usuário.")
            break

    cap.release()
    cv2.destroyAllWindows()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"[INFO] Processamento concluído. Frames salvos em {output_folder}.")
    print(f"[INFO] Tempo total de processamento: {elapsed_time:.2f} segundos.")

if __name__ == "__main__":
    video_path = "data/raw/reconstructed_video.mp4"
    background_subtraction(video_path)
