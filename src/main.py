import os
import time
from tqdm import tqdm
from create_video import main as run_create_video
from background_subtraction import main as run_background_subtraction
from generate_ground_truth import main as run_generate_ground_truth
from evaluate import evaluate_segmentation

def main():
    print("[INFO] Diretórios organizados!\n")

    tasks = [
        ("Criação do Vídeo", run_create_video),
        ("Segmentação de Fundo", run_background_subtraction),
        ("Geração da Ground Truth", run_generate_ground_truth),
        ("Avaliação da Segmentação", evaluate_segmentation)
    ]

    for task_name, task_function in tqdm(tasks, desc="Executando processos", unit="tarefa"):
        print(f"\n[INFO] Executando: {task_name} ...")
        start_time = time.time()
        task_function()
        elapsed_time = time.time() - start_time
        print(f"[INFO] {task_name} concluído em {elapsed_time:.2f} segundos!")

    print("\n[INFO] Processo completo!")

if __name__ == "__main__":
    main()
