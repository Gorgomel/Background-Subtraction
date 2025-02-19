import os
import sys
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(BASE_DIR, "..", "venv")  # Ajuste se necessário

# Caminhos para os scripts
CREATE_VIDEO_SCRIPT = os.path.join(BASE_DIR, "create_video.py")
BACKGROUND_SUBTRACTION_SCRIPT = os.path.join(BASE_DIR, "background_subtraction.py")
GENERATE_GT_SCRIPT = os.path.join(BASE_DIR, "generate_ground_truth.py")
EVALUATE_SCRIPT = os.path.join(BASE_DIR, "evaluate.py")
COMPARE_MASKS_SCRIPT = os.path.join(BASE_DIR, "compare_masks.py")
VALIDATE_EVALUATION_SCRIPT = os.path.join(BASE_DIR, "validate_evaluation.py")

def is_venv_active():
    """Verifica se o ambiente virtual está ativado."""
    return sys.prefix != sys.base_prefix

def activate_venv():
    """Ativa o ambiente virtual dependendo do sistema operacional."""
    if is_venv_active():
        print("✅ Ambiente virtual já está ativo.")
        return
    
    activate_script = None
    if os.name == "nt":  # Windows
        activate_script = os.path.join(VENV_DIR, "Scripts", "activate")
    else:  # Linux/Mac
        activate_script = os.path.join(VENV_DIR, "bin", "activate")

    if os.path.exists(activate_script):
        print(f"🔄 Ativando o ambiente virtual: {activate_script}")
        subprocess.run(f"source {activate_script}", shell=True, executable="/bin/bash")
    else:
        print("❌ Erro: Ambiente virtual não encontrado! Certifique-se de que o venv está configurado corretamente.")
        sys.exit(1)

def run_script(script_path):
    """Executa um script Python dentro do ambiente virtual."""
    print(f"\n🔄 Executando: {script_path}")
    result = subprocess.run([sys.executable, script_path], text=True)
    if result.returncode == 0:
        print(f"✅ Concluído: {script_path}")
    else:
        print(f"❌ Erro ao executar: {script_path}")

def main():
    """Executa os scripts na sequência correta."""
    
    # Ativar o ambiente virtual antes de executar qualquer script
    activate_venv()

    # Executa a criação do vídeo
    run_script(CREATE_VIDEO_SCRIPT)

    # Executa a subtração de fundo
    run_script(BACKGROUND_SUBTRACTION_SCRIPT)

    # Gera a ground truth
    run_script(GENERATE_GT_SCRIPT)

    # Avalia os resultados
    run_script(EVALUATE_SCRIPT)

    # Pergunta ao usuário sobre a geração de relatórios
    opcao = input("\n📊 Deseja gerar os relatórios de comparação e validação? (s/n): ").strip().lower()
    
    if opcao == 's':
        run_script(COMPARE_MASKS_SCRIPT)
        run_script(VALIDATE_EVALUATION_SCRIPT)

    print("\n🎉 Processo concluído!")

if __name__ == "__main__":
    main()
