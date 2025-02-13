#!/bin/bash

BASE_DIR=$(dirname $(realpath "$0"))

VENV_DIR="$BASE_DIR/venv"

echo "Verificando se o Python 3 está instalado..."
if ! command -v python3 &> /dev/null
then
    echo "Python 3 não encontrado! Instale o Python 3 para continuar."
    exit 1
fi

echo "Criando ambiente virtual..."
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

echo "Ativando ambiente virtual..."
source "$VENV_DIR/bin/activate"

echo "Atualizando pip..."
pip install --upgrade pip

echo "Instalando dependências..."
pip install -r "$BASE_DIR/requirements.txt"

echo "Ambiente configurado com sucesso!"
echo "Ambiente virtual ativado automaticamente."
echo "Agora você pode rodar diretamente: python $BASE_DIR/src/main.py"
