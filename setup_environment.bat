@echo off

set BASE_DIR=%~dp0

set VENV_DIR=%BASE_DIR%venv

echo Verificando se o Python está instalado...
where python >nul 2>nul
if %errorlevel% neq 0 (
    echo Python não encontrado! Instale o Python para continuar.
    exit /b 1
)

echo Criando ambiente virtual...
if not exist "%VENV_DIR%" (
    python -m venv "%VENV_DIR%"
)

echo Ativando ambiente virtual...
call "%VENV_DIR%\Scripts\activate.bat"

echo Atualizando pip...
pip install --upgrade pip

echo Instalando dependências...
pip install -r "%BASE_DIR%requirements.txt"

echo Ambiente configurado com sucesso!
echo Ambiente virtual ativado automaticamente.
echo Agora você pode rodar diretamente: python "%BASE_DIR%src\main.py"
