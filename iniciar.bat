@echo off
cd /d %~dp0

:: Verifica si existe el entorno virtual
if not exist "venv\" (
    echo Creando entorno virtual...
    python -m venv venv
)

:: Activar el entorno virtual
call venv\Scripts\activate.bat

:: Instalar dependencias si existe requirements.txt
if exist requirements.txt (
    echo Instalando dependencias...
    pip install -r requirements.txt
)

:: Ejecutar el programa principal
echo Iniciando sistema...
python primerpasoYolo.py

pause
