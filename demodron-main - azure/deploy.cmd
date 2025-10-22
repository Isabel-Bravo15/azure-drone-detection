@echo off 
echo "Azure Deployment Script for Drone Detection" 
 
:: Crear entorno virtual 
python -m venv venv 
call venv\Scripts\activate.bat 
 
:: Instalar dependencias 
pip install --upgrade pip 
pip install -r requirements_azure.txt 
 
:: Crear directorio para modelos 
if not exist models mkdir models 
 
echo "? Deployment completed" 
