# proyecto_tfm_2025

# Instrucciones para ejecutar la ETL
Requisitos previos:
    - Python v3.12
    - Java v17

Crear un virtual environment:

    """
    # Crear la carpeta para el venv
    python3 -m venv .venv
    # Activar el venv
    source .venv/bin/activate
    # Instalar las dependencias
    python3 -m pip install -r ETL/requirements.txt
    """
    Si quieres borrar el entorno
    """
    # Desactivar el venv
    deactivate
    # Borrar el entorno
    rm -r .venv
    """

# Leer los archivos desde Kaggle
Al descargar el dataset desde Kaggle Hub, los archivos se guardan en cache local, para leer luego los archivos con Spark se debe cambiar la ruta a la que devuelva el metodo "kagglehub.dataset_download()"