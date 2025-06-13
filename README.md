# proyecto_tfm_2025

## :office: Introducción

Este proyecto tiene como propósito aprovechar los datos que los distintos clubes de futbol europeo registran sobre sus jugadores para realizar un análisis del rendimiento del equipo. El objetivo es identificar tanto los puntos fuertes como las áreas que requieren mejora. Para ello, se analizarán aspectos como los partidos ganados, los goles anotados y la eficiencia defensiva.

El objetivo final es generar un análisis de rendimiento que permita predecir, con base en los datos históricos, cómo podría desempeñarse un equipo en futuros partidos o competiciones, así como proponer estrategias de mejora.

## :open_file_folder: Estructura

- ETL
    - ingesta_spark.ipynb
    - requirements.txt
- Data
- 
- 

## :card_index: Instrucciones para ejecutar la ETL

### :rocket: Requisitos previos:

- Python v3.12
- Java v17
- Crear un virtual environment:

Para evitar conflictos con los paquetes, creamos un entorno virtual para instalar las dependencias
```
# Crear la carpeta para el venv
python3 -m venv .venv
# Activar el venv
source .venv/bin/activate
# Instalar las dependencias
# Esto se debe de ejecutar desde la raiz
python3 -m pip install -r ETL/requirements.txt
```

Si quieres borrar el entorno

```
# Desactivar el venv
deactivate
# Borrar el entorno
rm -r .venv
``` 

### :arrow_down: Ingesta de archivos

Ingestamos los datos directamente desde Kaggle usando la librería **Kagglehub**,esto nos permite tener siempre la versión más reciente de los datos.

```Python
import kagglehub

# Descargar el dataset desde Kaggle Hub
path = kagglehub.dataset_download("davidcariboo/player-scores")

print("Path to dataset files:", path)

# Lista los archivos descargados
for root, _, files in os.walk(path):
    for file in files:
        print(os.path.join(root, file))
```

De los 10 archivos que forman el dataset, sólo utilizamos 9:

1. **clubs.csv**: contiene información sobre los clubes
2. **competitions.csv**: contiene información sobre las diferentes ligas y competiciones
3. **game_events.csv**: contiene información sobre las jugadas realizadas en los partidos
4. **game_lineups.csv**: contiene información sobre las alineaciones de los equipos
5. **appearances.csv**: contiene información sobre del rendimiento de los jugadores en los partidos
6. **player_valuations.csv**: contiene información sobre el valor de los jugadores a lo largo del tiempo
7. **games.csv**: contiene información sobre los partidos
8. **players.csv**: contiene información sobre los jugadores
9. **transfers.csv**: contiene información sobre el movimiento de jugadores entre clubes

El archivo ```club_games.csv``` no tiene información que no esté ya incluida en alguno de los otros archivos, por eso decidimos no usarlo.

### :shower: Limpieza de datos

Durante esta fase se llevaron a cabo varios procesos de limpieza para garantizar su calidad antes del análisis y modelado:
- Eliminar valores nulos en las columnas.
- Eliminar registros duplicados.
- Asignar los tipos correctos a las columnas.
- Identificar y tratar registros con valores no deseados.
- Formatear los valores string para que mantengan uniformidad.
- Prescindir de columnas con datos no relevantes para el análisis.
- Casos particulares:
    - En la tabla *competitions*, se reemplazo el id a los valores con id = -1.
    - En la tabla *game_lineups*, se detectaron filas anómalas que probablemente resultaron de un error al descargar el archivo.
    - En la tabla *appearances*, se eliminaron registros con id = -1.
    - En la tabla *games*, se rellenó los valores nulos de las columnas *home_club_position* y *away_club_position* para competiciones que no usan ranking.
    - En la tabla *players*, se rellenó los valores nulos de la columna *first_name* para los casos de jugadores con apodos.

Para que el proceso de limpieza fuese más rápido se definieron varias funciones que se utilizan a lo largo del código:
- **mostrar_sumario()**: muestra un sumario de las métricas de la tabla.
- **contar_nulos_por_columna()**: muestra los valores nulos de cada columna de la tabla.
- **duplicados_por_columna()**: muestra si hay valores duplicados en una columna y cuántas veces aparecen.
- **formatear_formaciones()**: esta función se usa exclusivamente para el conjunto de datos *games*, aplica un formato uniforme a las formaciones de los equipos.
- **get_country()**: esta función se usa exclusivamente para el conjunto de datos *players*, mapea los valores de la columna *city_of_birth* a su respectivo país.

## Instrucciones Modelado

