# proyecto_tfm_2025

## :office: Introducción

Este proyecto tiene como propósito aprovechar los datos que los distintos clubes de futbol europeo registran sobre sus jugadores para realizar un análisis del rendimiento del equipo. El objetivo es identificar tanto los puntos fuertes como las áreas que requieren mejora. Para ello, se analizarán aspectos como los partidos ganados, los goles anotados y la eficiencia defensiva.

El objetivo final es generar un análisis de rendimiento que permita predecir, con base en los datos históricos, cómo podría desempeñarse un equipo en futuros partidos o competiciones, así como proponer estrategias de mejora.

## :open_file_folder: Estructura

- ETL
    - ingesta_spark.ipynb
    - script_modelo.ipynb
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

    ```
    +---+-----------------+----------+------------+---+
    |...|             type|country_id|country_name|...|
    +---+-----------------+----------+------------+---+
    |...|international_cup|        -1|        NULL|...|
    |...|international_cup|        -1|        NULL|...|
    |...|            other|        -1|        NULL|...|
    |...|international_cup|        -1|        NULL|...|
    ```

    - En la tabla *game_lineups*, se detectaron filas anómalas que probablemente resultaron de un error al descargar el archivo.

    ```
    +--------------------+----+-------+---------+-------+-----------+----+--------+------+------------+
    |     game_lineups_id|date|game_id|player_id|club_id|player_name|type|position|number|team_captain|
    +--------------------+----+-------+---------+-------+-----------+----+--------+------+------------+
    |                  77|NULL|   NULL|     NULL|   NULL|       NULL|NULL|    NULL|  NULL|        NULL|
    |                  77|NULL|   NULL|     NULL|   NULL|       NULL|NULL|    NULL|  NULL|        NULL|
    |                  77|NULL|   NULL|     NULL|   NULL|       NULL|NULL|    NULL|  NULL|        NULL|
    ```

    - En la tabla *appearances*, se eliminaron registros con id = -1.

    ```
    +--------------+-------+---------+--------------+----------------------+----------+-----------+
    | appearance_id|game_id|player_id|player_club_id|player_current_club_id|      date|player_name|
    +--------------+-------+---------+--------------+----------------------+----------+-----------+
    |3084062_380365|3084062|   380365|         16486|                    -1|2018-09-05|       NULL|
    |3084059_411294|3084059|   411294|          3302|                    -1|2018-09-11|       NULL|
    |3084057_255495|3084057|   255495|         11596|                    -1|2018-09-12|       NULL|
    |3102749_380365|3102749|   380365|         16486|                    -1|2018-09-12|       NULL|
    |3106648_255495|3106648|   255495|         11596|                    -1|2018-10-17|       NULL|
    |3118604_411294|3118604|   411294|          3302|                    -1|2018-12-05|       NULL|
    +--------------+-------+---------+--------------+----------------------+----------+-----------+
    ```

    - En la tabla *games*, se rellenó los valores nulos de las columnas *home_club_position* y *away_club_position* para competiciones que no usan ranking.

    ```
    +-------+--------------+------+--------------------+---+------------------+------------------+---+-----------------+
    |game_id|competition_id|season|               round|...|home_club_position|away_club_position|...| competition_type|
    +-------+--------------+------+--------------------+---+------------------+------------------+---+-----------------+
    |2382266|           CDR|  2013|   4th round 2nd leg|...|       Sin Ranking|       Sin Ranking|...|     domestic_cup|
    |2428286|           GRP|  2013|Quarter-Finals 1s...|...|       Sin Ranking|       Sin Ranking|...|     domestic_cup|
    |2453103|          UKR1|  2013|        29. Matchday|...|                11|                 4|...|  domestic_league|
    |2492457|            EL|  2014|             group L|...|       Sin Ranking|       Sin Ranking|...|international_cup|
    |2501174|           GRP|  2014| First Round 1st leg|...|       Sin Ranking|       Sin Ranking|...|     domestic_cup|
    |2518602|           GR1|  2014|        24. Matchday|...|                12|                18|...|  domestic_league|
    ```

    - En la tabla *players*, se rellenó los valores nulos de la columna *first_name* para los casos de jugadores con apodos.
    
    ```
    +---------+----------+------------+------------+---+
    |player_id|first_name|   last_name|        name|...|
    +---------+----------+------------+------------+---+
    |       77|      NULL|       Lúcio|       Lúcio|...|
    |      109|      NULL|        Dedê|        Dedê|...|
    |     1426|      NULL|       Cacau|       Cacau|...|
    ```

Para que el proceso de limpieza fuese más rápido se definieron varias funciones que se utilizan a lo largo del código:
- **mostrar_sumario()**: muestra un sumario de las métricas de la tabla.
- **contar_nulos_por_columna()**: muestra los valores nulos de cada columna de la tabla.
- **duplicados_por_columna()**: muestra si hay valores duplicados en una columna y cuántas veces aparecen.
- **formatear_formaciones()**: esta función se usa exclusivamente para el conjunto de datos *games*, aplica un formato uniforme a las formaciones de los equipos.
- **get_country()**: esta función se usa exclusivamente para el conjunto de datos *players*, mapea los valores de la columna *city_of_birth* a su respectivo país.

## Instrucciones Modelado (script_modelo.ipynb)
Este notebook implementa un proceso ETL utilizando PySpark para transformar y modelar datos futbolísticos en un **esquema en estrella**, facilitando su análisis en herramientas como Power BI. A continuación se describen los pasos principales:

### 🔧 1. Configuración e Inicialización
- Se monta Google Drive para acceder a los archivos de entrada.
- Se crea una sesión de Spark con `SparkSession`.
    ```Python
    from google.colab import drive
    drive.mount('/content/drive')
    
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("StarSchemaETL").getOrCreate()
    ```
### 📂 2. Carga de Datos
- Se cargan archivos CSV limpios con información sobre competiciones, clubes, jugadores, partidos, apariciones y valuaciones de jugadores.
  ```Python
  competitions = spark.read.option("header", True).csv(competitions_file)
  clubs = spark.read.option("header", True).csv(clubs_file)
  players = spark.read.option("header", True).csv(players_file)
  games = spark.read.option("header", True).csv(games_file)
  appearances = spark.read.option("header", True).csv(appearances_file)
  player_valuations = spark.read.option("header", True).csv(player_valuations_file)
  ```

### 🧱 3. Construcción de Tablas Dimensionales
Se crean y exportan las siguientes dimensiones en formato CSV:
- **`dim_competition`**: Información sobre competiciones.
- **`dim_club`**: Detalles de los clubes participantes.
- **`dim_player`**: Datos demográficos y posicionales de los jugadores.
- **`dim_game`**: Información básica sobre los partidos.
- **`dim_season`**: Rango de fechas por temporada, calculado manualmente.

    ```Python
    # Dimensión Competición
    competitions.select("competition_id", "name", "type", "country_name") \
    .coalesce(1).write.mode("overwrite").csv(output_dir + "dim_competition.csv", header=True)

    # Dimensión Club
    clubs.select("club_id", "name", "domestic_competition_id", "squad_size",
             "average_age", "foreigners_number", "national_team_players", "stadium_name") \
    .coalesce(1).write.mode("overwrite").csv(output_dir + "dim_club.csv", header=True)

    # Dimensión Jugador
    players.select("player_id", "name", "position", "sub_position", "date_of_birth", "height_in_cm") \
        .coalesce(1).write.mode("overwrite").csv(output_dir + "dim_player.csv", header=True)
    
    # Dimensión Partido
    games.select("game_id", "date", "home_club_id", "away_club_id",
                 "home_club_goals", "away_club_goals", "round", "referee") \
        .coalesce(1).write.mode("overwrite").csv(output_dir + "dim_game.csv", header=True)
    
    # Dimensión Temporada (dim_season)
    # Se asume que cada temporada inicia el 1 de julio y termina el 30 de junio del siguiente año
    seasons = games.select("season").distinct().dropna().withColumn("season", col("season").cast("int"))
    dim_season = seasons.withColumn("season_year", col("season").cast("string").substr(1, 4)) \
                        .withColumn("season_start_date", concat(col("season_year"), lit("-07-01"))) \
                        .withColumn("next_season_year", (col("season") + 1).cast("string").substr(1, 4)) \
                        .withColumn("season_end_date", concat(col("next_season_year"), lit("-06-30"))) \
                        .drop("season_year", "next_season_year")
    
    dim_season.coalesce(1).write.mode("overwrite").csv(output_dir + "dim_season.csv", header=True)

    ```

### 📊 4. Creación de la Tabla de Hechos (`fact_performance`)
- Se transforma la tabla `appearances` para normalizar las métricas de rendimiento (goles, asistencias, minutos, tarjetas).
- Se une con `games` para incorporar fecha y temporada.
- Se empareja cada rendimiento con la **valuación de mercado más cercana en el tiempo**, usando una ventana particionada por jugador y partido.
- Se genera un ID único para cada fila.
    ```Python
    # Selección de columnas finales de la tabla de hechos
    fact_performance = ranked.selectExpr(
        "uuid() as performance_id",
        "game_id",
        "player_id",
        "club_id",
        "competition_id",
        "season",
        "goals",
        "assists",
        "minutes_played",
        "yellow_cards",
        "red_cards",
        "market_value_in_eur"
    )

    # Guardado como único archivo CSV
    fact_performance.coalesce(1).write.mode("overwrite").csv(output_dir + "fact_performance.csv", header=True)
    ```

### 💾 5. Exportación
- Todas las tablas (dimensiones y hechos) se guardan en formato CSV, listas para ser utilizadas en un modelo de datos de análisis.
  

Este proceso facilita la integración con herramientas de BI, permitiendo consultas analíticas eficientes y modelado multidimensional a partir de datos futbolísticos históricos.

## IA

## Descripción General

Esta parte del proyecto proyecto consta de dos componentes principales diseñados para predecir resultados relacionados con el fútbol utilizando técnicas de aprendizaje automático:

1.  **Predictor de Talento Joven**: Identifica jugadores de fútbol jóvenes prometedores basándose en sus métricas de rendimiento y valor de mercado.
    
2.  **Predictor de Resultados de Partidos**: Predice el resultado de partidos de fútbol basándose en estadísticas de equipos y jugadores.
    

Se utilizan conjuntos de datos que contienen información de jugadores y partidos, procesados mediante diversos modelos de aprendizaje automático, incluyendo Random Forest, Regresión Logística, SVM, Naive Bayes y Redes Neuronales. La aplicación está construida con Streamlit para una interfaz web interactiva.

## Archivos

-   **App.py**: El script principal de la aplicación que implementa la interfaz web de Streamlit para ambos predictores: Talento Joven y Resultados de Partidos. Incluye carga de datos, preprocesamiento, entrenamiento de modelos y funcionalidades de visualización.
    
-   **ModeladoTFM.ipynb**: Un cuaderno de Jupyter que contiene la exploración inicial de datos, preprocesamiento y entrenamiento de modelos para el Predictor de Resultados de Partidos. Sirve como prototipo para la implementación en App.py.
    

## Requisitos

-   **Python 3.8+**
    
-   **Librerías**:
    
    ```bash
    pip install pandas numpy streamlit plotly sklearn tensorflow imblearn joblib keras-tuner
    ```
    
-   **Conjuntos de Datos**: Los siguientes archivos CSV deben estar en la carpeta especificada (por defecto, ./dataset):
    
    -   appearances.csv
        
    -   clubs.csv
        
    -   competitions.csv
        
    -   game_events.csv
        
    -   game_lineups.csv
        
    -   games.csv
        
    -   player_valuations.csv
        
    -   players.csv
        
    -   transfers.csv
        

## Estructura del Proyecto

### App.py

-   **Interfaz de Usuario**: Utiliza Streamlit con dos pestañas:
    
    -   **Predictor de Talento Joven**: Permite filtrar jugadores por edad y partidos jugados, seleccionar modelos (Random Forest, Regresión Logística, SVM, Naive Bayes, Red Neuronal), y personalizar parámetros de la red neuronal.
        
    -   **Predictor de Resultados de Partidos**: Permite seleccionar equipos, fecha del partido e ID, y utiliza un modelo Random Forest para predecir resultados.
        
-   **Funcionalidades**:
    
    -   Carga y preprocesamiento de datos.
        
    -   Entrenamiento de modelos con métricas de rendimiento (precisión, recall, F1-score, ROC-AUC).
        
    -   Visualizaciones interactivas (histogramas, curvas ROC, gráficos de radar) usando Plotly.
        
    -   Balanceo de clases con SMOTE para el predictor de talento.
        
    -   Predicciones para jugadores destacados y partidos específicos.
        
-   **Configuración**:
    
    -   Ruta de datos ajustable en la barra lateral.
        
    -   Filtros para edad máxima (18-25) y partidos mínimos (10-50).
        
    -   Parámetros ajustables para modelos, como número de árboles (Random Forest) o capas neuronales (Red Neuronal).
        

### ModeladoTFM.ipynb

-   **Propósito**: Prototipo inicial para el Predictor de Resultados de Partidos.
    
-   **Contenido**:
    
    -   Carga de datos desde Google Drive.
        
    -   Preprocesamiento de datos, incluyendo cálculo de estadísticas de equipos (goles promedio, asistencias) y jugadores.
        
    -   Entrenamiento de un modelo Random Forest con evaluación de métricas (precisión, reporte de clasificación).
        
    -   Visualización de la importancia de características.
        
    -   Función para predecir resultados de partidos específicos.
        
-   **Ejemplo de Predicción**: Predice un partido entre Deportivo de La Coruña y Real Madrid con un 100% de probabilidad de victoria para el equipo visitante.
    
-   **Diferencias con App.py**: El cuaderno es más exploratorio, mientras que App.py está optimizado para una interfaz de usuario interactiva y modular.
    

## Instalación

1.  Clona el repositorio o descarga los archivos.
    
2.  Instala las dependencias:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3.  Coloca los archivos CSV en la carpeta ./dataset o ajusta la ruta en la interfaz de Streamlit.
    
4.  Ejecuta la aplicación:
    
    ```bash
    streamlit run App.py
    ```
    

## Uso

1.  **Iniciar la Aplicación**:
    
    -   Ejecuta streamlit run App.py y accede a la interfaz web en tu navegador (normalmente http://localhost:8501).
        
2.  **Predictor de Talento Joven**:
    
    -   Configura la ruta de los datos en la barra lateral.
        
    -   Ajusta la edad máxima (18-25) y el número mínimo de partidos (10-50).
        
    -   Selecciona modelos para entrenar (por defecto, Random Forest y Red Neuronal).
        
    -   Personaliza parámetros de la red neuronal (capas, neuronas, dropout, etc.).
        
    -   Haz clic en "Entrenar Modelos de Talento" para ver métricas, visualizaciones y los jugadores con mayor potencial.
        
3.  **Predictor de Resultados de Partidos**:
    
    -   Configura parámetros del modelo Random Forest (número de árboles, profundidad máxima).
        
    -   Selecciona equipos local y visitante, fecha e ID del partido.
        
    -   Haz clic en "Entrenar Modelo de Partidos" para entrenar el modelo.
        
    -   Usa "Predecir Resultado" para obtener la predicción del partido con probabilidades.
        

## Características Clave

-   **Datos Utilizados**:
    
    -   Estadísticas de jugadores: goles, asistencias, minutos jugados, tarjetas, posición, valor de mercado.
        
    -   Estadísticas de equipos: tamaño de plantilla, edad promedio, porcentaje de extranjeros, jugadores de selección, capacidad del estadio, goles promedio.
        
-   **Modelos**:
    
    -   Random Forest, Regresión Logística, SVM, Naive Bayes para talento joven.
        
    -   Random Forest para resultados de partidos.
        
    -   Red Neuronal personalizable con Keras para talento joven.
        
-   **Visualizaciones**:
    
    -   Histogramas de distribución de edad.
        
    -   Gráficos de radar para comparar métricas de modelos.
        
    -   Curvas ROC para evaluar el rendimiento.
        
    -   Tablas de jugadores destacados y métricas de rendimiento.
        
-   **Optimizaciones**:
    
    -   Procesamiento paralelo para estadísticas de equipos.
        
    -   Balanceo de clases con SMOTE.
        
    -   Caché de datos con @st.cache_data en Streamlit.
        

## Ejemplo de Resultados

-   **Talento Joven**: Identifica jugadores como Lamine Yamal o Jamal Musiala como de alto potencial basándose en sus estadísticas y valor de mercado.
    
-   **Resultados de Partidos**: Predice un partido entre Deportivo de La Coruña y Real Madrid con un 100% de probabilidad de victoria para Real Madrid (basado en el cuaderno).
    

## Limitaciones

-   Requiere datos completos y actualizados para predicciones precisas.
    
-   La red neuronal puede ser computacionalmente intensiva con muchas capas o épocas.
    
-   Las predicciones de partidos dependen de estadísticas históricas, lo que puede no capturar factores como lesiones recientes o cambios tácticos.

