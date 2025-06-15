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
