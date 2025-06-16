# README - Feature Engineering, Embeddings y Modelo Poisson para Análisis de Fútbol

Este notebook realiza un proceso integral de **feature engineering** sobre datos de fútbol para generar métricas relevantes y embeddings latentes que resumen características de países, formaciones, jugadores y equipos. Además, entrena un modelo estadístico **Poisson** para predecir el impacto de unas variables sobre otras.

---

## Contenido principal

- **Generación de features y métricas agregadas** para países, formaciones, jugadores y equipos.
- **Entrenamiento de autoencoders** para crear embeddings latentes compactos que capturan patrones complejos.
- **Entrenamiento y evaluación de un modelo Poisson** con `statsmodels` para modelar el número de goles.
- Exportación de resultados a CSV para su uso en posteriores análisis y modelos.

---

## Archivos generados

- `embeddings/country_features.csv`  
  Métricas agregadas a nivel país con indicadores de desempeño en competiciones internacionales.

- `embeddings/formation_embeddings.csv`  
  Embeddings numéricos que representan formaciones tácticas aprendidas desde datos históricos.

- `embeddings/player_season_embedding.csv`  
  Embeddings y métricas de desempeño individual por jugador y temporada.

- `embeddings/team_season_embedding.csv`  
  Embeddings que resumen características latentes de equipos por temporada.

---

## Modelo Poisson

Se utiliza un modelo de regresión **Poisson** para modelar y predecir la cantidad de goles marcados por equipos, aprovechando las variables procesadas y embeddings. El modelo es entrenado con la librería `statsmodels` y se basa en variables cuidadosamente preprocesadas que incluyen tanto variables numéricas estandarizadas como categóricas codificadas.

---

## Requisitos previos

### Datos necesarios

El notebook asume que tienes descargados y procesados los siguientes archivos de datos, ubicados en el directorio `../Data/`:

- `appearances.csv`
- `clubs.csv`
- `competitions.csv`
- `game_events.csv`
- `game_lineups.csv`
- `games.csv`
- `player_valuations.csv`
- `players.csv`
- `transfers.csv`

Estos datos se obtienen y procesan ejecutando previamente el notebook `../ETL/ingesta_spark.ipynb`.

---

### Directorios

Para que el notebook funcione correctamente, es necesario que estén creados los siguientes directorios:

- `embeddings/`  
  Donde se guardan los archivos CSV con las características y embeddings generados.

- `models/`  
  Donde se guardan los modelos entrenados, como el autoencoder para jugadores y equipos.

---

### Librerías requeridas

Asegúrate de tener instaladas las siguientes librerías:

- `pandas`
- `numpy`
- `seaborn`
- `matplotlib`
- `tensorflow`
- `scikit-learn`
- `statsmodels`

Puedes instalarlas con pip si es necesario:

```bash
pip install pandas numpy seaborn matplotlib tensorflow scikit-learn statsmodels
```

## Flujo general del notebook

1. **Carga y limpieza de datos**  
   Se cargan los datos desde `../Data/` y se realiza un preprocesamiento básico, como el manejo de valores nulos y conversiones.

2. **Feature engineering**  
   Se calculan métricas agregadas a nivel país y equipo, se crean variables indicadoras para datos faltantes y se codifican variables categóricas.

3. **Entrenamiento de autoencoders**  
   Para obtener embeddings latentes de jugadores, equipos y formaciones que capturan relaciones no lineales y patrones ocultos en los datos.

4. **Entrenamiento del modelo Poisson**  
   Modelado estadístico para predecir goles usando variables procesadas y embeddings, con evaluación de los coeficientes y significancia.

5. **Exportación**  
   Guardado de los resultados en CSV para análisis y uso posterior en otros proyectos.


---

### Consideraciones adicionales

- **Reproducibilidad:**  
  El notebook incluye pasos para asegurar que los resultados puedan replicarse, incluyendo la fijación de semillas y el manejo consistente de datos faltantes.

- **Escalabilidad:**  
  El uso de autoencoders permite capturar patrones complejos y reducir la dimensionalidad para mejorar el desempeño de modelos estadísticos posteriores.

- **Flexibilidad:**  
  Los embeddings generados pueden usarse como características para diferentes modelos predictivos o análisis exploratorios futuros.

- **Requisitos previos:**  
  Asegúrate de tener los directorios `embeddings` y `models` creados antes de ejecutar el notebook, para guardar los resultados y modelos entrenados.

- **Datos necesarios:**  
  El notebook depende de los datos procesados por el script `../ETL/ingesta_spark.ipynb`, que debe haberse ejecutado previamente para generar los archivos en `../Data/`.