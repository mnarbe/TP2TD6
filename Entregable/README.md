## Estructura de archivos
- En main_script.py se encuentra el archivo principal del proyecto. Ejecutar este archivo para correr los gráficos y el pipeline que genera el mejor resultado en la competencia.
- base_xgboost contiene funciones de entrenamiento y selección de atributos
- database_utils contiene funciones de lectura y transformación del dataset. Sobre todo creación de features
- constants.py contiene constantes básicas
- graficos_utils.py contiene las funciones de creación de gráficos
- /transform_data_scripts contiene scripts para obtener datos de la api de spotify y mergearlo al dataset original