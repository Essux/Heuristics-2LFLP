Este proyecto fue hecho como parte de la materia de Heurística.

## Ejecución
Para ejecutar el proyecto debe usarse el comando:
`python main.py`

## Contenidos
Los contenidos de cada archivo .py en la carpeta son:
- algorithm.py - Diferentes algoritmos para hallar soluciones
- objects.py - Definición de las clases usadas
- random_generator.py - Generación aleatoria de las entradas
- readinput.py - Lectura de los archivos de entrada y salida de los datos para CPLEX
- main.py - Script que dirige la ejecución de todo el proyecto. Además se encarga de hacer las gráficas.

## Descripción de las gráficas
En cada ejecución del programa, se genera una gráfica en matplotlib que contiene las siguientes convenciones:
- Puntos Rojos: Clientes. Incluyen el número que los identifica.
- Puntos Azules: Instalaciones de Nivel 1. Incluyen el número que los identifica.
- Puntos Verdes: Instalaciones de Nivel 2. Incluyen el número que los identifica.
- Líneas Rojas. Indican que una instalación de nivel 1 abastece a un cliente.
    El ancho de la línea es proporcional al flujo de material entre ambos.
- Líneas Azules. Indican que una instalación de nivel 2 abastece a una instalación de nivel 1.
    El ancho de la línea es proporcional al flujo de material entre ambos.

## Generación de archivos
Al ser ejecutado el proyecto genera los siguientes archivos:
- I1.dat - Incluye los datos para ser usados en CPLEX
- I1.in - Archivo donde se almacenan los datos de entrada
- I1.profile - Incluye información detallada sobre el rendimiento del algoritmo constructivo

## Requerimientos
- matplotlib
- cProfile (Parte de Python Standard)
- pstats (Parte de Python Standard)