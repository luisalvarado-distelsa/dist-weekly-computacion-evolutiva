# Computación Evolutiva - Distelsa Weekly

Este repositorio contiene el material relacionado a una presentación y proyecto semanal sobre computación evolutiva y su aplicación en problemas reales utilizando técnicas de machine learning e inteligencia artificial. La computación evolutiva es un enfoque basado en los principios de la evolución biológica para resolver problemas de optimización complejos que no tienen una solución determinística clara.

## Estructura del Repositorio

El proyecto está organizado en varias carpetas que contienen código fuente, datasets, presentaciones y scripts para el análisis de características. A continuación, se presenta la estructura principal del repositorio:

```
.
├── Presentación
│   ├── Images: Contiene imágenes y figuras usadas en la presentación.
│   ├── comp-evolutiva.pdf: Documento en PDF de la presentación sobre Computación Evolutiva.
│   └── comp-evolutiva.tex: Código fuente en LaTeX de la presentación.
├── ECE2024
│   ├── vehiculo_compartido.ipynb: Notebook con un caso de estudio de optimización de rutas.
│   ├── CNN.ipynb: Notebook con una implementación de una red neuronal convolucional (CNN).
│   └── WorkSpaceECE2024
│       ├── FeatureExtraction.py: Script para la extracción de características de las imágenes.
│       ├── afssa.py: Implementación de una búsqueda adaptativa para selección de características.
│       ├── FeatureSelection.ipynb: Notebook que explica la selección de características en el contexto del proyecto.
│       ├── DatasetECE2024: Carpeta con datasets de entrenamiento y validación.
├── LICENSE: Licencia Apache 2.0 para el proyecto.
├── main.py: Implementación de un algoritmo evolutivo simple para optimización de funciones.
└── funciones.txt: Descripción breve de las funciones principales usadas en el proyecto.
```

### Descripción de las Carpetas y Archivos

- **Presentación**: Contiene una presentación detallada sobre la computación evolutiva, incluyendo historia, conceptos y aplicaciones. El archivo `comp-evolutiva.tex` permite modificar el documento usando LaTeX.
- **ECE2024**: Esta carpeta contiene notebooks y scripts para trabajar con modelos de machine learning y optimización evolutiva:
  - `vehiculo_compartido.ipynb`: Un ejemplo de aplicación práctica donde se optimizan rutas de vehículos compartidos.
  - `CNN.ipynb`: Implementación de una red neuronal convolucional para clasificación de imágenes.
  - **WorkSpaceECE2024**: Directorio de trabajo con varias utilidades:
    - `FeatureExtraction.py`: Realiza la extracción de características, como intensidad, textura y morfología, de un conjunto de imágenes biomédicas.
    - `afssa.py`: Implementa un algoritmo de selección automática de características basado en la eficiencia multiobjetivo.
    - `FeatureSelection.ipynb`: Explica la importancia de la selección de características y cómo aplicarla a los datos del proyecto.
    - `DatasetECE2024`: Incluye imágenes de entrenamiento y validación con archivos de etiquetas correspondientes.

- **main.py**: Implementa un ejemplo de algoritmo evolutivo que busca optimizar una función objetivo. Permite explorar diferentes técnicas de cruce y mutación para llegar a la solución óptima.

### Algoritmo Evolutivo - `main.py`

El archivo `main.py` contiene un algoritmo evolutivo que busca optimizar la función objetivo \( f(x) = -(x-3)^2 + 10 \). En cada generación, los mejores individuos se seleccionan, cruzan y mutan para encontrar la solución óptima de manera iterativa. Puedes ajustar los parámetros como el tamaño de la población, el número de generaciones, y los métodos de cruce y mutación para probar diferentes configuraciones y estrategias.

### Ejecución del Proyecto

Para ejecutar los scripts principales del proyecto, sigue estos pasos:

1. Clona el repositorio:
   ```sh
   git clone https://github.com/tu_usuario/dist-weekly-computacion-evolutiva.git
   ```

2. Instala las dependencias requeridas. Puedes usar `pip` para instalar los paquetes necesarios:
   ```sh
   pip install -r requirements.txt
   ```

3. Ejecuta el script principal:
   ```sh
   python main.py
   ```

4. Puedes explorar los notebooks y scripts adicionales dentro de la carpeta `ECE2024` para realizar análisis más detallados y experimentos adicionales.

## Requisitos

- Python 3.8 o superior.
- Bibliotecas adicionales: `numpy`, `matplotlib`, `scikit-learn`, `Pillow`, `skimage`, `mahotas`.

## Licencia

Este proyecto está licenciado bajo la Apache License 2.0. Consulta el archivo `LICENSE` para más detalles.

## Contacto

Autor: Luis Alfredo Alvarado Rodríguez
- **Email**: luis.alvarado@distelsa.com.gt

Si tienes alguna pregunta o comentario, no dudes en contactar o abrir un Issue en este repositorio.

---

Este README ofrece una visión general del proyecto y los recursos disponibles. Si quieres contribuir o explorar más detalles, siéntete libre de navegar y experimentar.

