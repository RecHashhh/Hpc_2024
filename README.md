
# Optimización de Modelos de Machine Learning usando Ensembles y Paralelización con OpenMPI

Este proyecto combina técnicas de machine learning y computación de alto rendimiento utilizando MPI para entrenar modelos en paralelo, mejorando la eficiencia computacional y la calidad de predicción mediante ensamblajes.

## Tabla de Contenidos

- [Resumen del Proyecto](#resumen-del-proyecto)
- [Modelos Evaluados](#modelos-evaluados)
- [Resultados](#resultados)
- [Cómo Ejecutar](#cómo-ejecutar)
- [Requisitos](#requisitos)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

---

## Resumen del Proyecto: Enfoque Ensembles Paralelizado con MPI

### **Título**
**Optimización de Modelos de Machine Learning usando Ensembles y Paralelización con OpenMPI**

### **Descripción General**
El proyecto busca combinar modelos de machine learning entrenados de manera independiente en un sistema paralelo. Este enfoque mejora la eficiencia computacional y la precisión al utilizar técnicas de ensamblaje (ensemble) en predicciones, aplicando paralelización mediante la librería **mpi4py** (implementación de MPI en Python).

### **Objetivo**
- Entrenar varios modelos en paralelo utilizando un conjunto de datos distribuido entre múltiples nodos o procesos.
- Combinar los resultados de los modelos para generar una predicción final usando técnicas como promediado o votación ponderada, optimizando el uso de recursos computacionales.
- Mostrar la viabilidad del uso de herramientas de computación de alto rendimiento como OpenMPI para proyectos de machine learning.

### **Razones para su Validación**
1. **Relevancia Académica**: El proyecto combina conceptos avanzados como machine learning, computación paralela, y técnicas de ensamblaje, alineándose con los temas clave en Computación de Alto Rendimiento.
2. **Escalabilidad**: La distribución paralela permite manejar grandes volúmenes de datos y reducir significativamente los tiempos de entrenamiento.
3. **Innovación**: Integra múltiples modelos (Logistic Regression, Random Forest, Gradient Boosting, etc.) entrenados en paralelo y combinados mediante reducción paralela, utilizando herramientas prácticas y ampliamente usadas en HPC (High-Performance Computing).
4. **Impacto Práctico**: Es aplicable a diversos problemas de clasificación y predicción, demostrando cómo resolver problemas reales eficientemente.

### **Estrategia del Proyecto**
1. **Distribución de Datos**:
   - Dividir el conjunto de datos entre procesos paralelos usando `comm.scatter()`.
   - Cada proceso entrena un submodelo de manera independiente.
   
2. **Entrenamiento Local**:
   - Implementar modelos populares como Logistic Regression, Random Forest, y Gradient Boosting.
   - Medir métricas como tiempo de entrenamiento y precisión localmente.

3. **Reducción Paralela**:
   - Recopilar las métricas de cada proceso mediante `comm.gather()`.
   - Combinar los resultados con técnicas de reducción (reducción por promedio en este caso).
   
4. **Resultados Finales**:
   - Unificar y promediar las métricas (precisión, recall, F1, etc.) para obtener un conjunto de resultados finales representativos.

### **Resultados Esperados**
- Un conjunto de métricas finales que demuestren la eficiencia del enfoque ensemble paralelo.
- Validación de la mejora en rendimiento computacional y calidad de predicción en comparación con enfoques secuenciales.

### **Conclusión**
Este proyecto es un ejemplo robusto de cómo las técnicas de computación de alto rendimiento pueden integrarse con machine learning para resolver problemas de clasificación, reduciendo tiempos y optimizando recursos. Los resultados muestran mejoras significativas en rendimiento computacional y precisión, destacando la escalabilidad del enfoque en sistemas distribuidos.

---

## Modelos Evaluados

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)

---
## Resultados Obtenidos

### Cluster: Resultados Finales

#### Modelo: Logistic Regression
- **Accuracy**: 0.8703
- **Precision**: 0.8760
- **Recall**: 0.8703
- **F1_score**: 0.8711
- **Train time**: 0.0907
- **Memory used**: 0.7505

#### Modelo: Random Forest
- **Accuracy**: 0.9683
- **Precision**: 0.9691
- **Recall**: 0.9683
- **F1_score**: 0.9683
- **Train time**: 0.3137
- **Memory used**: 2.0962

#### Modelo: Gradient Boosting
- **Accuracy**: 0.9670
- **Precision**: 0.9676
- **Recall**: 0.9670
- **F1_score**: 0.9670
- **Train time**: 1.5339
- **Memory used**: 0.4756

#### Modelo: SVC
- **Accuracy**: 0.9230
- **Precision**: 0.9282
- **Recall**: 0.9230
- **F1_score**: 0.9235
- **Train time**: 0.0523
- **Memory used**: 1.2432

#### Modelo: KNN
- **Accuracy**: 0.9280
- **Precision**: 0.9337
- **Recall**: 0.9280
- **F1_score**: 0.9278
- **Train time**: 0.0008
- **Memory used**: 0.0000

---

### Entrenamiento en Serial usando Colab

#### Modelo: K-Nearest Neighbors
- **Accuracy**: 0.9930
- **Precision**: 0.9931
- **Recall**: 0.9930
- **F1 Score**: 0.9930
- **Train time**: 2.0560 segundos
- **Memory used**: 4.46 MiB
- **CPU Usage**: 6.50 %

#### Modelo: Support Vector Classifier
- **Accuracy**: 0.9773
- **Precision**: 0.9784
- **Recall**: 0.9773
- **F1 Score**: 0.9774
- **Train time**: 7.9673 segundos
- **Memory used**: 6.80 MiB
- **CPU Usage**: -34.00 %

#### Modelo: Gradient Boosting
- **Accuracy**: 0.9910
- **Precision**: 0.9911
- **Recall**: 0.9910
- **F1 Score**: 0.9910
- **Train time**: 19.1186 segundos
- **Memory used**: 9.10 MiB
- **CPU Usage**: -91.00 %

#### Modelo: Random Forest
- **Accuracy**: 0.9987
- **Precision**: 0.9987
- **Recall**: 0.9987
- **F1 Score**: 0.9987
- **Train time**: 8.0363 segundos
- **Memory used**: 11.25 MiB
- **CPU Usage**: -34.00 %

#### Modelo: Logistic Regression
- **Accuracy**: 0.8950
- **Precision**: 0.8959
- **Recall**: 0.8950
- **F1 Score**: 0.8952
- **Train time**: 3.1934 segundos
- **Memory used**: 6.80 MiB
- **CPU Usage**: 6.50 %

---

## Cómo Ejecutar

### Requisitos Previos
- **Ubuntu 20.04**.
- Instalar OpenMPI:
  ```bash
  sudo apt update
  sudo apt install openmpi-bin openmpi-common libopenmpi-dev
  ```
- Instalar las dependencias de Python:
  ```bash
  pip install -r requirements.txt
  ```

### Pasos
1. Clona este repositorio:
   ```bash
   git clone https://github.com/tuusuario/nombre-repo.git
   cd nombre-repo
   ```
2. Ejecuta el proyecto en modo paralelo:
   ```bash
   mpiexec -n 4 python3 Project.py
   ```

---

## Requisitos

- Python 3.8 o superior.
- Dependencias adicionales: `numpy`, `matplotlib`, `scikit-learn`, `mpi4py`.

---

## Contribuciones

¡Las contribuciones son bienvenidas! Si tienes ideas, sugerencias o encuentras problemas, no dudes en abrir un **Issue** o enviar un **Pull Request**.

---

## Licencia

Este proyecto está bajo la [MIT License](LICENSE).
