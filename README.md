
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
Este proyecto es un ejemplo robusto de cómo las técnicas de computación de alto rendimiento pueden integrarse con machine learning para resolver problemas de clasificación, reduciendo tiempos y optimizando recursos.

---

## Modelos Evaluados

- Logistic Regression
- Random Forest
- Gradient Boosting
- Support Vector Classifier (SVC)
- K-Nearest Neighbors (KNN)

---

## Resultados

Los resultados obtenidos bajo diferentes configuraciones (Cluster, MPI, Serial) se encuentran en el archivo `results_summary.txt`.

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
