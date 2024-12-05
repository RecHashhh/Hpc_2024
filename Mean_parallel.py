from mpi4py import MPI
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import psutil

# Configuración MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Cargar el dataset y dividir datos entre procesos

def load_and_distribute_data(file_path):
    if rank == 0:
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['fetal_health'])
        X = df.drop('fetal_health', axis=1)
        y = df['fetal_health']

        imputer = SimpleImputer(strategy='mean')
        X = imputer.fit_transform(X)

        # Dividir datos entre procesos
        X_split = np.array_split(X, size)
        y_split = np.array_split(y, size)
    else:
        X_split = None
        y_split = None

    # Enviar los datos a todos los procesos
    X_local = comm.scatter(X_split, root=0)
    y_local = comm.scatter(y_split, root=0)

    return X_local, y_local

# Entrenar modelo local

def train_local_model(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(),
        "SVC": SVC(),
        "KNN": KNeighborsClassifier()
    }

    results = {}
    for model_name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        results[model_name] = {
            'model': model,
            'train_time': train_time
        }

    return results

# Recopilar métricas y combinar resultados
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def evaluate_and_combine(models, X_test, y_test):
    local_results = {}

    # Evaluar cada modelo localmente
    for model_name, model_data in models.items():
        model = model_data['model']
        y_pred = model.predict(X_test)

        # Guardar las métricas para cada modelo
        local_results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }

    # Reducción paralela para combinar resultados
    global_results = comm.gather(local_results, root=0)

    if rank == 0:
        # Inicializar listas para cada métrica
        accuracy_values = []
        precision_values = []
        recall_values = []
        f1_score_values = []

        # Recopilar métricas de todos los procesos
        for result in global_results:
            for model_name, metrics in result.items():
                # Acceder directamente a las métricas sin usar items()
                accuracy_values.append(metrics['accuracy'])
                precision_values.append(metrics['precision'])
                recall_values.append(metrics['recall'])
                f1_score_values.append(metrics['f1_score'])

        # Calcular los promedios de las métricas
        averaged_results = {
            'accuracy': np.mean(accuracy_values),
            'precision': np.mean(precision_values),
            'recall': np.mean(recall_values),
            'f1_score': np.mean(f1_score_values)
        }

        # Imprimir los resultados promediados
        print("\nResultados Promediados Finales:")
        for metric, value in averaged_results.items():
            print(f"{metric.capitalize()}: {value:.4f}")

        return averaged_results
    else:
        return None


# Main
if __name__ == "__main__":
    file_path = 'balanced_fetal_health.csv'

    # Dividir datos
    X_local, y_local = load_and_distribute_data(file_path)
    
    # Dividir conjunto de entrenamiento y prueba local
    X_train, X_test, y_train, y_test = train_test_split(X_local, y_local, test_size=0.2, random_state=42)

    # Entrenar modelos localmente
    local_models = train_local_model(X_train, y_train)

    # Evaluar y combinar resultados
    final_results = evaluate_and_combine(local_models, X_test, y_test)
