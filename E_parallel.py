from mpi4py import MPI
import time
import psutil
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd

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
        # Medir tiempo y memoria antes del entrenamiento
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss  # Memoria usada antes del entrenamiento

        model.fit(X_train, y_train)

        # Medir tiempo y memoria después del entrenamiento
        train_time = time.time() - start_time
        end_memory = psutil.Process().memory_info().rss  # Memoria usada después del entrenamiento

        memory_used = (end_memory - start_memory) / (1024 * 1024)  # Convertir a MB

        results[model_name] = {
            'model': model,
            'train_time': train_time,
            'memory_used': memory_used
        }

    return results

# Recopilar métricas y combinar resultados
def evaluate_and_combine(models, X_test, y_test):
    local_results = {}

    for model_name, model_data in models.items():
        model = model_data['model']
        y_pred = model.predict(X_test)

        local_results[model_name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'train_time': model_data['train_time'],
            'memory_used': model_data['memory_used']
        }

    # Reducción paralela para combinar resultados
    global_results = comm.gather(local_results, root=0)

    if rank == 0:
        combined_results = {}
        for result in global_results:
            for model_name, metrics in result.items():
                if model_name not in combined_results:
                    combined_results[model_name] = {
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1_score': [],
                        'train_time': [],
                        'memory_used': []
                    }
                for metric, value in metrics.items():
                    combined_results[model_name][metric].append(value)

        # Promediar las métricas
        for model_name, metrics in combined_results.items():
            for metric in metrics:
                combined_results[model_name][metric] = sum(metrics[metric]) / len(metrics[metric])

        return combined_results
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

    if rank == 0:
        print("\nResultados Finales:\n")
        for model_name, metrics in final_results.items():
            print(f"Modelo: {model_name}")
            for metric, value in metrics.items():
                if metric in ['train_time', 'memory_used']:
                    print(f"  {metric.replace('_', ' ').capitalize()}: {value:.4f}")
                else:
                    print(f"  {metric.capitalize()}: {value:.4f}")
            print("-" * 40)
