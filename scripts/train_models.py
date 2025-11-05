# train_models.py
import numpy as np
import joblib
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from utils import ensure_dir

# -----------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------
FEATURES_DIR = Path("features")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")
ensure_dir(MODELS_DIR)
ensure_dir(REPORTS_DIR)

# Modelos a entrenar
MODELS = {
    "svm_rbf": SVC(kernel='rbf', C=10, gamma='scale', probability=True),
    "random_forest": RandomForestClassifier(n_estimators=150, random_state=42),
    "mlp": MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=300, random_state=42)
}

# -----------------------------------------------------------
# FUNCIONES AUXILIARES
# -----------------------------------------------------------
def load_features(feature_type, subset):
    X_path = FEATURES_DIR / f"{feature_type}_{subset}_X.npy"
    y_path = FEATURES_DIR / f"{feature_type}_{subset}_y.npy"
    if not X_path.exists() or not y_path.exists():
        return None, None
    X = np.load(X_path, allow_pickle=True)
    y = np.load(y_path, allow_pickle=True)
    return X, y


def evaluate_and_save(model_name, feature_type, clf, X_test, y_test):
    """Evalúa el modelo y guarda métricas."""
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds, target_names=np.unique(y_test))
    cm = confusion_matrix(y_test, preds)

    print(f"\nResultados [{model_name}] con {feature_type.upper()}:")
    print(f"   - Accuracy: {acc:.4f}")
    print(f"   - Reporte:\n{report}")

    # Guardar reporte en txt
    report_file = REPORTS_DIR / f"{model_name}_{feature_type}_report.txt"
    with open(report_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Feature: {feature_type}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

    print(f"Reporte guardado en: {report_file}")


def train_and_evaluate(feature_type):
    """Entrena todos los modelos sobre un tipo de característica."""
    print(f"\n==============================")
    print(f"Entrenando modelos con características: {feature_type.upper()}")
    print(f"==============================")

    # Cargar train y test
    X_train, y_train = load_features(feature_type, "train")
    X_test, y_test = load_features(feature_type, "test")

    if X_train is None or X_test is None:
        print(f"No se encontraron características para {feature_type}. Saltando...")
        return

    # Asegurar forma correcta
    X_train = X_train.reshape(len(X_train), -1)
    X_test = X_test.reshape(len(X_test), -1)

    for model_name, model in MODELS.items():
        print(f"\nEntrenando modelo: {model_name}...")
        clf = model
        clf.fit(X_train, y_train)

        # Guardar modelo
        model_file = MODELS_DIR / f"{model_name}_{feature_type}.pkl"
        joblib.dump(clf, model_file)
        print(f"Modelo guardado: {model_file}")

        # Evaluar y guardar reporte
        evaluate_and_save(model_name, feature_type, clf, X_test, y_test)

# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    print("Buscando características disponibles en carpeta 'features/'...")
    available_features = sorted(set([f.name.split('_')[0] for f in FEATURES_DIR.glob("*_train_X.npy")]))
    
    if not available_features:
        print("No se encontraron archivos de características. Ejecuta primero extract_features.py.")
    else:
        print(f"Características detectadas: {available_features}")

        for ft in available_features:
            train_and_evaluate(ft)

    print("\nEntrenamiento completado.")
