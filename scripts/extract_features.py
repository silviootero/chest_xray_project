# extract_features.py
import cv2
import numpy as np
from pathlib import Path
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray
from mahotas import features as mht
from utils import ensure_dir, list_images_in_folder, read_gray

# -----------------------------------------------------------
# CONFIGURACIÓN GENERAL
# -----------------------------------------------------------
DATASET_DIR = Path("data_preprocessed")  # imágenes ya redimensionadas
FEATURES_DIR = Path("features")
ensure_dir(FEATURES_DIR)

IMG_SIZE = (128, 128)
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
GABOR_ORIENTATIONS = 8
GABOR_FREQUENCY = 0.2

# -----------------------------------------------------------
# EXTRACCIÓN DE CARACTERÍSTICAS
# -----------------------------------------------------------
def extract_hog(img_gray):
    features, _ = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2), visualize=True, block_norm='L2-Hys')
    return features

def extract_lbp(img_gray):
    lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_haralick(img_gray):
    return mht.haralick(img_gray).mean(axis=0)

def extract_gabor(img_gray):
    feats = []
    for theta in np.arange(0, np.pi, np.pi / GABOR_ORIENTATIONS):
        kernel = cv2.getGaborKernel((9, 9), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(img_gray, cv2.CV_8UC3, kernel)
        feats.append(fimg.mean())
        feats.append(fimg.var())
    return np.array(feats)

def extract_sift(img_gray):
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img_gray, None)
    if des is None:
        des = np.zeros((1, 128))
    return des.flatten()[:2048] if des.size > 2048 else np.pad(des.flatten(), (0, 2048 - des.size))

# -----------------------------------------------------------
# PROCESAR IMÁGENES
# -----------------------------------------------------------
def process_subset(subset_name, feature_type):
    print(f"\nExtrayendo {feature_type.upper()} del conjunto {subset_name}...")
    subset_dir = DATASET_DIR / subset_name
    image_paths = list_images_in_folder(subset_dir)
    
    X, y = [], []

    for img_path in image_paths:
        try:
            img = read_gray(img_path)
            img = cv2.resize(img, IMG_SIZE)
            
            # Selección de característica
            if feature_type == "hog":
                feats = extract_hog(img)
            elif feature_type == "lbp":
                feats = extract_lbp(img)
            elif feature_type == "haralick":
                feats = extract_haralick(img)
            elif feature_type == "gabor":
                feats = extract_gabor(img)
            elif feature_type == "sift":
                feats = extract_sift(img)
            else:
                raise ValueError("Tipo de característica no reconocido.")

            X.append(feats)
            
            # Etiqueta = nombre de carpeta superior
            label = Path(img_path).parent.name
            if label.lower() in ["bacteria", "virus"]:
                label = "pneumonia"
            y.append(label)
        
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)

    # Guardar características
    out_X = FEATURES_DIR / f"{feature_type}_{subset_name}_X.npy"
    out_y = FEATURES_DIR / f"{feature_type}_{subset_name}_y.npy"
    ensure_dir(FEATURES_DIR)
    np.save(out_X, X)
    np.save(out_y, y)
    print(f"Guardado: {out_X.name}, {out_y.name} ({X.shape[0]} muestras, {X.shape[1]} features)")

# -----------------------------------------------------------
# MAIN: recorrer subconjuntos y tipos de características
# -----------------------------------------------------------
if __name__ == "__main__":
    feature_types = ["hog", "lbp", "haralick", "gabor", "sift"]
    subsets = ["train", "test"]

    for ft in feature_types:
        for sb in subsets:
            process_subset(sb, ft)

    print("\nExtracción de características completada.")
