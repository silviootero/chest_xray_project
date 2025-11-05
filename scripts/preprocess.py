# preprocess.py
import os
import cv2
import numpy as np
from pathlib import Path
from utils import ensure_dir, list_images_in_folder

# -----------------------------------------------------------
# CONFIGURACIÓN DE RUTAS
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "data_preprocessed"
IMG_SIZE = (128, 128)


# -----------------------------------------------------------
# FUNCIÓN DE PREPROCESAMIENTO
# -----------------------------------------------------------
def preprocess_images(src_dir, dst_dir):
    print(f"Procesando: {src_dir}")
    ensure_dir(dst_dir)
    image_paths = list_images_in_folder(src_dir)
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            print(f"No se pudo leer: {img_path}")
            continue
        img_resized = cv2.resize(img, IMG_SIZE)
        rel_path = Path(img_path).relative_to(src_dir)
        out_path = dst_dir / rel_path
        ensure_dir(out_path.parent)
        cv2.imwrite(str(out_path), img_resized)

# -----------------------------------------------------------
# PROCESAR TRAIN Y TEST
# -----------------------------------------------------------
for subset in ["train", "test"]:
    src = DATASET_DIR / subset
    dst = OUTPUT_DIR / subset
    if src.exists():
        preprocess_images(src, dst)
    else:
        print(f"No existe el subconjunto {subset}, se omite.")

print("Preprocesamiento completado. Imágenes listas en:", OUTPUT_DIR)
