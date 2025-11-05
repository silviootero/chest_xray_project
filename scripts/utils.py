# utils.py
import os
from pathlib import Path
import cv2
import numpy as np
from sklearn.model_selection import train_test_split


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def list_images_in_folder(folder, exts={'.png','.jpg','.jpeg','.bmp'}):
    files = []
    for root, _, filenames in os.walk(folder):
        for f in filenames:
            if Path(f).suffix.lower() in exts:
                files.append(os.path.join(root, f))
    return files


def read_gray(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Imagen no leída: {path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def save_npy(path, arr):
    ensure_dir(Path(path).parent)
    np.save(path, arr)


def load_npy(path):
    return np.load(path, allow_pickle=True)


def stratified_split(X, y, test_size=0.15, val_size=0.15, random_state=42):
    # Primero separar test, luego val desde train
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
    val_relative = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_relative, stratify=y_temp, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test