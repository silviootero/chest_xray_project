# ============================================================
# streamlit_app.py — Clasificador de Rayos X con múltiples features
# ============================================================
import sys
import streamlit as st
import cv2
import numpy as np
import joblib
from pathlib import Path
from skimage.feature import hog, local_binary_pattern
from skimage.color import rgb2gray, rgb2lab
from pathlib import Path
from PIL import Image
import io
from mahotas import features as mht
import pandas as pd
sys.path.append(str(Path(__file__).resolve().parent.parent))
from scripts.utils import ensure_dir

# -----------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
IMG_SIZE = (128, 128)
AVAILABLE_MODELS = list(MODELS_DIR.glob("*.pkl"))
ensure_dir(MODELS_DIR)

st.set_page_config(page_title="Clasificador de Rayos X", page_icon="", layout="wide")

# Estilos sutiles
st.markdown(
    """
    <style>
    .result-card {background: #0b1533; border: 1px solid #223; padding: 16px; border-radius: 12px;}
    .result-title {font-size: 1.1rem; color: #9bbcff; margin-bottom: 6px;}
    .result-pred {font-size: 1.6rem; font-weight: 700;}
    .small-note {color: #9aa3b2; font-size: 0.9rem;}
    .section {padding: 10px 0 0 0; margin-bottom: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# FUNCIONES DE EXTRACCIÓN DE CARACTERÍSTICAS
# -----------------------------------------------------------
def extract_hog(img):
    gray = rgb2gray(img)
    features, _ = hog(gray, pixels_per_cell=(8, 8),
                      cells_per_block=(2, 2),
                      orientations=9,
                      block_norm='L2-Hys',
                      visualize=True)
    return features

def extract_lbp(img):
    gray = rgb2gray(img)
    radius = 2
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3),
                           range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_gabor(img):
    gray = (rgb2gray(img) * 255).astype(np.uint8)
    feats = []
    GABOR_ORIENTATIONS = 8
    for theta in np.arange(0, np.pi, np.pi / GABOR_ORIENTATIONS):
        kernel = cv2.getGaborKernel((9, 9), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        fimg = cv2.filter2D(gray, cv2.CV_8UC3, kernel)
        feats.append(fimg.mean())
        feats.append(fimg.var())
    return np.array(feats)

def extract_haralick(img):
    gray = (rgb2gray(img) * 255).astype(np.uint8)
    return mht.haralick(gray).mean(axis=0)

def extract_lab(img):
    lab = rgb2lab(img)
    feats = [
        np.mean(lab[:, :, 0]), np.std(lab[:, :, 0]),
        np.mean(lab[:, :, 1]), np.std(lab[:, :, 1]),
        np.mean(lab[:, :, 2]), np.std(lab[:, :, 2])
    ]
    return np.array(feats)

def extract_sift(img):
    gray = (rgb2gray(img) * 255).astype(np.uint8)
    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(gray, None)
    if des is None:
        des = np.zeros((1, 128))
    flat = des.flatten()
    if flat.size >= 2048:
        return flat[:2048]
    else:
        return np.pad(flat, (0, 2048 - flat.size))

def extract_surf(img):
    gray = (rgb2gray(img) * 255).astype(np.uint8)
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        kps, des = surf.detectAndCompute(gray, None)
        if des is None:
            des = np.zeros((1, 64))
        return np.mean(des, axis=0)
    except Exception:
        st.warning("SURF no disponible en tu instalación de OpenCV. Usa otro modelo.")
        return np.zeros(64)

# -----------------------------------------------------------
# UTILIDADES
# -----------------------------------------------------------
def preprocess_image(image_bytes):
    """Lee bytes de imagen, valida y devuelve un np.array RGB redimensionado.
    Devuelve None si la imagen no es válida.
    """
    if image_bytes is None or len(image_bytes) == 0:
        st.error("No se recibieron datos de imagen.")
        return None

    # Intento 1: OpenCV
    try:
        img_array = np.frombuffer(image_bytes, np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img_resized = cv2.resize(img_bgr, IMG_SIZE)
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            return img_rgb
    except Exception:
        pass

    # Intento 2: PIL (más tolerante con algunos formatos)
    try:
        pil_img = Image.open(io.BytesIO(image_bytes))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img = pil_img.resize(IMG_SIZE)
        return np.array(pil_img)
    except Exception:
        st.error("No se pudo decodificar la imagen. Asegúrate de subir un JPG o PNG válido.")
        return None


def load_model_and_type(model_path):
    """Carga el modelo y deduce el tipo de característica por nombre de archivo.
    Espera archivos guardados como '{model_name}_{feature_type}.pkl'.
    """
    model_path = Path(model_path)
    clf = joblib.load(model_path)
    name = model_path.stem
    parts = name.split('_')
    feature_type = parts[-1] if len(parts) >= 2 else None
    if feature_type is None:
        st.warning("No se pudo deducir el tipo de características del nombre del modelo. Se intentará 'hog'.")
        feature_type = "hog"
    return clf, feature_type


def predict_image(model, feature_type, image):
    """Extrae las características y hace la predicción."""
    if image is None:
        return None

    if feature_type == "hog":
        feats = extract_hog(image)
    elif feature_type == "lbp":
        feats = extract_lbp(image)
    elif feature_type == "gabor":
        feats = extract_gabor(image)
    elif feature_type == "haralick":
        feats = extract_haralick(image)
    elif feature_type == "lab":
        feats = extract_lab(image)
    elif feature_type == "sift":
        feats = extract_sift(image)
    elif feature_type == "surf":
        feats = extract_surf(image)
    else:
        st.error(f"Tipo de característica '{feature_type}' no soportada.")
        return None

    feats = feats.reshape(1, -1)
    return model.predict(feats)[0]


# -----------------------------------------------------------
# INTERFAZ STREAMLIT
# -----------------------------------------------------------
st.title("Clasificador de Neumonía con Rayos X")
st.caption("Carga una imagen de tórax y usa un modelo preentrenado para clasificar entre NORMAL y PNEUMONIA.")

if not AVAILABLE_MODELS:
    st.warning("No se encontraron modelos entrenados en la carpeta 'models/'. Ejecuta train_models.py primero.")
else:
    # Sidebar: controles
    with st.sidebar:
        st.header("Configuración")
        # Construir etiquetas amigables: "FEATURE · MODEL"
        def _pretty_model_name(name):
            mapping = {"svm_rbf": "SVM (RBF)", "random_forest": "Random Forest", "mlp": "MLP"}
            return mapping.get(name, name.replace("_", " ").title())

        def _pretty_feature_name(name):
            return name.upper()

        model_options = []
        for p in AVAILABLE_MODELS:
            stem = p.stem
            parts = stem.split("_")
            if len(parts) >= 2:
                model_n = "_".join(parts[:-1])
                feat = parts[-1]
            else:
                model_n = stem
                feat = "hog"
            label = f"{_pretty_feature_name(feat)} · {_pretty_model_name(model_n)}"
            model_options.append((label, p))

        labels = [lbl for lbl, _ in model_options]
        choice = st.selectbox("Modelo", labels, help="Selecciona el modelo preentrenado a usar")
        model_path = next(p for lbl, p in model_options if lbl == choice)
        uploaded_file = st.file_uploader("Imagen de Rayos X", type=["jpg", "jpeg", "png"], help="Formatos admitidos: JPG/PNG")
        st.markdown("<div class='small-note'>La imagen se redimensiona a 128×128 para la extracción de características.</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.1, 0.9], gap="large")

    with col1:
        st.subheader("Vista previa")
        if uploaded_file is not None:
            image = preprocess_image(uploaded_file.read())
            if image is not None:
                st.image(image, caption="Imagen cargada", use_container_width=True)
            else:
                st.info("Carga una imagen válida para continuar.")
                st.stop()
        else:
            st.info("Sube una imagen desde la barra lateral.")
            image = None

    with col2:
        st.subheader("Detalles del modelo")
        st.write(f"**Seleccionado:** {choice}")
        st.divider()
        can_classify = uploaded_file is not None and image is not None
        disabled = not can_classify
        if st.button("Clasificar", use_container_width=True, disabled=disabled):
            with st.spinner("Clasificando..."):
                model, feature_type = load_model_and_type(model_path)
                pred = predict_image(model, feature_type, image)

                if pred is not None:
                    st.markdown("<div class='result-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='result-title'>Resultado</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='result-pred'>{pred.upper()}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                    # Probabilidades si están disponibles
                    try:
                        feats = None
                        if feature_type == "hog":
                            feats = extract_hog(image)
                        elif feature_type == "lbp":
                            feats = extract_lbp(image)
                        elif feature_type == "gabor":
                            feats = extract_gabor(image)
                        elif feature_type == "haralick":
                            feats = extract_haralick(image)
                        elif feature_type == "lab":
                            feats = extract_lab(image)
                        elif feature_type == "sift":
                            feats = extract_sift(image)
                        elif feature_type == "surf":
                            feats = extract_surf(image)

                        if feats is not None:
                            feats = feats.reshape(1, -1)
                            if hasattr(model, "predict_proba"):
                                proba = model.predict_proba(feats)[0]
                                classes = getattr(model, "classes_", np.array(["NORMAL", "PNEUMONIA"]))
                                df = pd.DataFrame({"Clase": classes, "Probabilidad": proba}).sort_values("Probabilidad", ascending=False)
                                st.markdown("<div class='section'></div>", unsafe_allow_html=True)
                                st.caption("Probabilidades")
                                st.bar_chart(df.set_index("Clase"))
                    except Exception:
                        pass
                else:
                    st.error("No se pudo generar la predicción.")

    st.markdown("<div class='small-note'>Consejo: elige distintos modelos para comparar resultados.</div>", unsafe_allow_html=True)
