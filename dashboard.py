import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import glob
import os
from ultralytics import YOLO

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="🐾 Animal Vision AI", layout="wide")

# ==========================
# CSS THEME
# ==========================
st.markdown("""
    <style>
        .stApp {
            background: linear-gradient(145deg, #141E30, #243B55);
            color: #f5f5f5;
            font-family: 'Poppins', sans-serif;
            background-image: url('https://images.unsplash.com/photo-1546182990-dffeafbe841d');
            background-size: cover;
            background-attachment: fixed;
            background-blend-mode: overlay;
        }
        .title {text-align:center; color:#FFCC70; font-size:45px; font-weight:900; margin-top:10px; text-shadow: 2px 2px 8px rgba(0,0,0,0.5);}
        .subtitle {text-align:center; color:#E5E5E5; font-size:18px; margin-bottom:25px;}
        .result-box {background: rgba(255,255,255,0.12); padding:20px; border-radius:16px; box-shadow: 0 4px 18px rgba(0,0,0,0.3); backdrop-filter: blur(8px);}
        footer {text-align:center; color:#ccc; margin-top:35px; padding:10px; border-top: 1px solid rgba(255,255,255,0.2);}
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
MODEL_FOLDER = "model"

def find_first(pattern):
    files = glob.glob(os.path.join(MODEL_FOLDER, pattern))
    return files[0] if files else None

@st.cache_resource
def load_models():
    # TensorFlow / Keras model
    h5_path = find_first("*.h5")
    model = None
    if h5_path:
        try:
            model = tf.keras.models.load_model(h5_path)
        except:
            model = None
    # YOLO model
    pt_path = find_first("*.pt")
    yolo_model = None
    if pt_path:
        try:
            yolo_model = YOLO(pt_path)
        except:
            yolo_model = None
    return model, h5_path, yolo_model, pt_path

model, h5_info, yolo_model, pt_info = load_models()

# ==========================
# CLASS LABELS & INFO HEWAN
# ==========================
class_names = ["spider", "cat", "dog", "chicken", "horse", "butterfly", "fish"]
animal_info = {
    "spider": {"nama": "🕷 Laba-laba", "habitat": "Taman, rumah, pepohonan.", "makanan": "Serangga kecil seperti lalat atau nyamuk.", "fakta": "Laba-laba membuat jaring sutra yang kuat untuk menangkap mangsanya."},
    "cat": {"nama": "🐱 Kucing", "habitat": "Lingkungan rumah manusia.", "makanan": "Ikan, daging, makanan kucing kering.", "fakta": "Kucing dapat tidur hingga 16 jam sehari!"},
    "dog": {"nama": "🐶 Anjing", "habitat": "Lingkungan rumah manusia.", "makanan": "Daging, tulang, makanan anjing kering.", "fakta": "Anjing dikenal sangat setia terhadap pemiliknya."},
    "chicken": {"nama": "🐔 Ayam", "habitat": "Kandang dan ladang peternakan.", "makanan": "Biji-bijian dan serangga kecil.", "fakta": "Ayam dapat mengenali lebih dari 100 wajah manusia!"},
    "horse": {"nama": "🐴 Kuda", "habitat": "Padang rumput dan peternakan.", "makanan": "Rumput, jerami, gandum.", "fakta": "Kuda bisa tidur sambil berdiri."},
    "butterfly": {"nama": "🦋 Kupu-kupu", "habitat": "Kebun, hutan, ladang bunga.", "makanan": "Nektar bunga.", "fakta": "Kupu-kupu mencicipi rasa dengan kakinya!"},
    "fish": {"nama": "🐟 Ikan", "habitat": "Air tawar dan laut.", "makanan": "Plankton, cacing, serangga air.", "fakta": "Beberapa ikan bisa tidur dengan mata terbuka!"}
}

# ==========================
# SIDEBAR NAVIGASI
# ==========================
mode = st.sidebar.radio(
    "📌 Pilih Halaman / Mode:",
    ("Beranda", "Klasifikasi", "Deteksi Objek (YOLO)", "Status Model")
)

# ==========================
# FUNGSIONALITAS
# ==========================
def preprocess_image(pil_img, model):
    input_shape = model.input_shape[1:3] if model else (224,224)
    img_resized = pil_img.resize(input_shape)
    arr = image.img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0)/255.0
    return arr

def predict_image(model, pil_img):
    arr = preprocess_image(pil_img, model)
    preds = model.predict(arr)
    idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = class_names[idx] if idx < len(class_names) else "unknown"
    return label, confidence

# ==========================
# HALAMAN BERANDA
# ==========================
if mode == "Beranda":
    st.markdown("<div class='title'>🐾 Selamat Datang di Animal Vision AI</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Gunakan sidebar untuk navigasi ke Klasifikasi, Deteksi Objek, atau Status Model</div>", unsafe_allow_html=True)
    st.info("📁 Pastikan file model (.h5 dan .pt) tersedia di folder 'model/' sebelum mulai.")

# ==========================
# HALAMAN STATUS MODEL
# ==========================
elif mode == "Status Model":
    st.header("📦 Status Model")
    if model is None:
        st.error("❌ Model klasifikasi (.h5) tidak tersedia")
    else:
        st.success(f"✅ Model klasifikasi dimuat dari: {h5_info}")
        st.write(f"📏 Input model: {model.input_shape}")
    if yolo_model is None:
        st.error("❌ Model YOLO (.pt) tidak tersedia")
    else:
        st.success(f"✅ Model YOLO dimuat dari: {pt_info}")

# ==========================
# HALAMAN KLASIFIKASI
# ==========================
elif mode == "Klasifikasi":
    st.header("🖼 Klasifikasi Gambar")
    uploaded_file = st.file_uploader("📤 Unggah gambar hewan (.jpg/.jpeg/.png)", type=["jpg","jpeg","png"])
    if uploaded_file:
        if model is None:
            st.error("Model klasifikasi tidak tersedia!")
        else:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="📸 Gambar yang diunggah", use_column_width=True)
            label, conf = predict_image(model, img)
            if label in animal_info:
                info_obj = animal_info[label]
                st.success(f"🌟 Teridentifikasi: {info_obj['nama']} — Confidence: {conf*100:.2f}%")
                st.markdown(f"""
                <div class='result-box'>
                    <h3>{info_obj['nama']}</h3>
                    <b>🌍 Habitat:</b> {info_obj['habitat']}<br>
                    <b>🍽 Makanan:</b> {info_obj['makanan']}<br>
                    <b>💡 Fakta menarik:</b> {info_obj['fakta']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning(f"Prediksi: {label} (data tidak lengkap). Confidence: {conf*100:.2f}%")
    else:
        st.info("📁 Unggah gambar untuk memulai klasifikasi.")

# ==========================
# HALAMAN DETEKSI OBJEK YOLO
# ==========================
elif mode == "Deteksi Objek (YOLO)":
    st.header("📌 Deteksi Objek YOLO")
    uploaded_file = st.file_uploader("📤 Unggah gambar hewan (.jpg/.jpeg/.png)", type=["jpg","jpeg","png"], key="yolo")
    if uploaded_file:
        if yolo_model is None:
            st.error("Model YOLO (.pt) tidak tersedia!")
        else:
            img = Image.open(uploaded_file).convert("RGB")
            results = yolo_model(img)
            result_img = results[0].plot()
            st.image(result_img, caption="📌 Hasil Deteksi YOLO", use_column_width=True)
    else:
        st.info("📁 Unggah gambar untuk memulai deteksi objek.")

# ==========================
# FOOTER
# ==========================
st.markdown("""
<footer>
    🐾 <b>Animal Vision AI</b> • by Rini Safariani 🌷<br>
    Letakkan model klasifikasi (.h5) dan YOLO (.pt) di folder <code>model/</code>
</footer>
""", unsafe_allow_html=True)
