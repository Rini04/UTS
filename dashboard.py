import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import random

# ==========================
# Load Models
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/Rini Safariani_Laporan 4.pt")  # Model deteksi objek
    classifier = tf.keras.models.load_model("model/model_Rini_Laporan 2.h5")  # Model klasifikasi
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# Informasi tambahan hewan
# ==========================
animal_info = {
    0: {"nama": "Ikan", "fakta": "Ikan dapat bernapas di dalam air menggunakan insang...", "habitat": "Sungai, danau, laut, dan kolam.", "makanan": "Plankton, serangga, cacing, atau pelet ikan.", "lingkungan": "Menyukai air bersih, cukup oksigen, dan suhu yang sesuai spesiesnya.", "bg": "https://i.ibb.co/3NdrmQ1/fish-bg.jpg"},
    1: {"nama": "Kucing", "fakta": "Kucing dapat tidur hingga 16 jam sehari...", "habitat": "Biasanya hidup di rumah, perkotaan, dan pedesaan.", "makanan": "Daging, ikan, dan makanan kucing komersial.", "lingkungan": "Menyukai lingkungan hangat dan aman.", "bg": "https://i.ibb.co/6B0K7Gd/cat-bg.jpg"},
    2: {"nama": "Anjing", "fakta": "Anjing memiliki indera penciuman yang 40 kali lebih tajam...", "habitat": "Hewan peliharaan yang hidup bersama manusia di rumah.", "makanan": "Daging, sayuran, dan makanan anjing komersial.", "lingkungan": "Menyukai lingkungan yang aktif dan sosial.", "bg": "https://i.ibb.co/V2L9Fgj/dog-bg.jpg"},
    3: {"nama": "Kuda", "fakta": "Kuda dapat berlari hingga 70 km/jam untuk jarak pendek...", "habitat": "Padang rumput, peternakan, dan daerah terbuka.", "makanan": "Rumput, jerami, biji-bijian.", "lingkungan": "Menyukai area terbuka, lapang, dan aman.", "bg": "https://i.ibb.co/SJjQ6gQ/horse-bg.jpg"},
    4: {"nama": "Ayam", "fakta": "Ayam bisa mengenali hingga 100 wajah berbeda...", "habitat": "Peternakan, halaman rumah, dan lingkungan pedesaan.", "makanan": "Jagung, biji-bijian, serangga, dan pelet ayam.", "lingkungan": "Menyukai lingkungan hangat dengan tempat bertelur dan berjemur.", "bg": "https://i.ibb.co/mz6z3Xt/chicken-bg.jpg"},
    5: {"nama": "Kupu-kupu", "fakta": "Kupu-kupu memiliki indra penciuman di kakinya...", "habitat": "Taman, hutan, padang bunga, dan daerah tropis.", "makanan": "Nektar bunga, buah matang.", "lingkungan": "Menyukai lingkungan dengan banyak bunga dan sinar matahari.", "bg": "https://i.ibb.co/r3qkGJY/butterfly-bg.jpg"},
    6: {"nama": "Laba-laba", "fakta": "Laba-laba membuat jaring untuk menangkap mangsa...", "habitat": "Rumah, kebun, hutan, dan sudut gelap.", "makanan": "Serangga kecil seperti lalat, nyamuk, dan ngengat.", "lingkungan": "Menyukai tempat yang aman, gelap, dan banyak serangga.", "bg": "https://i.ibb.co/TkFBrsj/spider-bg.jpg"}
}

# ==========================
# UI
# ==========================
st.title("🧠 Image Classification & Object Detection App")

menu = st.sidebar.selectbox("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])

uploaded_file = st.file_uploader("Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# Default background hewan random
# ==========================
random_bg = animal_info[random.choice(list(animal_info.keys()))]["bg"]
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{random_bg}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ==========================
# Jika ada file diupload
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Gambar yang Diupload", use_container_width=True)

    if menu == "Deteksi Objek (YOLO)":
        results = yolo_model(img)
        result_img = results[0].plot()
        st.image(result_img, caption="Hasil Deteksi", use_container_width=True)

    elif menu == "Klasifikasi Gambar":
        # Preprocessing
        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Prediksi
        prediction = classifier.predict(img_array)
        class_index = np.argmax(prediction)
        st.write("### Hasil Prediksi:", class_index)
        st.write("Probabilitas:", np.max(prediction))

        # Jika info hewan tersedia, tampilkan dan ganti background
        if class_index in animal_info:
            info = animal_info[class_index]
            bg_url = info["bg"]

            st.markdown(
                f'''
                <style>
                .stApp {{
                    background-image: url("{bg_url}");
                    background-size: cover;
                    background-position: center;
                    background-repeat: no-repeat;
                    background-attachment: fixed;
                }}
                </style>
                ''',
                unsafe_allow_html=True
            )

            st.write("### Nama Hewan:", info["nama"])
            st.write("**Fakta Menarik:**", info["fakta"])
            st.write("**Habitat:**", info["habitat"])
            st.write("**Makanan:**", info["makanan"])
            st.write("**Lingkungan:**", info["lingkungan"])
        else:
            st.write("Informasi tambahan untuk hewan ini belum tersedia.")
