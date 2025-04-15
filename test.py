import streamlit as st
import tensorflow as tf
import sqlite3
from PIL import Image
import numpy as np
import os
import uuid  

with open("index.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# MobileNet modelini yuklash
model = tf.keras.models.load_model('animal_model_mobilenet.h5')

# SQLite ma'lumotlar bazasini yaratish va ulanish
def create_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    animal TEXT,
                    predicted_label TEXT,
                    confidence REAL,
                    image_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()



# Bashoratni ma'lumotlar bazasiga saqlash
def save_prediction(animal, predicted_label, confidence, image_path):
    confidence = float(confidence)
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (animal, predicted_label, confidence, image_path)
        VALUES (?, ?, ?, ?)""",
        (animal, predicted_label, confidence, image_path))
    conn.commit()
    conn.close()




# Bashoratlarni olish
def get_predictions():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 5")  # So'nggi 5 ta bashoratni olish
    rows = c.fetchall()
    conn.close()
    return rows

# Ma'lumotlar bazasini yaratish
create_db()


# Streamlit interfeysi
st.title("Animal image classification")
# st.write("Rasmni yuklab, modelni ishlatib, bashorat oling.")


# Foydalanuvchi rasmni yuklash
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

os.makedirs("images", exist_ok=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Rasmni saqlash
    unique_filename = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join("images", unique_filename)
    image.save(image_path)

    # Rasmni model uchun tayyorlash
    img_array = np.array(image.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_labels = ["Cat", "Horse", "Tiger", "Lion", "Jaguar", "Wolf"]
    predicted_label = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Result: {predicted_label}")
    st.info(f"Accuracy: {confidence * 100:.2f}%")

    # Bashoratni saqlash (rasm yo'li bilan)
    save_prediction(predicted_label, predicted_label, confidence, image_path)

# So'nggi 5 ta bashoratni ko'rsatish
with st.sidebar:
    st.title("Last 5 prediction history:")
    predictions = get_predictions()




    for prediction in predictions:
        id = prediction[0]
        animal_name = prediction[1]
        label = prediction[2]
        confidence_value = float(prediction[3])
        image_path = prediction[4]
        timestamp = prediction[5]

        
        st.markdown(f"""
            <div class="prediction-item">
                <p><strong>🔹 ID:</strong> {id}</p>
                <p><strong>🔹 Tur:</strong> {animal_name}</p>
                <p><strong>🔹 Bashorat:</strong> {label}</p>
                <p><strong>🔹 Ishonch:</strong> {confidence_value*100:.1f}%</p>
                <p><strong>🔹 Vaqt:</strong> {timestamp}</p>
            </div>
        """, unsafe_allow_html=True)

        # Rasmni ko‘rsatish
        if image_path and os.path.exists(image_path):
            st.image(image_path, caption="Bashorat qilingan rasm", use_column_width=True)
        else:
            st.warning("Rasm topilmadi yoki yo‘q.")



