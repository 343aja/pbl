import streamlit as st
import tensorflow as tf
import sqlite3
from PIL import Image
import numpy as np
import os
import uuid  
import pandas as pd
import altair as alt

# CSS yuklash
with open("index.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Modelni yuklash
model = tf.keras.models.load_model('animal_model_mobilenet.h5')

# SQLite DB yaratish
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

# Bashoratni DB ga saqlash
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

# So'nggi bashoratlarni olish
def get_predictions():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 5")
    rows = c.fetchall()
    conn.close()
    return rows

# Ma'lumotlarni pandas dataframe ga yuklash
def load_data():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df

# DB yaratish
create_db()

# Streamlit UI
st.title("Animal Image Classification")
st.markdown('<h4 class="header">Cat, Horse, Jaguar, Lion, Tiger, Wolf</h4>', unsafe_allow_html=True)

# Rasm yuklash
uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

os.makedirs("images", exist_ok=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    unique_filename = f"{uuid.uuid4().hex}.png"
    image_path = os.path.join("images", unique_filename)
    image.save(image_path)

    img_array = np.array(image.resize((128, 128))) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_labels = ["Cat", "Horse", "Tiger", "Jaguar", "Lion", "Wolf"]
    predicted_label = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.success(f"Result: {predicted_label}")
    st.info(f"Accuracy: {confidence * 100:.2f}%")

    save_prediction(predicted_label, predicted_label, confidence, image_path)

# So‘nggi bashoratlar
with st.sidebar:
    st.title("Last 5 Prediction History")
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
                <p><strong>🔹 Animal name:</strong> {animal_name}</p>
                <p><strong>🔹 Prediction:</strong> {label}</p>
                <p><strong>🔹 Accuracy:</strong> {confidence_value*100:.1f}%</p>
                <p><strong>🔹 Time:</strong> {timestamp}</p>
            </div>
        """, unsafe_allow_html=True)

        if image_path and os.path.exists(image_path):
            st.image(image_path, caption="Predicted picture", use_container_width=True)
        else:
            st.warning("Image not found or missing.")

# EDA TAHLIL 📊
st.title("📊 EDA Tahlil: Bashorat qilingan ma'lumotlar")

df = load_data()

if df.empty:
    st.warning("Hozircha hech qanday bashorat mavjud emas.")
else:
    # 1. Har bir hayvon nechta marta bashorat qilingan
    st.subheader("1. Hayvonlar bo‘yicha bashoratlar soni")
    count_chart = alt.Chart(df).mark_bar().encode(
        x='predicted_label:N',
        y='count():Q',
        tooltip=['predicted_label', 'count()']
    ).properties(width=600, height=400)
    st.altair_chart(count_chart)

    # 2. O‘rtacha ishonch darajasi
    st.subheader("2. O'rtacha ishonch darajasi (%)")
    avg_conf = df['confidence'].astype(float).mean() * 100
    st.metric(label="O'rtacha ishonch", value=f"{avg_conf:.2f}%")

    # 3. Vaqt bo‘yicha bashoratlar soni
    st.subheader("3. Bashoratlar soni vaqt bo‘yicha")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    time_chart = alt.Chart(df).mark_line(point=True).encode(
        x='timestamp:T',
        y='count():Q',
        tooltip=['timestamp', 'count()']
    ).properties(width=600, height=400)
    st.altair_chart(time_chart)

    # 4. Eng ko‘p bashorat qilingan hayvon
    st.subheader("4. Eng ko‘p bashorat qilingan hayvon")
    most_common = df['predicted_label'].value_counts().idxmax()
    st.success(f"🔝 Eng ko‘p bashorat qilingan hayvon: **{most_common}**")
