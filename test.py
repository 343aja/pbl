import streamlit as st
import tensorflow as tf
import sqlite3
from PIL import Image
import numpy as np
import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# CSS faylni yuklash
with open("index.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Modelni yuklash
model = tf.keras.models.load_model('animal_model_mobilenet_1.h5')

# Bazani yaratish
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

def save_prediction(animal, predicted_label, confidence, image_path):
    confidence = float(confidence)
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("""INSERT INTO predictions (animal, predicted_label, confidence, image_path)
                 VALUES (?, ?, ?, ?)""", (animal, predicted_label, confidence, image_path))
    conn.commit()
    conn.close()

def get_predictions():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 5")
    rows = c.fetchall()
    conn.close()
    return rows

def get_all_predictions():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df

create_db()

# 📌 Tabs yaratish
tab1, tab2,tab3= st.tabs(["Animal Image Classification", "EDA (Exploratory Data Analysis)","EDA from Dataset"])

# 🔹 Tab 1: Upload
with tab1:
    st.title("Animal Image Classification")
    st.markdown('<h4 class="header">Cat, Horse, Jaguar, Lion, Tiger, Wolf</h4>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload image", type=["jpg", "png"])

    os.makedirs("images", exist_ok=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        unique_filename = f"{uuid.uuid4().hex}.png"
        image_path = os.path.join("images", unique_filename)
        image.save(image_path)

        img_array = np.array(image.resize((128, 128))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_labels = ["Cat", "Horse", "Jaguar", "Lion","Tiger","Wolf"]
        predicted_label = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction)

        st.success(f"Result: {predicted_label}")
        st.info(f"Accuracy: {confidence * 100:.2f}%")

        save_prediction(predicted_label, predicted_label, confidence, image_path)

# 🔹 Tab 2: EDA
with tab2:
    st.title("EDA (Exploratory Data Analysis)")

    df = get_all_predictions()

    if not df.empty:
        st.dataframe(df)

        # Countplot
        st.subheader("Prediction Count by Animal")
        fig1, ax1 = plt.subplots()
        sns.countplot(data=df, x='predicted_label', order=df['predicted_label'].value_counts().index, ax=ax1)
        ax1.set_xlabel("Animal")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

         # 3. Vaqt bo‘yicha bashoratlar soni
        st.subheader("Number of predictions by time")
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_chart = alt.Chart(df).mark_line(point=True).encode(
            x='timestamp:T',
            y='count():Q',
            tooltip=['timestamp', 'count()']
        ).properties(width=600, height=400)
        st.altair_chart(time_chart)

         # 2. O‘rtacha ishonch darajasi
        st.subheader("Average confidence level (%)")
        avg_conf = df['confidence'].astype(float).mean() * 100
        st.metric(label="Average confidence", value=f"{avg_conf:.2f}%")
        # Confidence distribution
        st.subheader("Confidence Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df['confidence'], bins=10, kde=True, ax=ax2)
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
    else:
        st.info("No prediction data available yet.")

# tab3, = st.tabs(["EDA from Dataset Folder"])

with tab3:
    st.title("EDA from Dataset Folder")

    dataset_path = "dataset/train"

    if os.path.isdir(dataset_path):
        image_data = []

        for class_name in os.listdir(dataset_path):
            class_folder = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_folder):
                for filename in os.listdir(class_folder):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(class_folder, filename)

                        try:
                            image = Image.open(file_path)
                            width, height = image.size
                            file_size_kb = os.path.getsize(file_path) / 1024  # KB
                            aspect_ratio = round(width / height, 2)

                            image_data.append({
                                'Class': class_name,
                                'Width': width,
                                'Height': height,
                                'FileSizeKB': file_size_kb,
                                'AspectRatio': aspect_ratio,
                                'Path': file_path
                            })
                        except Exception as e:
                            st.warning(f"Could not read image {file_path}: {e}")

        df_eda = pd.DataFrame(image_data)

        if not df_eda.empty:
            st.subheader("1. Image Size Distribution (Width & Height)")
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_eda, x='Class', y='Width', ax=ax1)
            ax1.set_title("Width Distribution by Class")
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_eda, x='Class', y='Height', ax=ax2)
            ax2.set_title("Height Distribution by Class")
            st.pyplot(fig2)

            st.subheader("2. File Size Distribution (KB)")
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=df_eda, x='Class', y='FileSizeKB', ax=ax3)
            ax3.set_title("File Size Distribution by Class")
            st.pyplot(fig3)

            st.subheader("3. Aspect Ratio (Width / Height)")
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            sns.violinplot(data=df_eda, x='Class', y='AspectRatio', ax=ax4)
            ax4.set_title("Aspect Ratio Distribution by Class")
            st.pyplot(fig4)

            st.subheader("4. Dominant Colors (Top 10)")
            from collections import Counter

            def get_dominant_color(image_path):
                image = Image.open(image_path).resize((50, 50))
                pixels = np.array(image).reshape(-1, 3)
                most_common = Counter([tuple(pixel) for pixel in pixels]).most_common(1)[0][0]
                return '#%02x%02x%02x' % most_common

            df_eda['DominantColor'] = df_eda['Path'].apply(get_dominant_color)

            top_colors = df_eda['DominantColor'].value_counts().nlargest(10)

            fig5, ax5 = plt.subplots(figsize=(8, 4))
            bars = ax5.bar(top_colors.index, top_colors.values, color=top_colors.index)
            ax5.set_title("Top 10 Dominant Colors")
            ax5.set_ylabel("Count")
            st.pyplot(fig5)

        else:
            st.warning("No image data found in subfolders.")
    else:
        st.warning("The folder path is invalid or doesn't exist.")

# 🔹 Tab 3: History
with st.sidebar:
    st.title("Last 5 Predictions")

    if st.button("Clear History"):
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute("DELETE FROM predictions")
        conn.commit()
        conn.close()

        # Rasm fayllarini ham o‘chirish
        import glob
        image_files = glob.glob("images/*.png")
        for file in image_files:
            os.remove(file)

        st.success("History has been cleared!")

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
           st.image(image_path, caption="Predicted picture", use_column_width=True)

        else:
            st.warning("Image not found or missing.")
