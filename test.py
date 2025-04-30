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
import streamlit.components.v1 as components
import altair as alt

# CSS faylni yuklash
with open("index.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Modelni yuklash
try:
    model = tf.keras.models.load_model("model.h5", compile=False, custom_objects={})
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()


# Bazani yaratish
def create_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    animal TEXT,
                    predicted_label TEXT,
                    confidence REAL,
                    image_path TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
    conn.commit()
    conn.close()


def save_prediction(animal, predicted_label, confidence, image_path):
    confidence = float(confidence)
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute(
        """INSERT INTO predictions (animal, predicted_label, confidence, image_path)
                 VALUES (?, ?, ?, ?)""",
        (animal, predicted_label, confidence, image_path),
    )
    conn.commit()
    conn.close()


def get_predictions():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 5")
    rows = c.fetchall()
    conn.close()
    return rows


def get_all_predictions():
    conn = sqlite3.connect("predictions.db")
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    conn.close()
    return df


create_db()

# 📌 Tabs yaratish
tab1, tab2, tab3 = st.tabs(
    [
        "Animal Image Classification",
        "EDA (Exploratory Data Analysis)",
        "EDA from Dataset",
    ]
)

# 🔹 Tab 1: Upload
with tab1:
    st.title("Animal Image Classification")
    st.markdown(
        '<h4 class="header">Cat, Horse, Jaguar, Lion, Tiger, Wolf</h4>',
        unsafe_allow_html=True,
    )

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
        class_labels = ["Cat", "Horse", "Jaguar", "Lion", "Tiger", "Wolf"]
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
        sns.countplot(
            data=df,
            x="predicted_label",
            order=df["predicted_label"].value_counts().index,
            ax=ax1,
        )
        ax1.set_xlabel("Animal")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

        # 3. Vaqt bo‘yicha bashoratlar soni
        st.subheader("Number of predictions by time")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        time_chart = (
            alt.Chart(df)
            .mark_line(point=True)
            .encode(x="timestamp:T", y="count():Q", tooltip=["timestamp", "count()"])
            .properties(width=600, height=400)
        )
        st.altair_chart(time_chart)

        # 2. O‘rtacha ishonch darajasi
        st.subheader("Average confidence level (%)")
        avg_conf = df["confidence"].astype(float).mean() * 100
        st.metric(label="Average confidence", value=f"{avg_conf:.2f}%")
        # Confidence distribution
        st.subheader("Confidence Distribution")
        fig2, ax2 = plt.subplots()
        sns.histplot(df["confidence"], bins=10, kde=True, ax=ax2)
        ax2.set_xlabel("Confidence")
        ax2.set_ylabel("Frequency")
        st.pyplot(fig2)
    else:
        st.info("No prediction data available yet.")

# tab3, = st.tabs(["EDA from Dataset Folder"])
with tab3:
    st.title("Full EDA from Dataset Folder (with Chart.js)")

    dataset_path = "dataset/train"

    if os.path.isdir(dataset_path):
        data = []

        for class_name in os.listdir(dataset_path):
            class_folder = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_folder):
                for img_file in os.listdir(class_folder):
                    if img_file.endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(class_folder, img_file)
                        try:
                            img = Image.open(img_path)
                            img_size = os.path.getsize(img_path) / 1024  # KB
                            width, height = img.size

                            # Dominant color
                            img_array = np.array(img)
                            img_array = img_array.reshape(-1, 3)
                            dominant_color = tuple(np.mean(img_array, axis=0).astype(int))

                            data.append(
                                {
                                    "Class": class_name,
                                    "Width": width,
                                    "Height": height,
                                    "Size_KB": round(img_size, 2),
                                    "Dominant_Color": dominant_color,
                                }
                            )
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")

        if data:
            df = pd.DataFrame(data)

            # 1️⃣ Pie Chart - Image Count per Class
            class_counts = df["Class"].value_counts()
            labels = class_counts.index.tolist()
            counts = class_counts.values.tolist()

            st.subheader("Pie Chart - Image Count per Class")
            pie_chart = f"""
            <canvas id="pieChart"></canvas>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
            const ctxPie = document.getElementById('pieChart').getContext('2d');
            new Chart(ctxPie, {{
                type: 'pie',
                data: {{
                    labels: {labels},
                    datasets: [{{
                        data: {counts},
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.6)',
                            'rgba(54, 162, 235, 0.6)',
                            'rgba(255, 206, 86, 0.6)',
                            'rgba(75, 192, 192, 0.6)',
                            'rgba(153, 102, 255, 0.6)',
                            'rgba(255, 159, 64, 0.6)'
                        ],
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            position: 'top',
                        }},
                        title: {{
                            display: true,
                            text: 'Image Count per Class'
                        }}
                    }}
                }}
            }});
            </script>
            """
            components.html(pie_chart, height=800)

            # 2️⃣ Bar Chart - Average Image Size per Class
            avg_size = df.groupby("Class")["Size_KB"].mean().round(2)
            labels_size = avg_size.index.tolist()
            avg_sizes = avg_size.values.tolist()

            st.subheader("Bar Chart - Average Image File Size (KB) per Class")
            bar_chart_size = f"""
            <canvas id="barChartSize"></canvas>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
            const ctxBarSize = document.getElementById('barChartSize').getContext('2d');
            new Chart(ctxBarSize, {{
                type: 'bar',
                data: {{
                    labels: {labels_size},
                    datasets: [{{
                        label: 'Avg Size (KB)',
                        data: {avg_sizes},
                        backgroundColor: 'rgba(54, 162, 235, 0.6)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }},
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        title: {{
                            display: true,
                            text: 'Average Image File Size by Class'
                        }}
                    }}
                }}
            }});
            </script>
            """
            components.html(bar_chart_size, height=500)

            # 3️⃣ Dominant Colors Visualization
            st.subheader("Dominant Colors by Class")
            dominant_colors = df.groupby("Class")["Dominant_Color"].first().reset_index()

            color_blocks = ""
            for _, row in dominant_colors.iterrows():
                rgb = row["Dominant_Color"]
                hex_color = "#%02x%02x%02x" % rgb
                color_blocks += f"""
                <div style='display:inline-block; margin:10px; text-align:center;'>
                    <div style='width:80px; height:80px; background-color:{hex_color}; border-radius:10px;'></div>
                    <p style='margin-top:5px;'>{row['Class']}</p>
                </div>
                """

            st.markdown(color_blocks, unsafe_allow_html=True)

            # 4️⃣ Bubble Chart - Image Width vs Height
            st.subheader("Bubble Chart - Image Width vs Height")
            bubble_data = [
                {
                    "x": w,
                    "y": h,
                    "r": max(3, min(int(s / 50), 10)),
                }  # radius controlled by size
                for w, h, s in zip(df["Width"], df["Height"], df["Size_KB"])
            ]

            bubble_chart = f"""
            <canvas id="bubbleChart"></canvas>
            <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
            <script>
            const ctxBubble = document.getElementById('bubbleChart').getContext('2d');
            new Chart(ctxBubble, {{
                type: 'bubble',
                data: {{
                    datasets: [{{
                        label: 'Image Dimensions and Size',
                        data: {bubble_data},
                        backgroundColor: 'rgba(255, 99, 132, 0.5)'
                    }}]
                }},
                options: {{
                    scales: {{
                        x: {{
                            title: {{
                                display: true,
                                text: 'Width (px)'
                            }}
                        }},
                        y: {{
                            title: {{
                                display: true,
                                text: 'Height (px)'
                            }}
                        }}
                    }},
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Image Width vs Height with Size (Bubble)'
                        }}
                    }}
                }}
            }});
            </script>
            """
            components.html(bubble_chart, height=600)

        else:
            st.warning("No images found in the dataset.")

    else:
        st.warning("The folder path is invalid or doesn't exist.")
# 🔹 Tab 3: History
with st.sidebar:
    st.title("Last 5 Predictions")

    if st.button("Clear History"):
        conn = sqlite3.connect("predictions.db")
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

        st.markdown(
            f"""
            <div class="prediction-item">
                <p><strong>🔹 ID:</strong> {id}</p>
                <p><strong>🔹 Animal name:</strong> {animal_name}</p>
                <p><strong>🔹 Prediction:</strong> {label}</p>
                <p><strong>🔹 Accuracy:</strong> {confidence_value*100:.1f}%</p>
                <p><strong>🔹 Time:</strong> {timestamp}</p>
            </div>
        """,
            unsafe_allow_html=True,
        )

        if image_path and os.path.exists(image_path):
            st.image(image_path, caption="Predicted picture", use_column_width=True)

        else:
            st.warning("Image not found or missing.")
