# ========================
# مشروع AI لتصنيف الصور - Colab + Streamlit + Flask + ngrok
# ========================

# 1️⃣ تثبيت المكتبات
!pip install streamlit flask tensorflow pillow pyngrok -q

# 2️⃣ رفع نموذجك best_model.h5
from google.colab import files
uploaded = files.upload()  # اختر ملف best_model.h5

# 3️⃣ إعداد authtoken الخاص بـ ngrok
from pyngrok import ngrok
ngrok.set_auth_token("32zSoBpfC4qDnggeDfG7s2mh1Un_4d4wzJTCBoMNHXESB1NUq")

# 4️⃣ إنشاء تطبيق Streamlit + Flask API
app_code = """
import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# تحميل النموذج
model = load_model("best_model.h5")

# أسماء الفئات
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# ================= Flask API =================
flask_app = Flask(__name__)

@flask_app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]
        img = Image.open(file).convert("L")
        img = img.resize((28, 28))
        img_array = np.array(img)/255.0
        img_array = img_array.reshape(1,28,28,1)

        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        class_name = class_names[class_index]
        confidence = float(np.max(predictions))

        return jsonify({"class": class_name, "confidence": round(confidence,2)})
    except Exception as e:
        return jsonify({"error": str(e)})

# تشغيل Flask في Thread
def run_flask():
    flask_app.run(port=5000)

Thread(target=run_flask).start()

# ================= Streamlit UI =================
st.title("🖼️ Image Classification AI System")
st.write("ارفع صورة ليقوم الذكاء الاصطناعي بتصنيفها 👇")

uploaded_file = st.file_uploader("اختر صورة ...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L")
    st.image(img, caption="الصورة التي رفعتها", use_column_width=True)

    img = img.resize((28,28))
    img_array = np.array(img)/255.0
    img_array = img_array.reshape(1,28,28,1)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    confidence = float(np.max(predictions))

    st.subheader("🔮 النتيجة:")
    st.write(f"الفئة: **{class_name}**")
    st.write(f"نسبة الثقة: **{confidence:.2f}**")
"""

# إنشاء ملف app.py
with open("app.py","w") as f:
    f.write(app_code)

# 5️⃣ فتح نفق ngrok وتشغيل Streamlit
import os
public_url = ngrok.connect(8501)
print("🌐 رابط التطبيق الخارجي:", public_url)

os.system("streamlit run app.py --server.port 8501 --server.headless true")
