import streamlit as st
import numpy as np
from PIL import Image
import joblib
import time

# -------------------- Page --------------------
st.set_page_config(
    page_title="Male vs Female Classifier",
    page_icon="👨‍💻",
    layout="wide"
)

# -------------------- Theme --------------------
dark = st.sidebar.toggle("🌙 Dark Mode")

bg = "#0E1117" if dark else "#F4F6F9"
text = "white" if dark else "black"

st.markdown(f"""
<style>
.stApp {{
    background:{bg};
    color:{text};
}}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model --------------------
model = joblib.load("male_female_model.pkl")
IMG_SIZE = 64

# -------------------- Sidebar --------------------
st.sidebar.title("📋 Project Info")
st.sidebar.write("**Model:** Logistic Regression")
st.sidebar.write("**Classes:**")
st.sidebar.write("👩 Female")
st.sidebar.write("👨 Male")

# -------------------- Header --------------------
st.title("👨👩 Male vs Female Face Classifier")
st.caption("Upload a clear face image to predict gender using Machine Learning.")

uploaded_file = st.file_uploader(
    "📤 Upload Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------- Prediction --------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img).flatten()

    with st.spinner("Analyzing Image..."):
        time.sleep(1)
        prediction = model.predict([img])[0]
        probability = model.predict_proba([img])[0]

    if prediction == 0:
        result = "👩 Female"
        confidence = probability[0] * 100
    else:
        result = "👨 Male"
        confidence = probability[1] * 100

    with col2:
        st.subheader("🎯 Prediction")
        st.success(result)

        st.subheader("📊 Confidence")
        st.info(f"{confidence:.2f}%")

        st.toast("Prediction Completed ✅", icon="🎉")

        report = f"""
Male vs Female Prediction Report

Prediction: {result}

Confidence: {confidence:.2f}%
"""

        st.download_button(
            "📥 Download Report",
            report,
            file_name="Prediction_Report.txt"
        )

# -------------------- How it Works --------------------
with st.expander("ℹ️ How it Works"):
    st.write("""
- Upload a face image.
- The image is resized to **64 × 64** pixels.
- The Logistic Regression model predicts whether the face is **Male** or **Female**.
- The prediction and confidence score are displayed.
""")

st.markdown("---")
st.caption("❤️ Developed using Python • Streamlit • Scikit-learn")


