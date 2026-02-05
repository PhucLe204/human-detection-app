import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import gdown

st.set_page_config(page_title="Human Detector")

# --- CẤU HÌNH GOOGLE DRIVE ---
file_id = '15CXWjx2bMR2BnuPEOl-kgSJ2GL1jadjc' 
url = f'https://drive.google.com/uc?id={file_id}'
output = 'human_detector.h5'

# --- HÀM TẢI VÀ LOAD MODEL ---
@st.cache_resource
def load_model_from_drive():
    # Kiểm tra nếu file chưa tồn tại thì tải về
    if not os.path.exists(output):
        with st.spinner('Đang tải model từ Google Drive (124MB)... Vui lòng đợi...'):
            gdown.download(url, output, quiet=False)
            st.success("Tải model thành công!")
    
    # Load model sau khi đã có file
    model = tf.keras.models.load_model(output)
    return model

st.title(" Web Nhận Diện Con Người")

# Gọi hàm load model (Tự động tải nếu chưa có)
try:
    model = load_model_from_drive()
except Exception as e:
    st.error(f"Lỗi không tải được Model. Kiểm tra lại ID Google Drive nhé! Chi tiết: {e}")
    st.stop()

# --- PHẦN DỰ ĐOÁN (GIỮ NGUYÊN) ---
def import_and_predict(image_data, model):
    size = (150, 150)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    img = img / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

file = st.file_uploader("Upload ảnh để kiểm tra", type=["jpg", "png", "jpeg"])

if file is None:
    st.info("Vui lòng chọn ảnh.")
else:
    image = Image.open(file)
    st.image(image, caption="Ảnh upload", use_container_width=True)
    
    if st.button("Kiểm tra"):
        pred = import_and_predict(image, model)
        score = pred[0][0]
        
        st.write(f"Raw Score: {score}")
        
        if score < 0.5:
            st.success(" ĐÂY LÀ CON NGƯỜI")
            st.balloons()
        else:
            st.error(" KHÔNG PHẢI NGƯỜI")