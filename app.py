import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os
import gdown

st.set_page_config(page_title="Human Detector", page_icon="🤖")

# --- 1. CẤU HÌNH ID FILE MỚI ---
# ⚠️ QUAN TRỌNG: Thay ID này bằng ID của file 'human_detector_new.h5' bạn vừa train xong
file_id = '11l2Rh27p97monvzduZ_UeMaMx8-DfCRo' 
url = f'https://drive.google.com/uc?id={file_id}'
output = 'human_detector.h5' # Tên file lưu trên server (giữ nguyên cũng được)

# --- 2. HÀM TẢI & LOAD MODEL ---
@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(output):
        with st.spinner('Đang tải Model mới từ Drive...'):
            gdown.download(url, output, quiet=False)
            
    # Load model
    model = tf.keras.models.load_model(output)
    return model

try:
    model = load_model_from_drive()
except Exception as e:
    st.error("Chưa thay ID mới hoặc chưa bật quyền Share 'Anyone with link' cho file trong Drive.")
    st.stop()

# --- 3. HÀM DỰ ĐOÁN (Đã thêm fix lỗi ảnh PNG) ---
def import_and_predict(image_data, model):
    size = (150, 150)
    # .convert('RGB') giúp tránh lỗi nếu ảnh có 4 kênh màu (PNG trong suốt)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS).convert('RGB')
    img = np.asarray(image)
    img = img / 255.0
    img_reshape = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img_reshape)
    return prediction

# --- 4. GIAO DIỆN ---
st.title(" Web Nhận Diện Con Người")
st.write("Upload ảnh để kiểm tra ")

file = st.file_uploader("Chọn ảnh...", type=["jpg", "png", "jpeg"])

if file is None:
    st.text("Vui lòng upload ảnh")
else:
    image = Image.open(file)
    st.image(image, use_container_width=True)
    
    if st.button("Kiểm tra ngay"):
        pred = import_and_predict(image, model)
        score = pred[0][0] # Giá trị từ 0.0 đến 1.0
        
        st.write(f"Raw Score: {score}")

        # --- 5. LOGIC CHUẨN (Folder 1 là Human) ---
        if score > 0.5:
            st.success(f" ĐÂY LÀ CON NGƯỜI ")
            st.balloons()
        else:
            st.error(f" KHÔNG PHẢI NGƯỜI ")

