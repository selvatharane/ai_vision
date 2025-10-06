import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import cv2
import base64
from io import BytesIO
import os

# ===========================
# 1. Page Setup
# ===========================
st.set_page_config(page_title="AI Vision Extraction", layout="wide")

# ===========================
# 2. Custom Styling
# ===========================
custom_style = """
<style>
/* Animated background gradient */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg, #f0f4ff, #d6e0ff, #f0f4ff, #e1e9ff);
    background-size: 400% 400%;
    animation: gradientShift 12s ease infinite;
}
@keyframes gradientShift {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172554 0%, #1e3a8a 50%, #1e40af 100%);
    color: #e2e8f0 !important;
    padding-right: 0.5rem;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, 
[data-testid="stSidebar"] li, [data-testid="stSidebar"] ol {
    color: #f1f5f9 !important;
}
[data-testid="stSidebarNav"] button, [data-testid="stSidebarCollapseControl"] button {
    color: white !important;
}

/* Card style for main content */
.glass-card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0px 4px 25px rgba(0, 0, 0, 0.1);
}

/* Button styling */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #3b82f6);
    color: white;
    border: none;
    padding: 0.6rem 1.2rem;
    font-weight: 600;
    border-radius: 10px;
    transition: 0.3s ease-in-out;
    box-shadow: 0px 3px 10px rgba(59,130,246,0.4);
}
.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #1e40af, #2563eb);
}

/* Reduce image width & shadow */
img {
    border-radius: 14px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
}
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# ===========================
# 3. App Header
# ===========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.title("✨ AI Vision Extraction")
st.markdown("#### Example: Object Extraction")
st.markdown('</div>', unsafe_allow_html=True)

# ===========================
# 4. Display Sample Image
# ===========================
sample_img = "sample_extraction.jpg"
if os.path.exists(sample_img):
    st.image(sample_img, caption=None, width=450)
else:
    st.warning("⚠️ Sample image not found! Please add `sample_extraction.jpg` to your repo.")

# ===========================
# 5. Sidebar Content
# ===========================
st.sidebar.markdown("""
## 📘 About This App
AI Vision Extraction allows you to automatically isolate the main object from any image using an advanced **DeepLabV3+** segmentation model.

---

### 🪄 How to Use
1. **Upload** an image (JPG, JPEG, or PNG).  
2. The **AI extracts** the main object and removes the background.  
3. **Preview & download** your extracted image instantly.  

💡 *Works best with clear, well-lit images where the subject is distinct.*
""")

# ===========================
# 6. Load Model
# ===========================
@st.cache_resource
def load_model(model_path="best_deeplab.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

try:
    model, device = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    model, device = None, None

# ===========================
# 7. Helper Functions
# ===========================
def preprocess_image(image: Image.Image):
    img = image.convert("RGB")
    img = np.array(img)
    img_resized = cv2.resize(img, (256, 256))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return img_tensor, img

def predict_mask(model, device, image: Image.Image):
    img_tensor, orig = preprocess_image(image)
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        output = model(img_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
    mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))
    mask = (mask > 0.5).astype(np.uint8)
    result = orig * mask[:, :, np.newaxis]
    return result.astype(np.uint8)

def convert_to_bytes(image_np):
    img = Image.fromarray(image_np)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ===========================
# 8. Upload & Predict Section
# ===========================
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 🧠 Try It Yourself")

uploaded_file = st.file_uploader("Upload your image here:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    input_img = Image.open(uploaded_file)
    result = predict_mask(model, device, input_img)

    col1, col2 = st.columns(2)
    with col1:
        st.image(input_img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(result, caption="Extracted Object", use_container_width=True)

    img_bytes = convert_to_bytes(result)
    st.download_button(
        label="📥 Download Extracted Image",
        data=img_bytes,
        file_name="extracted_object.png",
        mime="image/png"
    )
st.markdown('</div>', unsafe_allow_html=True)
