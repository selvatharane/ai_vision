import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import numpy as np
import cv2
from io import BytesIO
import os

# ===========================
# 1. Page Config
# ===========================
st.set_page_config(page_title="AI Vision Extraction", layout="wide")

# ===========================
# 2. Custom Styling
# ===========================
custom_style = """
<style>
/* ======== Animated Background ======== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(270deg, #cce5ff, #e0ccff, #ffd6e8, #ccf0e8);
    background-size: 800% 800%;
    animation: bgAnimation 20s ease infinite;
}
@keyframes bgAnimation {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ======== Sidebar ======== */
[data-testid="stSidebar"] {
    background-color: #2C3E50;
    color: white;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] li, [data-testid="stSidebar"] ol {
    color: #f1f5f9 !important;
}
button[title="Collapse"] {
    color: white !important;
}

/* ======== Header & Text ======== */
h1, h2, h3, h4 {
    color: #0f172a !important;
    text-align: center;
    font-weight: 800;
}
p, li {
    color: #1e293b !important;
    font-size: 1rem;
}

/* ======== Image Styling ======== */
img {
    display: block;
    margin-left: auto;
    margin-right: auto;
    border-radius: 18px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
}

/* ======== Buttons ======== */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    border: none;
    padding: 0.7rem 1.5rem;
    border-radius: 10px;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(124,58,237,0.4);
    transition: all 0.3s ease-in-out;
}
.stButton > button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #1e40af, #6d28d9);
}

/* ======== File Uploader ======== */
[data-testid="stFileUploader"] {
    text-align: center !important;
}

/* ======== Columns & Container Centering ======== */
.block-container {
    max-width: 1000px;
    margin: auto;
}
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)

# ===========================
# 3. App Header
# ===========================
st.title("✨ AI Vision Extraction")
st.markdown("<h4 style='text-align:center;'>Example: Object Extraction</h4>", unsafe_allow_html=True)

# ===========================
# 4. Sample Image (Centered)
# ===========================
sample_img = "sample_extraction.jpg"
if os.path.exists(sample_img):
    st.image(sample_img, caption=None, width=500)
else:
    st.warning("⚠️ Sample image not found! Please add `sample_extraction.jpg` to your repo.")

# ===========================
# 5. Sidebar Info
# ===========================
st.sidebar.markdown("""
## 📘 About This App
AI Vision Extraction allows you to automatically isolate the main object from any image using an advanced **DeepLabV3+** segmentation model.

---

### 🪄 How to Use
1. **Upload** an image (JPG, JPEG, or PNG).  
2. The **AI extracts** the main object and removes the background.  
3. **Preview & download** your extracted image instantly.  

💡 *Works best with clear images where the subject is distinct.*
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
# 8. Upload & Predict
# ===========================
st.markdown("<h3 style='text-align:center;'>🧠 Try It Yourself</h3>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload your image below 👇", type=["jpg", "jpeg", "png"])

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
