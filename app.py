import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
from io import BytesIO
import os

# ===========================
# 1. Page Setup
# ===========================
st.set_page_config(
    page_title="AI Vision Extractor", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# ===========================
# 2. Sidebar Content
# ===========================
st.sidebar.title("📘 About This App")
st.sidebar.info(
    """
    AI Vision Extraction allows you to automatically isolate the main object from any image 
    using an advanced **DeepLabV3+** segmentation model.
    """
)
st.sidebar.markdown("🎯 **How to Use**")
st.sidebar.markdown("""
1. Upload an image (JPG, JPEG, or PNG).  
2. The AI extracts the main object and removes the background.  
3. Preview & download your extracted image instantly.  

💡 Works best with clear images where the subject is distinct.
""")
st.sidebar.markdown("---")
# TTA toggle
tta_enabled = st.sidebar.checkbox("Enable TTA (slower but more accurate)", value=True)

# ===========================
# 3. Custom CSS (background, sidebar, scrollbars, hover effects)
# ===========================
st.markdown("""
<style>
/* Animated gradient background */
@keyframes gradientBG {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stApp {
    background: linear-gradient(270deg, #ff9a9e, #fad0c4, #a1c4fd, #c2e9fb);
    background-size: 800% 800%;
    animation: gradientBG 15s ease infinite;
    color: #000000;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
    color: #000000;
    font-weight: 500;
}
button[title="Toggle sidebar"] {
    color: #000000 !important;
    background-color: #e0e0e0 !important;
    border-radius: 5px;
    z-index: 9999;
    transition: all 0.3s ease;
}
button[title="Toggle sidebar"]:hover {
    background-color: #c0c0c0 !important;
    transform: scale(1.05);
}

/* Center only the sample image */
.center-img {
    display: flex;
    justify-content: center;
    margin: 20px 0;
}

/* Scrollbar customization */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-track { background: #f0f2f6; }
::-webkit-scrollbar-thumb { background: #c0c0c0; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #888888; }

/* Button hover effects */
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
    border-radius: 5px;
    transition: transform 0.2s, background-color 0.3s;
}
.stButton>button:hover {
    background-color: #45a049;
    transform: scale(1.05);
}

/* Headings and captions readable */
.css-1v3fvcr, .css-10trblm { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# ===========================
# 4. App Title
# ===========================
st.title("🧠 AI Vision Extraction App")
st.caption("Upload an image to extract the object using DeepLabV3+ segmentation")

# ===========================
# 5. Show Sample Image (centered)
# ===========================
sample_img = "sample_extraction.jpg"
if os.path.exists(sample_img):
    st.markdown('<div class="center-img">', unsafe_allow_html=True)
    st.image(sample_img, caption="Example: Object Extraction", use_container_width=False, width=400)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.warning("⚠️ Add a 'sample_extraction.jpg' in your repo to show a preview here.")

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
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    model, device = None, None

# ===========================
# 7. Helper Functions
# ===========================
def preprocess_image(image):
    transform = T.Compose([
        T.Resize((224, 224)),  # smaller = faster
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# Fast or TTA prediction
def predict(model, input_tensor, tta=True):
    if tta:
        aug_list = [
            lambda x: x,
            lambda x: torch.flip(x, [3]),
            lambda x: torch.flip(x, [2]),
            lambda x: torch.rot90(x, k=1, dims=[2, 3]),
            lambda x: torch.rot90(x, k=3, dims=[2, 3])
        ]
    else:
        aug_list = [lambda x: x, lambda x: torch.flip(x, [3])]  # faster

    outputs = []
    for f in aug_list:
        t = f(input_tensor)
        with torch.no_grad():
            out = model(t)
            out = torch.sigmoid(out)
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out, [3])
                elif f == aug_list[2]: out = torch.flip(out, [2])
                elif f == aug_list[3]: out = torch.rot90(out, k=3, dims=[2,3])
                elif f == aug_list[4]: out = torch.rot90(out, k=1, dims=[2,3])
            outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)

def postprocess_mask(mask_tensor):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

def extract_object(image, mask):
    image_np = np.array(image.convert("RGB"))
    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    output = image_np.copy()
    output[mask_resized==0] = [0,0,0]
    return output

# ===========================
# 8. Upload & Predict
# ===========================
st.header("🖼️ Try Your Own Image")
uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg","jpeg","png"])

if uploaded_file is not None and model is not None:
    input_img = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔍 Extracting object... Please wait..."):
        input_tensor = preprocess_image(input_img).to(device)
        mask_pred = predict(model, input_tensor, tta=tta_enabled)
        mask_np = postprocess_mask(mask_pred)
        extracted = extract_object(input_img, mask_np)

    st.success("✅ Extraction complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.image(input_img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(extracted, caption="Extracted Object", use_container_width=True)

    result_img = Image.fromarray(extracted)
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="📥 Download Extracted Image",
        data=byte_im,
        file_name="extracted_object.png",
        mime="image/png"
    )
