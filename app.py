import streamlit as st
import torch
import segmentation_models_pytorch as smp
from PIL import Image
import torchvision.transforms as T
import numpy as np
import cv2
import os
from io import BytesIO

# ===========================
# 1. Page Setup
# ===========================
st.set_page_config(page_title="AI Vision Extractor", layout="centered")
st.title("🧠 AI Vision Extraction App")
st.caption("Upload an image to extract the object using DeepLabV3+ segmentation")

# ===========================
# 2. Show Sample Image
# ===========================
sample_img = "sample_extraction.jpeg"
if os.path.exists(sample_img):
    st.image(sample_img, caption="Example: Object Extraction", use_container_width=True)
else:
    st.warning("⚠️ Add a 'sample_extraction.jpeg' in your repo to show a preview here.")

# ===========================
# 3. Load Model
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
# 4. Helper Functions
# ===========================
def preprocess_image(image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
    ])
    return transform(image).unsqueeze(0)

# --- TTA Inference ---
def predict_with_tta(model, input_tensor):
    aug_list = [
        lambda x: x,
        lambda x: torch.flip(x, [3]),
        lambda x: torch.flip(x, [2]),
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),
        lambda x: torch.rot90(x, k=3, dims=[2, 3])
    ]
    outputs = []
    for f in aug_list:
        t = f(input_tensor)
        with torch.no_grad():
            out = model(t)
            out = torch.sigmoid(out)
            # Undo augmentations
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out, [3])
                elif f == aug_list[2]: out = torch.flip(out, [2])
                elif f == aug_list[3]: out = torch.rot90(out, k=3, dims=[2, 3])
                elif f == aug_list[4]: out = torch.rot90(out, k=1, dims=[2, 3])
            outputs.append(out)
    return torch.mean(torch.stack(outputs), dim=0)

# --- Postprocessing ---
def postprocess_mask(mask_tensor):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)
    return mask

# --- Overlay / Extraction ---
def extract_object(image, mask):
    image_np = np.array(image.convert("RGB"))
    mask_resized = cv2.resize(mask, (image_np.shape[1], image_np.shape[0]))
    output = image_np.copy()
    output[mask_resized == 0] = [0, 0, 0]  # black background
    return output

# ===========================
# 5. Upload & Predict
# ===========================
st.header("🖼️ Try Your Own Image")

uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    input_img = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔍 Extracting object... Please wait..."):
        input_tensor = preprocess_image(input_img).to(device)
        mask_pred = predict_with_tta(model, input_tensor)
        mask_np = postprocess_mask(mask_pred)
        extracted = extract_object(input_img, mask_np)

    # Display results
    st.success("✅ Extraction complete!")

    col1, col2 = st.columns(2)
    with col1:
        st.image(input_img, caption="Original Image", use_container_width=True)
    with col2:
        st.image(extracted, caption="Extracted Object", use_container_width=True)

    # Download button
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
