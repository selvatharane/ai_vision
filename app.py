import streamlit as st
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import io

# ===========================
# Load DeepLabV3+ model
# ===========================
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1
    ).to(device)

    model.load_state_dict(torch.load("best_deeplab.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# ===========================
# Preprocess
# ===========================
def preprocess_image(image: Image.Image):
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# ===========================
# TTA Inference
# ===========================
def predict_with_tta(model, input_tensor):
    aug_list = [
        lambda x: x,  # original
        lambda x: torch.flip(x, [3]),  # horizontal
        lambda x: torch.flip(x, [2]),  # vertical
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # rotate 90
        lambda x: torch.rot90(x, k=3, dims=[2, 3])   # rotate 270
    ]
    outputs = []
    for f in aug_list:
        t = f(input_tensor)
        with torch.no_grad():
            out = model(t.to(device))
            out = torch.sigmoid(out)
            # undo transforms
            if f != aug_list[0]:
                if f == aug_list[1]: out = torch.flip(out, [3])
                elif f == aug_list[2]: out = torch.flip(out, [2])
                elif f == aug_list[3]: out = torch.rot90(out, k=3, dims=[2, 3])
                elif f == aug_list[4]: out = torch.rot90(out, k=1, dims=[2, 3])
            outputs.append(out.cpu())
    return torch.mean(torch.stack(outputs), dim=0)

# ===========================
# Postprocess mask
# ===========================
def postprocess_mask(mask_tensor):
    mask = mask_tensor.squeeze().numpy()
    mask = (mask * 255).astype(np.uint8)
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="AI Vision Extraction", layout="wide")
st.title("🖼️ AI VISION EXTRACTION")

uploaded_files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Read image
        img = Image.open(uploaded_file).convert("RGB")
        input_tensor = preprocess_image(img)

        # Inference
        mask = predict_with_tta(model, input_tensor)
        mask_np = postprocess_mask(mask)

        # Overlay subject on black bg
        img_resized = np.array(img.resize((256, 256)))
        extracted = img_resized.copy()
        extracted[mask_np == 0] = [0, 0, 0]

        # Show results side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(extracted, caption="Extracted Object", use_container_width=True)

        # Download option
        result_img = Image.fromarray(extracted)
        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        byte_im = buf.getvalue()
        st.download_button(
            label="⬇️ Download Extracted Image",
            data=byte_im,
            file_name=f"extracted_{uploaded_file.name}",
            mime="image/png"
        )
