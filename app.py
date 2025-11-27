import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import gdown
from torchvision.models.segmentation import deeplabv3_resnet50

# =====================
# Model Loading
# =====================
@st.cache_resource
def load_model(weights_path="best_deeplabv3_oilspill.pth", drive_id=None):
    if not os.path.exists(weights_path) and drive_id is not None:
        url = f"https://drive.google.com/uc?id={drive_id}"
        gdown.download(url, weights_path, quiet=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build DeepLabV3 model (2 classes: background + oil spill)
    model = deeplabv3_resnet50(weights=None, num_classes=2)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

# =====================
# Preprocessing & Prediction
# =====================
def preprocess_image(uploaded_file, image_size=(256, 256)):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(image_size[0], image_size[1]),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2()
    ])
    transformed = transform(image=image)
    tensor = transformed["image"].unsqueeze(0)
    return image, tensor

def predict_oil_spill(model, tensor, device):
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)['out']  # DeepLabV3 returns a dict
        probs = torch.softmax(outputs, dim=1)
        mask = torch.argmax(probs, dim=1).cpu().squeeze().numpy()
    return mask

# =====================
# Visualization
# =====================
def overlay_prediction(image, mask):
    mask_resized = cv2.resize(mask.astype(np.uint8),
                              (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

    overlay = image.copy()
    overlay[mask_resized == 1] = [255, 0, 0]  # red overlay for oil spill
    blended = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    return blended

# =====================
# Streamlit App UI
# =====================
def main():
    st.title("🛢️ AI SpillGuard - Oil Spill Detection")
    st.write("Upload an aerial image to detect oil spills using a trained DeepLabV3 model.")

    # Your Drive ID for DeepLabV3 model
    DRIVE_ID = "1kmgwoK6D9IdzG3rWkkYfLirTsqXgFjEL"

    model, device = load_model("best_deeplabv3_oilspill.pth", drive_id=DRIVE_ID)

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        original, tensor = preprocess_image(uploaded_file)
        mask = predict_oil_spill(model, tensor, device)
        overlayed = overlay_prediction(original, mask)

        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original Image", use_container_width=True)
        with col2:
            st.image(overlayed, caption="Prediction Overlay", use_container_width=True)

if __name__ == "__main__":
    main()

