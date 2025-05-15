import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import json

st.set_page_config(page_title="👗 FashionFit AI", layout="centered")
st.title("👗 FashionFit AI – Body Type Outfit Recommender")

# Load model
@st.cache_resource
def load_model():
    model = torch.load("bodytype_model/model.pt", map_location=torch.device('cpu'))
    model.eval()
    return model

# Dummy class labels
CLASS_NAMES = ["Hourglass", "Rectangle", "Inverted Triangle", "Pear"]

# Load recommendations
with open("recommendations.json", "r") as f:
    tips = json.load(f)

model = load_model()

# Image upload
uploaded_file = st.file_uploader("📤 Upload a full-body photo", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    input_tensor = transform(image).unsqueeze(0)
    
    with st.spinner("🤖 Analyzing body type..."):
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        body_type = CLASS_NAMES[predicted_class]

    st.success(f"✅ Detected Body Type: **{body_type}**")

    recs = tips[body_type]
    st.markdown("### 👚 Recommended Styles")
    st.markdown(", ".join([f"**{style}**" for style in recs["styles"]]))
    st.info(recs["tip"])

    if st.button("💾 Save Recommendation"):
        with open("recommendation.txt", "w") as f:
            f.write(f"Body Type: {body_type}\n")
            f.write("Recommended Styles: " + ", ".join(recs["styles"]) + "\n")
            f.write("Tip: " + recs["tip"])
        st.success("Saved to `recommendation.txt`")
