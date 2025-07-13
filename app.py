import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os

# Cache the model so it's not reloaded every time
@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)

    if torch.cuda.is_available():
        pipe.to("cuda")
    else:
        pipe.to("cpu")

    return pipe

pipe = load_model()

st.set_page_config(page_title="Pm.AI Text to Image", layout="centered")
st.title("🧠 WelCome To Pm.Ai Text to Image Generator ")

prompt = st.text_input("🎨 Enter your prompt:")

if st.button("🚀 Generate Image"):
    if not prompt:
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image... please wait..."):
            image = pipe(prompt).images[0]
            st.image(image, caption="Generated Image", use_column_width=True)
            image.save("output.png")
            with open("output.png", "rb") as f:
                st.download_button("💾 Download Image", f, file_name="output.png")
