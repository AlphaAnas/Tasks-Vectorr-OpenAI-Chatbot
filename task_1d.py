# 1d. Image editing using OpenAI
#  		Use case: Allow text + image input with text+ image output



'''
0. Image INput with text prompt or only image
1. STREAMING
2. GUARDRAILS

3. UI On huggingface spaces
4. Allow multiple users to connect simulataneously
'''


# For gpt-image-1, each image should be a png, webp, or jpg file less than 50MB. You can provide up to 16 images.

# For dall-e-2, you can only provide one image, and it should be a square png file less than 4MB.

import gradio as gr
import base64
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

MODEL = "gpt-image-1"  # or "dall-e-2"

def edit_image_gradio(prompt, image):
    if not prompt or not prompt.strip():
        return "Error: Prompt cannot be empty!", None
    if image is None:
        return "Error: Please upload an image.", None

    # Save temp image
    temp_path = "temp_input_image.png"
    image.save(temp_path)

    if not temp_path.lower().endswith(('.png', '.jpg', '.webp')):
        return "Error: Unsupported image format.", None
    if MODEL == "gpt-image-1" and os.path.getsize(temp_path) > 50 * 1024 * 1024:
        return "Error: Image size exceeds 50MB limit for gpt-image-1.", None
    if MODEL == "dall-e-2" and os.path.getsize(temp_path) > 4 * 1024 * 1024:
        return "Error: Image size exceeds 4MB limit for dall-e-2.", None

    try:
        result = client.images.edit(
            model=MODEL,
            image=open(temp_path, "rb"),
            prompt=prompt,
            n=1,
        )
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        output_path = "output_image.png"
        with open(output_path, "wb") as f:
            f.write(image_bytes)
        return "Success", output_path
    except Exception as e:
        return f"Error: {str(e)}", None

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("## OpenAI Image Editor")
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Describe your desired edit...")
            image_input = gr.Image(label="Upload Image", type="pil")
            submit_btn = gr.Button("Generate Edit")
        with gr.Column():
            status = gr.Textbox(label="Status")
            output_image = gr.Image(label="Edited Image")

    submit_btn.click(edit_image_gradio, inputs=[prompt, image_input], outputs=[status, output_image])

if __name__ == "__main__":
    demo.launch(share=True)
