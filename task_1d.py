import base64
import os
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)

MODEL = "gpt-image-1"  # "dall-e-2"

def edit_image(prompt: str, image_path):
    if not image_path:
        return "Error: Please provide a Valid Image Path"

    if not (prompt and prompt.strip()):
        return "Error: Prompt cannot be empty!"

    if not os.path.exists(image_path):
        return "Error: Image path does not exist!"

    if not image_path.lower().endswith(('.png', '.jpg', '.webp')):
        return "Error: Unsupported image format. Please use PNG, JPG, or WEBP."
    
    if MODEL == "gpt-image-1" and os.path.getsize(image_path) > 50 * 1024 * 1024:
        return "Error: Image size exceeds 50MB limit for gpt-image-1."
    if MODEL == "dall-e-2" and os.path.getsize(image_path) > 4 * 1024 * 1024:
        return "Error: Image size exceeds 4MB limit for dall-e-2."

    try:
        result = client.images.edit(
            model=MODEL,
            image=open(image_path, "rb"),
            prompt=prompt,
            n=1,
        )

        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        output_path = "output_image.png"
        with open(output_path, "wb") as f:
            f.write(image_bytes)

        return output_path

    except Exception as e:
        return f"Unknown Error Occurred: {e}"

def gradio_interface(prompt, image):
    if image is None:
        return "Please upload an image."
    temp_path = "temp_input.png"
    image.save(temp_path)
    return edit_image(prompt, temp_path)

iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Textbox(label="Enter your edit prompt"),
        gr.Image(label="Upload Image", type="pil"),
    ],
    outputs=gr.Image(label="Edited Image"),
    title="OpenAI Image Editor",
    description="Edit images with OpenAI's GPT-Image-1 or DALL-E-2 using natural language prompts.",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch(share=True)
