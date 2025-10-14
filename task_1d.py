# 1d. Image editing using OpenAI
#  		Use case: Allow text + image input with text+ image output



'''
1. STREAMING
2. GUARDRAILS
3. UI On huggingface spaces
4. Allow multiple users to connect simulataneously
'''


# For gpt-image-1, each image should be a png, webp, or jpg file less than 50MB. You can provide up to 16 images.

# For dall-e-2, you can only provide one image, and it should be a square png file less than 4MB.

import base64
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)



prompt = """
Generate a photorealistic image of a gift basket on a white background 
labeled 'Relax & Unwind' with a ribbon and handwriting-like font, 
containing all the items in the reference pictures.
"""

result = client.images.edit(
    model="gpt-image-1",
    image=[
        open("input_imgs/body_lotion.jpg", "rb"),

    ],
    prompt=prompt
)

image_base64 = result.data[0].b64_json
image_bytes = base64.b64decode(image_base64)

# Save the image to a file
with open("gift-basket.png", "wb") as f:
    f.write(image_bytes)
