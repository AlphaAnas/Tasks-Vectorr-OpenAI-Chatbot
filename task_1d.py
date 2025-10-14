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

import base64
import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


MODEL = "gpt-image-1"  # "dall-e-2"

def edit_image(prompt: str, image_path):

    if (not (image_path or image_path.strip())):
        return "Error: Please provide a Valid Image Path"

    if (not (prompt or prompt.strip())):
        return "Error: Prompt cannot be empty!"

    if (not os.path.exists(image_path)):  # think if this were in a server than user will send an online image path or a json base64 string
        return "Error: Image path does not exist!"

    if not image_path.lower().endswith(('.png', '.jpg', '.webp')):
        return "Error: Unsupported image format. Please use PNG, JPG, or WEBP."
    
    # File size max 50MB for gpt-image-1 and 4MB for dall-e-2 and 16 images for gpt-image-1
    if (MODEL == "gpt-image-1" and os.path.getsize(image_path) > 50 * 1024 * 1024):
        return "Error: Image size exceeds 50MB limit for gpt-image-1."
    if (MODEL == "dall-e-2" and os.path.getsize(image_path) > 4 * 1024 * 1024):
        return "Error: Image size exceeds 4MB limit for dall-e-2."



    try:
        print("PATH:", image_path)
        result = client.images.edit(
            model=MODEL,
            image=
                open(image_path, "rb"), # LATER CORRECT THIS TO HANDLE MULTIPLE IMAGES FOR gpt-image-1

            
            prompt=prompt
        )

        print("Result:", result)
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)

        # Save the image to a file
        with open("output_image.png", "wb") as f:
            f.write(image_bytes)





    except Exception as e:
        return "Unknown Error Occured: ",e 

if __name__ == "__main__":
    # Example usage
    # Ensure you have a valid image path
    response = edit_image("A cat in the basket", "gift-basket.png")
    print("Response:", response)

