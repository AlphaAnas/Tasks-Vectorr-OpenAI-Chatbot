	# 1c. Image generation using OpenAI
 	# 	Use case: Allow text input with text and image output



'''
1. TRY CATCH
2. guardrails
3. UI On huggingface spaces
4. Allow multiple users to connect simulataneously

'''


import base64
from openai import APIError, OpenAI
from dotenv import load_dotenv
import os
from openai import RateLimitError, BadRequestError
import gradio as gr
import requests

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=API_KEY)


def generate_image(prompt: str):

    if (not prompt) or (not prompt.strip()):
        print("Prompt is empty or invalid.")
        return None, "Error, prompt cannot be empty."

    try:
        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt,
            n=1,
            size='256x256',
            # quality="standard", # standard is the only option for dall-e-2
        )

        url = response.data[0].url
        img_response = requests.get(url)

        image_path = "generated_image.png"

        with open(image_path, "wb") as img_file:
            img_file.write(img_response.content)
        return image_path, "Image generated successfully."
    


 # --- Guardrail 2: Specific OpenAI Error Handling ---
    except BadRequestError as e:
        # This can happen if the prompt is rejected by the safety system.
        print(f"OpenAI BadRequestError: {e}")
        return None, f"Error: Your request was rejected. This might be due to a safety policy violation. Please modify your prompt. (Details: {e})"
    except RateLimitError as e:
        print(f"OpenAI RateLimitError: {e}")
        return None, "Error: You have exceeded your API usage limit. Please check your OpenAI account."
    except APIError as e:
        print(f"OpenAI APIError: {e}")
        return None, f"Error: An unexpected error occurred with the OpenAI API. Please try again later. (Details: {e})"

    except Exception as e:
        # Catch any other unexpected errors.
        print(f"An unexpected error occurred: {e}")
        return None, f"An unexpected error occurred: {e}"



with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# OPENAI Image Generation")
    gr.Markdown("Enter a text prompt to generate an image using OpenAI Image Generation Model.")

    with gr.Row():
        # Input component
        prompt_input = gr.Textbox(
            label="Prompt",
            placeholder="e.g., A cute baby sea otter in a top hat",
            lines=2
        )
    with gr.Row():
        # Action button
        submit_button = gr.Button("Generate Image", variant="primary")

    with gr.Row():
        # Output components
        image_output = gr.Image(label="Generated Image")
        status_output = gr.Textbox(label="Status", interactive=False)


    # Connect the button click to the core logic function
    submit_button.click(
        fn=generate_image,
        inputs=prompt_input,
        outputs=[image_output, status_output]
    )

if __name__ == "__main__":
    # The launch() method creates and runs the web server.
    demo.launch()