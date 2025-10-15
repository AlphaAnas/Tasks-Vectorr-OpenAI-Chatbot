from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")


app = FastAPI()
client = OpenAI()




class PromptRequest(BaseModel):
    prompt: str


@app.post("/call_chatbot")
async def call_chatbot(request: PromptRequest):

    response = client.responses.create(
        model="gpt-4.1",
        input=str(request.prompt)
    )
    # print("Response:", response)
    return {"response": response.output_text}












