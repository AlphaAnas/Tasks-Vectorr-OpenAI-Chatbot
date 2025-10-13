# task 02 - conversational chatbot 

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os

load_dotenv()


BASE_URL = "NESTLE URL"
BASE_DIR = "sample_nestle.txt"

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    question: str

def scrape_text(url: str) -> str:
    text = ""
    if not url.startswith("http"):
        # its a local file path
        with open(url, 'r', encoding='utf-8') as file:
            text = file.read()
    else: # it's a web URL

        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")
        # remove scripts and styles
        for tag in soup(["script", "style"]):
            tag.extract()
        text = " ".join(soup.get_text().split())
        
    return text[:6000]  # truncate to fit context limit

@app.post("/chat")
async def chat_with_nestleAI(request: ChatRequest):
    site_text = scrape_text(BASE_DIR)
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=f"Use the following website content to answer:\n\n{site_text}\n\nQuestion: {request.question}"
    )
    return {"answer": response.output_text}
