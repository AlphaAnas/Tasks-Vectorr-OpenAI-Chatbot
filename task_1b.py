# task 02 - conversational chatbot 

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import json

load_dotenv()


BASE_URL = "NESTLE URL"
BASE_DIR = "sample_nestle.txt"
HISTORY_FILE = "conversation_history.json"

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatRequest(BaseModel):
    question: str

def scrape_text(url: str) -> str:
    text = ""
    if not url.startswith("http"):
        # its a local file path
        try:
            with open(url, 'r', encoding='utf-8') as file:
                text = file.read()
        except FileNotFoundError:
            return ""
    else: # it's a web URL
        try:
            res = requests.get(url)
            res.raise_for_status()
            soup = BeautifulSoup(res.text, "html.parser")
            # remove scripts and styles
            for tag in soup(["script", "style"]):
                tag.extract()
            text = " ".join(soup.get_text().split())
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL: {e}")
            return ""
        
    return text[:6000]  # truncate to fit context limit

def load_history() -> list:
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []
    return []

def save_history(history: list):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)

def chatbot(question: str, context: str, history: list) -> str:
    system_message = {
        "role": 'system',
        "content": f'You are a helpful assistant for Nestle. Use the following context to answer questions: "{context}". If the answer is not in the context, say you do not know.',
    }
    
    messages = [system_message] + history + [{"role": "user", "content": question}]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        max_tokens=150,
        messages=messages,
    )
    
    return response.choices[0].message.content


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% APIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

@app.post("/chat")
async def chat_with_nestleAI(request: ChatRequest):
    site_text = scrape_text(BASE_DIR)
    history = load_history()

    answer = chatbot(request.question, site_text, history)
    
    # Update history
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": answer})
    
    # Keep history from getting too long
    if len(history) > 10:
        history = history[-10:]

    save_history(history)
 
    return {"answer": answer}
