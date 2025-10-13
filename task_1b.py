# task 02 - conversational chatbot 

from uuid import uuid4
from fastapi import FastAPI, Request
from pydantic import BaseModel
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import json

load_dotenv()


BASE_URL = "NESTLE URL"
BASE_DIR = "sample_nestle.txt" # later we will use RAG or webscrapeging to get live data
HISTORY_FILE = "conversation_history.json"




app = FastAPI()

sessions = {}  # {session_id: history_list}



class ChatRequest(BaseModel):
    question: str

class ConversationalChatbot:


    def __init__(self, api_key: str, base_dir: str, history_file: str):
        self.client = OpenAI(api_key=api_key)
        self.base_dir = base_dir
        self.history_file = history_file

    def scrape_text(self, url: str) -> str:
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

    def get_history_path(self, session_id: str) -> str:
        return f"history_{session_id}.json"

    def load_history(self, session_id: str) -> list:
        path = self.get_history_path(session_id)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def save_history(self, session_id: str, history: list):
        path = self.get_history_path(session_id)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2)

    def summarize_text(self, text:str) -> str:
        messages = [
            {"role": "system", "content": "Summarize the following text:"},
            {"role": "user", "content": text}
        ]
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=350,
            n=1,
        )
        return response.choices[0].message.content

    def chatbot(self, question: str, context: str, history: list) -> str:
        system_message = {
            "role": 'system',
            "content": f'You are a helpful assistant for Nestle. Use the following context to answer questions: "{context}". If the answer is not in the context, say you do not know.',
        }
        
        messages = [system_message] + history + [{"role": "user", "content": question}]
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            max_tokens=150,
            messages=messages,
        )
        
        return response.choices[0].message.content


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% APIS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Instantiate the chatbot
chatbot_instance = ConversationalChatbot(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_dir=BASE_DIR,
    history_file=HISTORY_FILE
)

@app.post("/chat")
async def chat_with_nestleAI(req: Request, request_body: ChatRequest):

     # Identify user session
    session_id = req.headers.get("X-Session-ID")
    if not session_id:
        session_id = str(uuid4())  # new session if not provided
    
    # Load history for this session
    history = chatbot_instance.load_history(session_id)


    site_text = chatbot_instance.scrape_text(chatbot_instance.base_dir)

    answer = chatbot_instance.chatbot(request_body.question, site_text, history)
    
    # Update historyrequest_body
    history.append({"role": "user", "content": request_body.question})
    history.append({"role": "assistant", "content": answer})



    # WHY SUMMARIZE IS NOT WORKING

# Limit history size — summarize old content if needed
    if len(history) > 2:
        # combine older messages except last few
        old_text = " ".join([h["content"] for h in history[:-6]])
        summary = chatbot_instance.summarize_text(old_text)
        # keep summarized version + last few turns
        history = [{"role": "system", "content": f"Summary of previous chat: {summary}"}] + history[-6:]
        

    chatbot_instance.save_history(session_id, history)
 
    return {"answer": answer}
