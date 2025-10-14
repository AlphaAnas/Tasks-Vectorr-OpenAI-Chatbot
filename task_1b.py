# task 02 - conversational chatbot 

from uuid import uuid4
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import json

load_dotenv()



'''
1. USE RAG
2. MEMORY SHOULD BE SAVED IN FOLDER NEATLY
3. ALLOW MULTIPLE USERS TO CONNECT SIMULTANEOUSLY
4. SUMMARIZATION OF CHAT HISTORY WAS NOT WORKING

'''


BASE_DIR = "sample_nestle.txt" # later can be replaced with a URL when its a RAG

class ConversationalChatbot:
    def __init__(self, api_key: str, base_dir: str):
        self.client = OpenAI(api_key=api_key)
        self.base_dir = base_dir

    def scrape_text(self, url: str) -> str:
        text = ""
        if not url.startswith("http"):
            try:
                with open(url, 'r', encoding='utf-8') as file:
                    text = file.read()
            except FileNotFoundError:
                print(f"Error: The file {url} was not found.")
                return ""
        else:
            try:
                res = requests.get(url)
                res.raise_for_status()
                soup = BeautifulSoup(res.text, "html.parser")
                for tag in soup(["script", "style"]):
                    tag.extract()
                text = " ".join(soup.get_text().split())
            except requests.exceptions.RequestException as e:
                print(f"Error fetching URL: {e}")
                return ""
        return text[:6000]

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

    def summarize_text(self, text: str) -> str:
        if not text.strip():
            return ""
        messages = [
            {"role": "system", "content": "Summarize the following conversation concisely:"},
            {"role": "user", "content": text}
        ]
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                max_tokens=350,
                n=1,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during summarization: {e}")
            return ""

    def chatbot(self, question: str, context: str, history: list) -> str:
        system_message = {
            "role": 'system',
            "content": f'You are a helpful assistant for Nestle. Use the following context to answer questions: "{context}". If the answer is not in the context, say you do not know.',
        }
        messages = [system_message] + history + [{"role": "user", "content": question}]
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                max_tokens=150,
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling chatbot API: {e}")
            return "Sorry, I encountered an error."

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return

    chatbot_instance = ConversationalChatbot(api_key=api_key, base_dir=BASE_DIR)
    
    session_id = str(uuid4())
    history = chatbot_instance.load_history(session_id)
    print(f"Welcome to the Nestle Chatbot! Your session ID is {session_id}")
    print("Type 'exit' or 'quit' to end the session.")

    site_text = chatbot_instance.scrape_text(chatbot_instance.base_dir)
    if not site_text:
        print("Could not load context data. Exiting.")
        return

    while True:
        try:
            question = input("You: ")
            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            answer = chatbot_instance.chatbot(question, site_text, history)
            print(f"Bot: {answer}")
            
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

            if len(history) > 3:
                old_text = " ".join([h["content"] for h in history[:-4]])
                summary = chatbot_instance.summarize_text(old_text)
                if summary:
                    history = [{"role": "system", "content": f"Summary of previous chat: {summary}"}] + history[-4:]
            
            chatbot_instance.save_history(session_id, history)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    main()
