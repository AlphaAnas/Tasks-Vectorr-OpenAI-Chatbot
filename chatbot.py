# ---------------- Base class ----------------
from openai import OpenAI


class BaseChatbot:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def query(self, prompt: str):
        raise NotImplementedError("Subclasses must implement this method")