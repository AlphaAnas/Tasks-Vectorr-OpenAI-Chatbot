# task_02 - conversational chatbot (simple RAG version, no LangChain)

import os
import json
from uuid import uuid4
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from chatbot import BaseChatbot


load_dotenv()
os.makedirs("chat_memory", exist_ok=True)

# ---------------- Sample Documents ----------------
docs = [
    "Nestle is a global food and beverage company headquartered in Switzerland.",
    "Nestle's product range includes baby food, bottled water, cereals, coffee, tea, dairy products, ice cream, pet foods, and snacks.",
    "Our company is certified under ISO 22000 and HACCP standards.",
    "We produce organic snacks using natural ingredients.",
    "Food safety management ensures hygiene and compliance with regulations.",
]


# ---------------- Dense Retriever (FAISS) ----------------
class DenseRetriever:
    def __init__(self, client, texts):
        self.client = client
        self.texts = texts
        self.embeddings = self._embed_texts(texts)
        self.index = self._build_index(self.embeddings)

    def _embed_texts(self, texts):
        embeds = [
            self.client.embeddings.create(model="text-embedding-3-small", input=t).data[0].embedding
            for t in texts
        ]
        return np.array(embeds, dtype="float32")

    def _build_index(self, embeddings):
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    def get_relevant_documents(self, query, k=2):
        q_emb = self.client.embeddings.create(model="text-embedding-3-small", input=query).data[0].embedding
        q_emb = np.array([q_emb], dtype="float32")
        D, I = self.index.search(q_emb, k)
        return [self.texts[i] for i in I[0]]


# ---------------- Sparse Retriever (BM25-like TF-IDF) ----------------
class SparseRetriever:
    def __init__(self, texts):
        self.texts = texts
        self.vectorizer = TfidfVectorizer().fit(texts)
        self.vectors = self.vectorizer.transform(texts)

    def get_relevant_documents(self, query, k=2):
        q_vec = self.vectorizer.transform([query])
        scores = (self.vectors @ q_vec.T).toarray().ravel()
        top_indices = np.argsort(scores)[::-1][:k]
        return [self.texts[i] for i in top_indices if scores[i] > 0]


# ---------------- Hybrid Retriever ----------------
class HybridRetriever:
    def __init__(self, sparse, dense, alpha=0.5):
        self.sparse = sparse
        self.dense = dense
        self.alpha = alpha

    def get_relevant_documents(self, query):
        s_docs = self.sparse.get_relevant_documents(query)
        d_docs = self.dense.get_relevant_documents(query)
        combined = list({d: None for d in s_docs + d_docs}.keys())
        return combined


# ---------------- Conversational Chatbot ----------------
class ConversationalChatbot(BaseChatbot):
    def __init__(self, api_key, retriever):
        super().__init__(api_key)
        self.retriever = retriever
        self.client = OpenAI(api_key=api_key)

    def generate_answer(self, question, history):
        retrieved_docs = self.retriever.get_relevant_documents(question)
        context = "\n".join(retrieved_docs)

        system_prompt = (
            "You are a helpful assistant for Nestle. "
            "Use only the provided context to answer. "
            "If the answer isn't in context, say 'I do not know.'\n\n"
            f"Context:\n{context}"
        )

        messages = [{"role": "system", "content": system_prompt}] + history + [
            {"role": "user", "content": question}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during generation: {e}")
            return "Error occurred."

    def get_history_path(self, session_id):
        return os.path.join("chat_memory", f"history_{session_id}.json")

    def load_history(self, session_id):
        path = self.get_history_path(session_id)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return []
        return []

    def save_history(self, session_id, history):
        path = self.get_history_path(session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    def summarize_history(self, history):
        if len(history) < 6:
            return history
        text = " ".join([h["content"] for h in history[:-4]])
        messages = [
            {"role": "system", "content": "Summarize this conversation briefly:"},
            {"role": "user", "content": text},
        ]
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=200,
            )
            summary = response.choices[0].message.content
            return [{"role": "system", "content": f"Summary of previous chat: {summary}"}] + history[-4:]
        except Exception as e:
            print(f"Summarization error: {e}")
            return history


# ---------------- Main ----------------
def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set.")
        return

    client = OpenAI(api_key=api_key)
    dense_retriever = DenseRetriever(client, docs)
    sparse_retriever = SparseRetriever(docs)
    hybrid_retriever = HybridRetriever(sparse_retriever, dense_retriever)

    chatbot_instance = ConversationalChatbot(api_key, hybrid_retriever)

    session_id = str(uuid4())
    history = chatbot_instance.load_history(session_id)

    print(f"Nestle Chatbot | Session ID: {session_id}")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye.")
            break

        answer = chatbot_instance.generate_answer(user_input, history)
        print(f"Bot: {answer}\n")

        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": answer})

        history = chatbot_instance.summarize_history(history)
        chatbot_instance.save_history(session_id, history)


if __name__ == "__main__":
    main()
