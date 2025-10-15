import os
import json
from uuid import uuid4
from datetime import datetime, timezone
import ntplib
from dotenv import load_dotenv
from task_1c import generate_image
from task_1d import edit_image
from chatbot import BaseChatbot


# ---------------- Utility function ----------------
def get_internet_time():
    try:
        client = ntplib.NTPClient()
        response = client.request("pool.ntp.org", version=3)
        utc_time = datetime.fromtimestamp(response.tx_time, tz=timezone.utc)
        return str(utc_time)
    except Exception as e:
        return f"Could not sync with time server: {e}"


class ConversationalImageChatbot(BaseChatbot):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.model = "gpt-4o-mini"

        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "generate_image",
                    "description": "Generate an image from text prompt",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Prompt to generate an image"},
                        },
                        "required": ["prompt"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "edit_image",
                    "description": "Edit an existing image based on a text prompt",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Editing instructions"},
                            "image_path": {
                                "type": ["string", "array"],
                                "items": {"type": "string"},
                                "description": "Path(s) to the image(s) for editing",
                            },
                        },
                        "required": ["prompt", "image_path"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_internet_time",
                    "description": "Get the current accurate UTC time from NTP",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

    def query(self, prompt: str):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You can generate images, edit images, or return the internet time. Decide the best tool or answer directly.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=self.tools,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # Check for tool calls
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_call = msg.tool_calls[0]
                name = tool_call.function.name

                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    return "Invalid arguments for tool call."

                if name == "generate_image":
                    result = generate_image(**args)
                elif name == "edit_image":
                    result = edit_image(**args)
                elif name == "get_internet_time":
                    result = get_internet_time()
                else:
                    result = f"Unknown tool called: {name}"

                follow_up = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "user", "content": prompt},
                        msg,
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": str(result),
                        },
                    ],
                )
                return follow_up.choices[0].message.content

            # No tool call
            return msg.content or "No response from model."

        except Exception as e:
            return f"Error: {e}"


# ---------------- Main loop ----------------
def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set")
        return

    chatbot = ConversationalImageChatbot(api_key)
    print("Chatbot ready. Type a request to generate/edit an image or get the time.")

    while True:
        q = input("You: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        print("Bot:", chatbot.query(q))


if __name__ == "__main__":
    main()
