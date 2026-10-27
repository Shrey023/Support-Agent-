import os 
import json
from dotenv import load_dotenv
from google import genai
from tools import get_ticket_status

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise valueError("GEMINI_API_KEY is not found in env")

client = genai.Client(api_key=api_key)

chat = client.chats.create(
    model="gemini-3.8-flash",
    config={"tools": [get_ticket_status]}
)

response = chat.send_message(
    "What is the status of ticket 999? Use the tool to look it up."
)

# data = json.loads(response.text)
# print(data["role"])

# print(get_ticket_status(101))
# print(get_ticket_status(102))
# print(get_ticket_status(999))
print(response.text)
