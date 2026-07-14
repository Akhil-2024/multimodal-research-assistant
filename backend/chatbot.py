from openai import OpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv("../.env")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

print("===== Multi-Modal Research Assistant =====")
print("Type 'exit' to quit\n")

MEMORY_FILE = "memory.json"

# Load previous memory
if os.path.exists(MEMORY_FILE):

    with open(MEMORY_FILE, "r") as file:
        messages = json.load(file)

else:

    messages = [
        {
            "role": "system",
            "content": "You are a concise AI research assistant. Answer briefly."
        }
    ]

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    try:
        recent_messages = [messages[0]] + messages[1:][-6:]

        response = client.chat.completions.create(
           model="gpt-4.1-mini",
           messages=recent_messages,
           max_tokens=120
        )

        answer = response.choices[0].message.content

        messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        # Save memory
        with open(MEMORY_FILE, "w") as file:
            json.dump(messages, file, indent=4)

        print("\nAssistant:", answer)
        print()

    except Exception as e:
        print("\nError:", e)
        print()