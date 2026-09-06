"""
test_client.py — Beginner-friendly way to test your running API.

Instead of remembering curl commands, just run this file and type your
question when asked.

Make sure app.py is ALREADY RUNNING in another terminal before you run this
(uvicorn app:app --reload), otherwise this will fail to connect.

Run it:
    python test_client.py
"""

import requests

API_URL = "http://localhost:8000/ask"

print("=== GenAI RAG Assistant — Test Client ===")
print("Type your question and press Enter. Type 'quit' to exit.\n")

while True:
    question = input("Your question: ").strip()
    if question.lower() in ("quit", "exit"):
        break
    if not question:
        continue

    try:
        response = requests.post(API_URL, json={"question": question})
        response.raise_for_status()
        data = response.json()

        print("\n--- Answer ---")
        print(data["answer"])
        print(f"\n(Sources used: {data['sources']})")
        print(f"(Chunks retrieved: {data['retrieved_chunk_count']})\n")

    except requests.exceptions.ConnectionError:
        print("\n[Error] Could not connect to the API.")
        print("Did you run 'uvicorn app:app --reload' in another terminal first?\n")
        break
    except Exception as e:
        print(f"\n[Error] Something went wrong: {e}\n")
