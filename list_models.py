import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

print("Models available to your API key that support generateContent:\n")
for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(f"  {m.name}")
