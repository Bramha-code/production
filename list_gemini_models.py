"""List available Gemini models"""
import os
import google.generativeai as genai

api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCknYGBTAfwMGJyBmtHR8KfMv-C6OOPodI")
genai.configure(api_key=api_key)

print("Available Gemini models:")
print("=" * 60)

for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"\nModel: {model.name}")
        print(f"  Display Name: {model.display_name}")
        print(f"  Description: {model.description}")
        print(f"  Supported methods: {model.supported_generation_methods}")
