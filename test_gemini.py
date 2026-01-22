"""
Test script for Gemini 1.5 Flash integration
"""
import os
import google.generativeai as genai

# Get API key from environment
api_key = os.environ.get("GEMINI_API_KEY", "AIzaSyCknYGBTAfwMGJyBmtHR8KfMv-C6OOPodI")

print("=" * 60)
print("GEMINI 2.5 FLASH TEST")
print("=" * 60)

print(f"\n[OK] API Key: {api_key[:20]}...")

# Configure Gemini
try:
    genai.configure(api_key=api_key)
    print("[OK] Gemini configured successfully")
except Exception as e:
    print(f"[ERROR] Failed to configure Gemini: {e}")
    exit(1)

# Create model
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    print("[OK] Model 'gemini-2.5-flash' loaded")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Test simple generation
print("\n" + "-" * 60)
print("Testing simple generation...")
print("-" * 60)

try:
    response = model.generate_content("Say 'Hello! Gemini is working!' in a friendly way.")
    print(f"\n[OK] Response received:\n")
    # Handle Windows encoding issues
    try:
        print(response.text)
    except UnicodeEncodeError:
        print(response.text.encode('ascii', 'ignore').decode('ascii'))
except Exception as e:
    print(f"[ERROR] Generation failed: {e}")
    exit(1)

# Test streaming
print("\n" + "-" * 60)
print("Testing streaming generation...")
print("-" * 60)

try:
    print("\n[OK] Streaming response:\n")
    response = model.generate_content(
        "Write a 2-sentence summary of what Gemini 2.5 Flash is good for.",
        stream=True
    )

    for chunk in response:
        if chunk.text:
            try:
                print(chunk.text, end='', flush=True)
            except UnicodeEncodeError:
                print(chunk.text.encode('ascii', 'ignore').decode('ascii'), end='', flush=True)

    print("\n")
except Exception as e:
    print(f"[ERROR] Streaming failed: {e}")
    exit(1)

# Test with temperature and max tokens (matching our API server config)
print("\n" + "-" * 60)
print("Testing with custom generation config...")
print("-" * 60)

try:
    response = model.generate_content(
        "Explain in one sentence why Gemini 2.5 Flash has a 1 million token context window.",
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=100,
        )
    )
    try:
        print(f"\n[OK] Response:\n{response.text}")
    except UnicodeEncodeError:
        print(f"\n[OK] Response:\n{response.text.encode('ascii', 'ignore').decode('ascii')}")
except Exception as e:
    print(f"[ERROR] Custom config test failed: {e}")
    exit(1)

print("\n" + "=" * 60)
print("[SUCCESS] ALL TESTS PASSED! Gemini 2.5 Flash is ready to use!")
print("=" * 60)
print("\nYou can now:")
print("1. Start your API server: python api_server.py")
print("2. Open the chat UI: http://localhost:8000/chat")
print("3. Select 'Gemini 2.5 Flash' from the model dropdown")
print("4. Start chatting!")
