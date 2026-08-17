# test_server.py
import json
import httpx

SERVER_URL = "http://127.0.0.1:8000"

def test_health():
    print("\n" + "=" * 50)
    print("1. Testing /health Endpoint")
    print("=" * 50)
    response = httpx.get(f"{SERVER_URL}/health")
    print("Status Code:", response.status_code)
    print("Response   :", response.json())

def test_chat_stream():
    print("\n" + "=" * 50)
    print("2. Testing /api/chat (Real-Time Token Stream)")
    print("=" * 50)
    payload = {"message": "Explain what an LLM is in one short sentence."}
    
    # httpx.stream reads SSE chunks line-by-line as tokens arrive
    with httpx.stream("POST", f"{SERVER_URL}/api/chat", json=payload, timeout=30.0) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "token" in data:
                    print(data["token"], end="", flush=True)
    print()

def test_rag_stream():
    print("\n" + "=" * 50)
    print("3. Testing /api/rag (ChromaDB Retrieval + Grounding)")
    print("=" * 50)
    payload = {"message": "What is the secret server override passcode?"}
    
    with httpx.stream("POST", f"{SERVER_URL}/api/rag", json=payload, timeout=30.0) as response:
        for line in response.iter_lines():
            if line.startswith("data: "):
                data = json.loads(line[6:])
                if "sources" in data:
                    print(f"[Retrieved Source Chunks]:\n{data['sources']}\n")
                    print("[Model Grounded Answer]: ", end="")
                elif "token" in data:
                    print(data["token"], end="", flush=True)
    print()

def test_agent():
    print("\n" + "=" * 50)
    print("4. Testing /api/agent (ReAct Tool Execution)")
    print("=" * 50)
    payload = {"message": "Calculate (15 * 12) + 450 and give me the result."}
    response = httpx.post(f"{SERVER_URL}/api/agent", json=payload, timeout=30.0)
    print("Status Code:", response.status_code)
    print("Output     :\n", json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_health()
    test_chat_stream()
    test_rag_stream()
    test_agent()