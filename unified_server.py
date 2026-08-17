# unified_server.py
import json
import re
import threading
import chromadb
import numexpr as ne
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = "./my_finetuned_slm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(title="Unified LLM Server", version="1.0.0")

# Enable CORS for React/web frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# 1. Initialize Neural Network & Vector Database
# -------------------------------------------------------------
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map=DEVICE
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).merge_and_unload()
model.eval()

print("Initializing ChromaDB and Embedding Engine...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# Seed initial knowledge chunks
seed_docs = [
    "Project Orion is a custom autonomous drone developed in 2026, equipped with an RTX-powered onboard edge computing unit.",
    "The primary tech stack consists of PyTorch for deep learning, FastAPI for backend microservices, and React for the UI.",
    "The secret server override passcode for maintenance mode is ALPHA-992-DELTA.",
    "Expense Limit: Daily lunch meal reimbursement during client travel is capped at $45 per person without prior approval."
]
collection.add(
    documents=seed_docs,
    ids=[f"doc_{i}" for i in range(len(seed_docs))]
)

# -------------------------------------------------------------
# 2. Agent Tools
# -------------------------------------------------------------
def tool_calculate(expression: str) -> str:
    try:
        clean = expression.replace("^", "**")
        return f"{ne.evaluate(clean).item():g}"
    except Exception as e:
        return f"Error: {e}"

def tool_system_status(_: str = "") -> str:
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        return f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {allocated:.1f}MB / {total:.1f}MB"
    return "Running on CPU"

TOOLS = {
    "calculate": tool_calculate,
    "get_system_status": tool_system_status
}

AGENT_PROMPT = """You are an AI Agent with access to external tools.
Available tools:
- calculate: Evaluates mathematical formulas. Input must be a valid math string (e.g. '145 * 38').
- get_system_status: Returns GPU/VRAM health. Input is an empty string ''.

If you need a tool, reply ONLY with this exact JSON:
{"action": "tool_name", "action_input": "arguments"}

If you have the final answer, reply:
Final Answer: [Your complete response here]"""

# -------------------------------------------------------------
# 3. Schemas & Endpoints
# -------------------------------------------------------------
class QueryRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 200

class KnowledgeDoc(BaseModel):
    text: str
    doc_id: str

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "vram_mb": f"{torch.cuda.memory_allocated(0)/(1024**2):.1f}" if torch.cuda.is_available() else "0"
    }

# Endpoint 1: Direct Token Streaming Chat
@app.post("/api/chat")
async def chat_stream(req: QueryRequest):
    messages = [{"role": "user", "content": req.message}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    
    threading.Thread(target=model.generate, kwargs=gen_kwargs).start()

    def event_stream():
        for token in streamer:
            yield f"data: {json.dumps({'token': token})}\n\n"
            
    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Endpoint 2: RAG-Grounded Streaming Chat
@app.post("/api/rag")
async def rag_stream(req: QueryRequest):
    # Vector Search in ChromaDB
    query_vec = embedder.encode(req.message).tolist()
    search = collection.query(query_embeddings=[query_vec], n_results=2)
    retrieved_chunks = search["documents"][0] if search["documents"] else []
    context_str = "\n".join([f"- {c}" for c in retrieved_chunks])
    
    system_instruction = (
        "You are a factual knowledge assistant. Use this context to answer concisely. "
        "If not found in context, state that you do not know.\n\n"
        f"Context:\n{context_str}"
    )
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": req.message}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=req.max_tokens,
        do_sample=True,
        temperature=0.3,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )
    
    threading.Thread(target=model.generate, kwargs=gen_kwargs).start()

    def event_stream():
        # First send the retrieved sources to the frontend
        yield f"data: {json.dumps({'sources': retrieved_chunks})}\n\n"
        for token in streamer:
            yield f"data: {json.dumps({'token': token})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

# Endpoint 3: Agentic Tool Execution
@app.post("/api/agent")
def agent_execute(req: QueryRequest):
    history = f"User: {req.message}\n"
    thoughts = []
    
    for _ in range(3):
        messages = [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": history}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.2,
                pad_token_id=tokenizer.eos_token_id
            )
            
        resp = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        if "Final Answer:" in resp:
            final_ans = resp.split("Final Answer:")[-1].strip()
            return {"thoughts": thoughts, "final_answer": final_ans}
            
        json_match = re.search(r"\{.*?\}", resp, re.DOTALL)
        if json_match:
            try:
                tool_call = json.loads(json_match.group(0))
                tool_name = tool_call.get("action")
                tool_input = tool_call.get("action_input", "")
                
                if tool_name in TOOLS:
                    observation = TOOLS[tool_name](tool_input)
                    thoughts.append({"action": tool_name, "input": tool_input, "output": observation})
                    history += f"{resp}\nObservation: {observation}\n"
                else:
                    history += f"{resp}\nObservation: Tool '{tool_name}' not found.\n"
            except json.JSONDecodeError:
                return {"thoughts": thoughts, "final_answer": resp}
        else:
            return {"thoughts": thoughts, "final_answer": resp}
            
    return {"thoughts": thoughts, "final_answer": "Execution loop limit reached."}

# Endpoint 4: Dynamic Document Ingestion
@app.post("/api/documents")
def add_document(doc: KnowledgeDoc):
    collection.add(documents=[doc.text], ids=[doc.doc_id])
    return {"status": "success", "indexed_id": doc.doc_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)