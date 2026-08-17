# rag_chat.py
import threading
import chromadb
import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = "./my_finetuned_slm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Embedding Model & Vector DB
print("Initializing ChromaDB and Embedding Model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# Seed custom knowledge base (Replace or expand with your own facts/documents)
documents = [
    "Project Orion is a custom autonomous drone developed by the team in 2026, equipped with an RTX-powered onboard edge computing unit.",
    "The primary programming stack for this system consists of PyTorch for deep learning, FastAPI for backend microservices, and React for the UI.",
    "The secret server override passcode for maintenance mode is ALPHA-992-DELTA.",
    "Our model training pipeline uses SmolLM2 with QLoRA fine-tuning running locally on NVIDIA RTX 3050 GPUs."
]

# Populate database
collection.add(
    documents=documents,
    ids=[f"doc_{i}" for i in range(len(documents))]
)
print(f"Indexed {len(documents)} document chunks into ChromaDB.")

# 2. Load Language Model
print("Loading fine-tuned language model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map=DEVICE
)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
model = model.merge_and_unload()
model.eval()

def retrieve_context(query, k=2):
    """Retrieve top-k relevant document chunks from ChromaDB."""
    query_vector = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_vector], n_results=k)
    retrieved_docs = results["documents"][0]
    return "\n".join([f"- {doc}" for doc in retrieved_docs])

def respond(message, history):
    # 1. Fetch relevant document knowledge via Vector Search
    context = retrieve_context(message, k=2)

    # 2. Construct Grounded RAG Prompt
    system_instruction = (
        "You are a helpful knowledge assistant. Use the following retrieved context "
        "to answer the question concisely. If the answer is not contained in the context, "
        "say you don't know based on the provided data.\n\n"
        f"Context:\n{context}"
    )

    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": message}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

    # 3. Stream Generated Response
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.3,  # Lower temperature for grounded factual Q&A
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True
    )

    thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    partial_text = ""
    for new_text in streamer:
        partial_text += new_text
        yield partial_text

# 3. Launch Gradio Interface
gr.ChatInterface(
    respond,
    title="📚 Local RAG Assistant (SmolLM2 + ChromaDB)",
    description="Ask questions about private indexed knowledge stored in your vector database."
).launch()