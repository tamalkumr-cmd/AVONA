# agent.py
import json
import re
import numexpr as ne
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_DIR = "./my_finetuned_slm"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load Model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16, device_map=DEVICE)
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).merge_and_unload()
model.eval()

# 2. Define Available Tools
def calculate(expression: str) -> str:
    """Safely evaluates a math expression."""
    try:
        clean_expr = expression.replace("^", "**")
        result = ne.evaluate(clean_expr).item()
        return f"{result:g}"
    except Exception as e:
        return f"Error evaluating expression: {e}"

def get_system_status(_: str = "") -> str:
    """Returns local GPU/VRAM health."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        return f"GPU: {torch.cuda.get_device_name(0)} | VRAM Used: {allocated:.1f}MB / {total:.1f}MB"
    return "Running on CPU (No GPU detected)"

TOOL_REGISTRY = {
    "calculate": calculate,
    "get_system_status": get_system_status
}

# 3. Agent Execution Prompt
SYSTEM_PROMPT = """You are an AI Agent with access to external tools.
Available tools:
- calculate: Evaluates mathematical formulas. Input must be a valid mathematical expression string (e.g., '145 * 38').
- get_system_status: Returns GPU/VRAM runtime metrics. Input is an empty string ''.

If you need a tool, reply ONLY with this exact JSON format:
{"action": "tool_name", "action_input": "arguments"}

If you have the final answer, reply in this format:
Final Answer: [Your complete response here]"""

def run_agent(user_query: str, max_iterations: int = 3):
    conversation_history = f"User: {user_query}\n"
    
    for _ in range(max_iterations):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": conversation_history}
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
            
        response = tokenizer.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\n[Agent Thought/Action]:\n{response}")
        
        # Check if the agent returned the final answer
        if "Final Answer:" in response:
            return response.split("Final Answer:")[-1].strip()
            
        # Parse JSON tool call
        try:
            json_match = re.search(r"\{.*?\}", response, re.DOTALL)
            if json_match:
                tool_call = json.loads(json_match.group(0))
                tool_name = tool_call.get("action")
                tool_input = tool_call.get("action_input", "")
                
                if tool_name in TOOL_REGISTRY:
                    observation = TOOL_REGISTRY[tool_name](tool_input)
                    print(f"[Tool Execution ({tool_name})]: {observation}")
                    conversation_history += f"{response}\nObservation: {observation}\n"
                else:
                    conversation_history += f"{response}\nObservation: Unknown tool '{tool_name}'\n"
            else:
                return response
        except json.JSONDecodeError:
            return response
            
    return "Execution reached maximum loop limit."

if __name__ == "__main__":
    print("--- Test 1: Math Execution ---")
    ans1 = run_agent("Calculate 2^16 multiplied by 15.5")
    print(f"\nResult 1: {ans1}\n")
    
    print("--- Test 2: System Lookup ---")
    ans2 = run_agent("What is the current GPU and VRAM status?")
    print(f"\nResult 2: {ans2}\n")