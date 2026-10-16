import json
import re
import torch
from json_repair import repair_json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from agent_core.tools import execute_tool

SYSTEM_PROMPT = (
    "You are an autonomous coding agent. Always output strictly ONE valid JSON object per turn "
    "with keys: 'thought', 'tool', and 'args'. Available tools: 'write_file', 'read_file', 'execute_code', 'finish'.\n"
    "Important: When using write_file, ensure all Python strings, brackets, and quotes are properly closed."
)
BASE_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
ADAPTER_PATH = "./agent_adapter"

print("Loading Fine-Tuned Agent onto RTX 3050...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

def extract_and_repair_json(text: str) -> dict:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
        if match:
            text = match.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        repaired = repair_json(text, return_objects=False)
        return json.loads(repaired)

def run_agent(task_description: str, max_turns: int = 8):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_description}
    ]
    print(f"\n{'='*60}\nTASK: {task_description}\n{'='*60}")

    last_observation = ""

    for step in range(1, max_turns + 1):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Use temperature sampling to prevent identical repeated code loops on errors
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        raw_output = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        ).strip()

        print(f"\n[Step {step}]")
        try:
            action = extract_and_repair_json(raw_output)
        except Exception:
            print("  [Warning]: Invalid format. Retrying...")
            messages.append({"role": "assistant", "content": raw_output})
            messages.append({
                "role": "user",
                "content": json.dumps({"observation": "Error: Output must be strictly valid JSON with 'thought', 'tool', 'args'."})
            })
            continue

        thought = action.get("thought", "")
        tool = action.get("tool", "")
        args = action.get("args", {})

        print(f"  Thought: {thought}")
        print(f"  Tool:    {tool}")
        print(f"  Args:    {args}")

        # Guardrail: Prevent finishing while code still has errors
        if tool == "finish":
            if "EXECUTION_ERROR" in last_observation:
                print("  [Guardrail]: Blocked premature finish. Requesting code patch...")
                messages.append({"role": "assistant", "content": json.dumps(action)})
                messages.append({
                    "role": "user",
                    "content": json.dumps({"observation": f"The script still has errors:\n{last_observation}\nPlease use write_file to fix syntax/logic errors before finishing."})
                })
                continue

            print(f"\nTASK COMPLETED: {args.get('summary', 'Finished.')}")
            break

        # Execute tool locally inside ./agent_workspace
        observation = execute_tool(tool, args)
        last_observation = observation
        print(f"  Result:  {observation}")

        messages.append({"role": "assistant", "content": json.dumps(action)})
        messages.append({"role": "user", "content": json.dumps({"observation": observation})})

if __name__ == "__main__":
    task = input("\nEnter task (or press Enter for default): ").strip()
    if not task:
        task = "Write a script log_parser.py that extracts all IPv4 addresses from a log block and counts uniques."
    run_agent(task)