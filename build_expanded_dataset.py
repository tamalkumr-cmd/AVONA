import json
import os
import random
from datasets import load_dataset

SYSTEM_PROMPT = (
    "You are an autonomous coding agent. Always output strictly ONE valid JSON object per turn "
    "with keys: 'thought', 'tool', and 'args'. Available tools: 'write_file', 'read_file', 'execute_code', 'finish'."
)

def make_coding_traj(task, filename, code, error_msg, fixed_code, success_output, summary):
    run_cmd = f"g++ -O2 {filename} -o run.exe && run.exe" if filename.endswith(".cpp") else f"python {filename}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
        {"role": "assistant", "content": json.dumps({
            "thought": f"I will implement the solution in {filename} with complete verification tests.",
            "tool": "write_file",
            "args": {"path": filename, "content": code}
        })},
        {"role": "user", "content": json.dumps({"observation": f"Successfully wrote {len(code)} characters to {filename}"})},
        {"role": "assistant", "content": json.dumps({
            "thought": f"I will execute {filename} to verify test assertions.",
            "tool": "execute_code",
            "args": {"command": run_cmd}
        })}
    ]

    if error_msg and fixed_code:
        messages.extend([
            {"role": "user", "content": json.dumps({"observation": error_msg})},
            {"role": "assistant", "content": json.dumps({
                "thought": "Execution returned an error. Analyzing traceback and patching source code.",
                "tool": "write_file",
                "args": {"path": filename, "content": fixed_code}
            })},
            {"role": "user", "content": json.dumps({"observation": f"Successfully wrote {len(fixed_code)} characters to {filename}"})},
            {"role": "assistant", "content": json.dumps({
                "thought": "Re-executing the patched code to verify the fix.",
                "tool": "execute_code",
                "args": {"command": run_cmd}
            })},
            {"role": "user", "content": json.dumps({"observation": success_output})},
            {"role": "assistant", "content": json.dumps({
                "thought": "All assertions and execution checks passed.",
                "tool": "finish",
                "args": {"summary": summary}
            })}
        ])
    else:
        messages.extend([
            {"role": "user", "content": json.dumps({"observation": success_output})},
            {"role": "assistant", "content": json.dumps({
                "thought": "Execution completed successfully with expected outputs.",
                "tool": "finish",
                "args": {"summary": summary}
            })}
        ])
    return {"messages": messages}

CODE_TASKS = [
    ("Write a C++ program bsearch.cpp implementing binary search on [1, 3, 5, 7, 9, 11] looking for 7 and asserting the index.",
     "bsearch.cpp",
     "#include <iostream>\n#include <vector>\n#include <cassert>\nint bsearch(const std::vector<int>& v, int t){\n    int l=0, r=v.size()-1;\n    while(l<=r){\n        int m=l+(r-l)/2;\n        if(v[m]==t) return m;\n        if(v[m]<t) l=m+1; else r=m-1;\n    }\n    return -1;\n}\nint main(){\n    std::vector<int> a = {1,3,5,7,9,11};\n    assert(bsearch(a, 7) == 3);\n    std::cout << \"BSEARCH_OK\\n\";\n    return 0;\n}",
     None, None, "BSEARCH_OK", "Implemented verified binary search in C++."),

    ("Write a script lru.py implementing an LRU cache with capacity 2 and verify put/get operations.",
     "lru.py",
     "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, cap):\n        self.cap = cap\n        self.c = OrderedDict()\n    def get(self, k):\n        if k not in self.c: return -1\n        self.c.move_to_end(k)\n        return self.c[k]\n    def put(self, k, v):\n        if k in self.c: self.c.move_to_end(k)\n        self.c[k] = v\n        if len(self.c) > self.cap: self.c.popitem(last=False)\n\nl = LRUCache(2)\nl.put(1,1); l.put(2,2); assert l.get(1)==1; l.put(3,3); assert l.get(2)==-1\nprint('LRU_OK')",
     None, None, "LRU_OK", "Built and tested LRU Cache implementation in Python."),

    ("Write a script dijkstra.py finding the shortest path in a weighted graph from node A to C.",
     "dijkstra.py",
     "import heapq\ndef dijkstra(graph, start):\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n    pq = [(0, start)]\n    while pq:\n        cur_d, u = heapq.heappop(pq)\n        if cur_d > distances[u]: continue\n        for v, w in graph[u]:\n            if distances[u] + w < distances[v]:\n                distances[v] = distances[u] + w\n                heapq.heappush(pq, (distances[v], v))\n    return distances\n\ng = {'A': [('B', 1), ('C', 4)], 'B': [('C', 2)], 'C': []}\nres = dijkstra(g, 'A')\nassert res['C'] == 3\nprint('DIJKSTRA_OK')",
     None, None, "DIJKSTRA_OK", "Implemented Dijkstra shortest path algorithm."),

    ("Write a script quick_select.py that finds the kth smallest element in [7, 10, 4, 3, 20, 15] for k=3 and prints RESULT:7.",
     "quick_select.py",
     "def quickselect(arr, k):\n    if not arr: return None\n    pivot = arr[len(arr)//2]\n    lows = [x for x in arr if x < pivot]\n    highs = [x for x in arr if x > pivot]\n    pivots = [x for x in arr if x == pivot]\n    if k < len(lows): return quickselect(lows, k)\n    elif k < len(lows) + len(pivots): return pivot\n    else: return quickselect(highs, k - len(lows) - len(pivots))\n\nnums = [7, 10, 4, 3, 20, 15]\nval = quickselect(nums, 2)\nassert val == 7\nprint(f'RESULT:{val}')",
     None, None, "RESULT:7", "Implemented Quickselect algorithm in Python.")
]

all_dataset = []

# 1. Expand Code Trajectories (2,000 samples)
for _ in range(500):
    for task in CODE_TASKS:
        all_dataset.append(make_coding_traj(*task))

# 2. Add Math CoT from GSM8k (1,500 samples)
print("Downloading GSM8k...")
gsm = load_dataset("openai/gsm8k", "main", split="train[:1500]")
for ex in gsm:
    all_dataset.append({"messages": [
        {"role": "system", "content": "You are a precise reasoning assistant. Output clear step-by-step calculations followed by the final answer."},
        {"role": "user", "content": ex["question"]},
        {"role": "assistant", "content": ex["answer"]}
    ]})

# 3. Add Factual QA from SQuAD (1,000 samples)
print("Downloading SQuAD v2...")
squad = load_dataset("rajpurkar/squad_v2", split="train[:1000]")
for ex in squad:
    ans = ex["answers"]["text"][0] if len(ex["answers"]["text"]) > 0 else "The provided context does not contain sufficient facts to answer."
    all_dataset.append({"messages": [
        {"role": "system", "content": "Answer the question strictly and accurately based on the provided reference context."},
        {"role": "user", "content": f"Reference Context:\n{ex['context']}\n\nQuestion: {ex['question']}"},
        {"role": "assistant", "content": ans}
    ]})

# 4. Add Conversational & Instruction Data from SmolTalk (1,500 samples)
print("Downloading SmolTalk dialogues...")
try:
    smoltalk = load_dataset("HuggingFaceTB/smoltalk", "everyday-conversations", split="train[:1500]")
    for ex in smoltalk:
        all_dataset.append({"messages": ex["messages"]})
except Exception:
    print("Falling back to SlimOrca for conversational data...")
    orca = load_dataset("Open-Orca/SlimOrca-Dedup", split="train[:1500]")
    for ex in orca:
        msgs = [{"role": m["from"].replace("human", "user").replace("gpt", "assistant"), "content": m["value"]} for m in ex["conversations"]]
        all_dataset.append({"messages": msgs})

random.shuffle(all_dataset)
os.makedirs("data", exist_ok=True)
with open("data/master_train.jsonl", "w", encoding="utf-8") as f:
    for item in all_dataset:
        f.write(json.dumps(item) + "\n")

print(f"Generated {len(all_dataset)} samples in data/master_train.jsonl")