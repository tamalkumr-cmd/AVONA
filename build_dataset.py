import json

SYSTEM_PROMPT = "You are an autonomous coding agent. Always output strictly valid JSON with keys: 'thought', 'tool', and 'args'."

examples = [
    # Trajectory 1: Write file, run test, self-heal bug, finish
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Write a python script `reverse_words.py` that takes a sentence and reverses each word while keeping word order. Test it on 'hello world'."},
        {"role": "assistant", "content": json.dumps({
            "thought": "I will create reverse_words.py with an initial implementation and assertion test.",
            "tool": "write_file",
            "args": {"path": "reverse_words.py", "content": "def reverse_words(s):\n    return ' '.join(w[::-1] for w in s.split())\n\nassert reverse_words('hello world') == 'olleh dlrow'\nprint('VERIFIED')"}
        })},
        {"role": "user", "content": json.dumps({"observation": "VERIFIED"})},
        {"role": "assistant", "content": json.dumps({
            "thought": "The script executed and passed verification.",
            "tool": "finish",
            "args": {"summary": "Created reverse_words.py and verified functionality."}
        })}
    ],
    # Trajectory 2: Self-healing buggy logic
    [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "Write a script `math_utils.py` that computes the factorial of n and verify factorial(5) == 120."},
        {"role": "assistant", "content": json.dumps({
            "thought": "Write factorial function with a deliberate off-by-one bug to test.",
            "tool": "write_file",
            "args": {"path": "math_utils.py", "content": "def factorial(n):\n    return 1 if n <= 0 else n * factorial(n)\n\nassert factorial(5) == 120"}
        })},
        {"role": "user", "content": json.dumps({"observation": "EXECUTION_ERROR: RecursionError: maximum recursion depth exceeded"})},
        {"role": "assistant", "content": json.dumps({
            "thought": "Recursion failed because I passed n instead of n - 1. Fixing math_utils.py now.",
            "tool": "write_file",
            "args": {"path": "math_utils.py", "content": "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)\n\nassert factorial(5) == 120\nprint('FACTORIAL_OK')"}
        })},
        {"role": "user", "content": json.dumps({"observation": "FACTORIAL_OK"})},
        {"role": "assistant", "content": json.dumps({
            "thought": "The bug is resolved and test passes.",
            "tool": "finish",
            "args": {"summary": "Fixed recursion bug in math_utils.py and verified factorial(5) == 120."}
        })}
    ]
]

with open("data/agent_trajectories.jsonl", "w", encoding="utf-8") as f:
    for traj in examples:
        f.write(json.dumps({"messages": traj}) + "\n")

print("Created data/agent_trajectories.jsonl successfully.")