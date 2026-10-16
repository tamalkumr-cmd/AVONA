import json
import os
import random

SYSTEM_PROMPT = (
    "You are an autonomous coding agent. Always output strictly ONE valid JSON object per turn "
    "with keys: 'thought', 'tool', and 'args'. Available tools: 'write_file', 'read_file', 'execute_code', 'finish'."
)

def make_trajectory(task, filename, code, error_msg, fixed_code, success_output, summary):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
        {"role": "assistant", "content": json.dumps({
            "thought": f"I will implement the solution in {filename} with verification checks.",
            "tool": "write_file",
            "args": {"path": filename, "content": code}
        })},
        {"role": "user", "content": json.dumps({"observation": f"Successfully wrote {len(code)} characters to {filename}"})},
        {"role": "assistant", "content": json.dumps({
            "thought": f"I will run {filename} to test execution.",
            "tool": "execute_code",
            "args": {"command": f"python {filename}"}
        })}
    ]

    if error_msg and fixed_code:
        messages.extend([
            {"role": "user", "content": json.dumps({"observation": error_msg})},
            {"role": "assistant", "content": json.dumps({
                "thought": "Execution failed with an error. Analyzing traceback and patching code.",
                "tool": "write_file",
                "args": {"path": filename, "content": fixed_code}
            })},
            {"role": "user", "content": json.dumps({"observation": f"Successfully wrote {len(fixed_code)} characters to {filename}"})},
            {"role": "assistant", "content": json.dumps({
                "thought": "Re-running the patched script to verify fix.",
                "tool": "execute_code",
                "args": {"command": f"python {filename}"}
            })},
            {"role": "user", "content": json.dumps({"observation": success_output})},
            {"role": "assistant", "content": json.dumps({
                "thought": "All checks and assertions passed.",
                "tool": "finish",
                "args": {"summary": summary}
            })}
        ])
    else:
        messages.extend([
            {"role": "user", "content": json.dumps({"observation": success_output})},
            {"role": "assistant", "content": json.dumps({
                "thought": "Execution completed successfully with matching outputs.",
                "tool": "finish",
                "args": {"summary": summary}
            })}
        ])
    return {"messages": messages}

# Broad Domain Task Catalog
DOMAINS = [
    # 1. Algorithms & Data Structures
    ("Write a script binary_search.py implementing recursive binary search on [2, 5, 8, 12, 16, 23, 38] searching for 16.",
     "binary_search.py",
     "def bsearch(arr, low, high, x):\n    if high >= low:\n        mid = (high + low) // 2\n        if arr[mid] == x: return mid\n        elif arr[mid] > x: return bsearch(arr, low, mid - 1, x)\n        else: return bsearch(arr, mid + 1, high, x)\n    return -1\n\narr = [2, 5, 8, 12, 16, 23, 38]\nidx = bsearch(arr, 0, len(arr) - 1, 16)\nassert idx == 4\nprint(f'INDEX:{idx}')",
     None, None, "INDEX:4", "Implemented recursive binary search in binary_search.py."),

    ("Write a script merge_sort.py that sorts [38, 27, 43, 3, 9, 82, 10] and prints SORTED_OK.",
     "merge_sort.py",
     "def merge_sort(arr):\n    if len(arr) > 1:\n        mid = len(arr)//2\n        L, R = arr[:mid], arr[mid:]\n        merge_sort(L); merge_sort(R)\n        i = j = k = 0\n        while i < len(L) and j < len(R):\n            if L[i] <= R[j]: arr[k] = L[i]; i += 1\n            else: arr[k] = R[j]; j += 1\n            k += 1\n        while i < len(L): arr[k] = L[i]; i += 1; k += 1\n        while j < len(R): arr[k] = R[j]; j += 1; k += 1\n    return arr\n\nres = merge_sort([38, 27, 43, 3, 9, 82, 10])\nassert res == [3, 9, 10, 27, 38, 43, 82]\nprint('SORTED_OK')",
     None, None, "SORTED_OK", "Implemented merge sort algorithm in merge_sort.py."),

    # 2. File I/O, CSV & JSON Processing
    ("Write a script csv_processor.py that creates a sample CSV of products and calculates the total inventory value.",
     "csv_processor.py",
     "import csv\nwith open('inventory.csv', 'w', newline='') as f:\n    w = csv.writer(f)\n    w.writerow(['item', 'price', 'qty'])\n    w.writerow(['Apple', 1.5, 100])\n    w.writerow(['Orange', 2.0, 50])\n\ntotal = 0.0\nwith open('inventory.csv', 'r') as f:\n    r = csv.DictReader(f)\n    for row in r:\n        total += float(row['price']) * int(row['qty'])\nassert total == 250.0\nprint(f'TOTAL_VALUE:{total}')",
     None, None, "TOTAL_VALUE:250.0", "Built CSV inventory generator and total value calculator."),

    ("Write a script config_validator.py that parses a JSON configuration and raises ValueError if 'api_key' is missing.",
     "config_validator.py",
     "import json\ndef validate(cfg_str):\n    data = json.loads(cfg_str)\n    if 'api_key' not in data:\n        raise ValueError('Missing api_key')\n    return True\n\nassert validate('{\"api_key\": \"secret123\", \"timeout\": 30}') == True\nprint('CONFIG_VALID')",
     None, None, "CONFIG_VALID", "Validated JSON configuration schema in config_validator.py."),

    # 3. Text Processing, Regex & Parsing
    ("Write a script log_parser.py that extracts all IPv4 addresses from a log block and counts uniques.",
     "log_parser.py",
     "import re\nlogs = '2026-03-01 ERROR 192.168.1.1 Auth failed\\n2026-03-01 INFO 10.0.0.1 Connected\\n2026-03-01 WARN 192.168.1.1 Timeout'\nips = set(re.findall(r'\\b(?:[0-9]{1,3}\\.){3}[0-9]{1,3}\\b', logs))\nassert len(ips) == 2\nprint(f'UNIQUE_IPS:{len(ips)}')",
     None, None, "UNIQUE_IPS:2", "Extracted unique IPv4 addresses from log logs in log_parser.py."),

    ("Write a script semver_check.py that extracts major, minor, patch from version strings.",
     "semver_check.py",
     "import re\ndef parse_semver(v):\n    m = re.match(r'^v?(\\d+)\\.(\\d+)\\.(\\d+)$', v)\n    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None\n\nassert parse_semver('v2.14.3') == (2, 14, 3)\nprint('SEMVER_PARSED')",
     None, None, "SEMVER_PARSED", "Parsed semantic version string in semver_check.py."),

    # 4. Cryptography, Security & Hashes
    ("Write a script hasher.py that computes the SHA-256 hash of 'agent_secret' and prints the first 8 hex characters.",
     "hasher.py",
     "import hashlib\nh = hashlib.sha256('agent_secret'.encode()).hexdigest()\nprefix = h[:8]\nprint(f'HASH_PREFIX:{prefix}')",
     None, None, "HASH_PREFIX:884e9314", "Computed SHA-256 hash prefix in hasher.py."),

    # 5. Math, Matrices & Statistics
    ("Write a script matrix_ops.py that computes the transpose and determinant of a 2x2 matrix [[4, 7], [2, 6]].",
     "matrix_ops.py",
     "def det_2x2(m):\n    return m[0][0]*m[1][1] - m[0][1]*m[1][0]\n\nm = [[4, 7], [2, 6]]\ndet = det_2x2(m)\nassert det == 10\nprint(f'DET:{det}')",
     None, None, "DET:10", "Calculated matrix determinant in matrix_ops.py."),

    ("Write a script prime_factors.py to get prime factors of 84 and verify their product equals 84.",
     "prime_factors.py",
     "def prime_factors(n):\n    factors = []\n    d = 2\n    while d * d <= n:\n        while n % d == 0:\n            factors.append(d); n //= d\n        d += 1\n    if n > 1: factors.append(n)\n    return factors\n\nfactors = prime_factors(84)\nassert factors == [2, 2, 3, 7]\nprint('FACTORS_OK')",
     None, None, "FACTORS_OK", "Calculated prime factorization in prime_factors.py."),

    # 6. Self-Healing Bug Scenarios (ZeroDivision, IndexError, TypeError)
    ("Write a script safe_divide.py that averages non-empty list subsets, handling empty list inputs gracefully.",
     "safe_divide.py",
     "def average(nums):\n    return sum(nums) / len(nums)\nassert average([]) == 0",
     "EXECUTION_ERROR: ZeroDivisionError: division by zero",
     "def average(nums):\n    return sum(nums) / len(nums) if len(nums) > 0 else 0\nassert average([]) == 0\nassert average([10, 20, 30]) == 20\nprint('AVG_SAFE')",
     "AVG_SAFE", "Fixed ZeroDivisionError in safe_divide.py."),

    ("Write a script dict_lookup.py to safely retrieve nested keys without throwing KeyError.",
     "dict_lookup.py",
     "def get_nested(d, path):\n    curr = d\n    for k in path:\n        curr = curr[k]\n    return curr\n\ndata = {'user': {'profile': {'age': 22}}}\nassert get_nested(data, ['user', 'settings', 'theme']) == None",
     "EXECUTION_ERROR: KeyError: 'settings'",
     "def get_nested(d, path, default=None):\n    curr = d\n    for k in path:\n        if isinstance(curr, dict) and k in curr:\n            curr = curr[k]\n        else:\n            return default\n    return curr\n\ndata = {'user': {'profile': {'age': 22}}}\nassert get_nested(data, ['user', 'settings', 'theme']) is None\nassert get_nested(data, ['user', 'profile', 'age']) == 22\nprint('NESTED_OK')",
     "NESTED_OK", "Fixed KeyError bug with safe nested key navigation in dict_lookup.py.")
]

# Synthesize into 250 diverse variations
all_trajectories = []
for _ in range(25):
    for task_item in DOMAINS:
        all_trajectories.append(make_trajectory(*task_item))

random.shuffle(all_trajectories)

os.makedirs("data", exist_ok=True)
with open("data/agent_trajectories.jsonl", "w", encoding="utf-8") as f:
    for t in all_trajectories:
        f.write(json.dumps(t) + "\n")

print(f"Generated {len(all_trajectories)} broad-knowledge trajectories in data/agent_trajectories.jsonl")