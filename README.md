
# KACHOW: Autonomous Local AI Coding & Self-Healing Agent

> An ultra-lightweight, offline autonomous coding agent built for consumer hardware. Powered by a custom QLoRA fine-tuned 360M parameter model with local tool execution, compiler self-healing, and a unified desktop GUI.

---

## Overview

**KACHOW** is a self-contained, offline desktop AI coding assistant engineered to run entirely on low-VRAM edge GPUs. By fine-tuning `SmolLM2-360M-Instruct` on a multi-domain dataset using 4-bit quantization and response-only loss masking, KACHOW delivers fast, deterministic JSON tool dispatch, multi-step math reasoning, and local code compilation without relying on third-party cloud APIs.

---

## Key Features

- **Local Tool Execution:** Writes, reads, and executes C++ and Python code directly inside a local workspace sandbox.
- **Autonomous Compiler Self-Healing:** Catches `g++` build errors and Python stack traces, analyzes failure modes in its `thought` stream, and patches its own code until `exit_code: 0` is reached.
- **Ultra-Low Memory Footprint:** Capped at **~2.4 GB VRAM** peak usage—ideal for 4 GB mobile GPUs (NVIDIA RTX 3050 Laptop GPU tested).
- **Multi-Domain Intelligence:** Fine-tuned on a blended mixture of custom synthetic tool-use trajectories, **GSM8k** (Chain-of-Thought math), and **SQuAD 2.0** (grounded factual extraction).
- **Unified Architecture:** Runs an in-memory FastAPI backend paired with a native `PyWebview` desktop window for sub-millisecond local inter-process communication.

---

## Technical Specifications

| Parameter | Specification |
| :--- | :--- |
| **Base Model** | `HuggingFaceTB/SmolLM2-360M-Instruct` |
| **Fine-Tuning Method** | QLoRA (Rank $r=64$, $\alpha=128$, Dropout $0.05$) |
| **Quantization** | 4-bit NormalFloat (NF4) base precision |
| **Training Dataset** | 6,000 multi-domain samples $\times$ 3 epochs (18,000 passes / 2,250 steps) |
| **Optimizer** | Paged AdamW 8-bit (`learning_rate=2e-4` with Cosine Annealing) |
| **Loss Strategy** | Response-Only Loss Masking (`labels = -100` on system/user tokens) |
| **Final Loss Metrics** | Final Step Loss: `0.1315` \| Average Loss: `0.2087` \| Grad Norm: `0.1623` |
| **Inference Throughput** | 40–60+ tokens/second on RTX 3050 Laptop GPU |

---

## Architecture Diagram


```

┌─────────────────────────────────────────────────────────────┐
│                 PyWebview GUI Desktop Window                │
└──────────────────────────────┬──────────────────────────────┘
│ Local HTTP / Loopback (127.0.0.1)
▼
┌─────────────────────────────────────────────────────────────┐
│                 Unified FastAPI Backend                     │
│  • Single In-Memory PyTorch Instance                        │
│  • NF4 Base Weights + Local ./agent_adapter                 │
│  • Predicts structured JSON tool calls & thoughts          │
└──────────────────────────────┬──────────────────────────────┘
│ Subprocess Execution
▼
┌─────────────────────────────────────────────────────────────┐
│                 Workspace Subprocess Engine                 │
│  • Executes g++ / python3 scripts                           │
│  • Captures stdout, stderr, and exit codes                  │
└──────────────────────────────┬──────────────────────────────┘
│
[ If exit_code != 0 ] ──► (Autonomous Repair Loop)
│
▼
┌─────────────────────────────────────────────────────────────┐
│                  Verified Execution Output                  │
└─────────────────────────────────────────────────────────────┘

```

---

## Repository Structure


```

kachow-agent/
├── agent_adapter/          # Fine-tuned QLoRA weights & adapter config
├── data/
│   └── master_train.jsonl   # 6k multi-domain SFT dataset
├── ui/
│   ├── index.html          # PyWebview frontend layout
│   └── app.js              # UI state management & event handlers
├── main_app.py             # Desktop app entrypoint (Multiprocessing & GUI launcher)
├── server.py               # Unified FastAPI server & PyTorch model loader
├── run_agent.py            # CLI-based interactive agent testing tool
├── build_app.spec          # PyInstaller standalone build configuration
├── requirements.txt        # Python dependency manifest
└── README.md               # Project documentation

```

---

## Getting Started

### Prerequisites

- **OS:** Windows 10/11 (64-bit)
- **GPU:** NVIDIA GPU with $\ge 4\text{ GB}$ VRAM (CUDA 11.8+ / 12.1+ supported)
- **Compiler:** `g++` (MinGW-w64) for C++ code execution, Python 3.10+
- **CUDA Toolkit:** Drivers supporting PyTorch GPU acceleration

### Installation

1. **Clone the repository:**
   ```powershell
   git clone [https://github.com/your-username/kachow-agent.git](https://github.com/your-username/kachow-agent.git)
   cd kachow-agent

```

2. **Create and activate a virtual environment:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

```


3. **Install PyTorch with CUDA support:**
```powershell
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)

```


4. **Install remaining dependencies:**
```powershell
pip install -r requirements.txt

```



---

## Usage

### Running the Desktop GUI App

To launch the full desktop application (FastAPI backend + PyWebview UI):

```powershell
python main_app.py

```

### Running the CLI Agent Sandbox

To test the fine-tuned model directly inside the terminal:

```powershell
python run_agent.py

```

### Example Verification Prompt

Try prompting the agent with a deliberate compiler bug to verify self-healing:

> *"Write a C++ program in `calc.cpp` that calculates the factorial of 10. Deliberately omit `#include <iostream>`, compile it, catch the error, fix the include, and run it."*

---

## Packaging into Standalone `.exe`

To package KACHOW into an executable for Windows distribution:

1. **Install PyInstaller:**
```powershell
pip install pyinstaller

```


2. **Build using the specification file:**
```powershell
pyinstaller --noconfirm build_app.spec

```


3. Find the compiled output folder at `dist/KACHOW/` and launch **`KACHOW.exe`**.

---

## License & Attribution

* **Base Model:** `SmolLM2-360M-Instruct` is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).
* **LoRA Adapter Weights:** Proprietary / Included under MIT License.
* **Dependencies:** Built using PyTorch, Hugging Face Transformers, PEFT, bitsandbytes, FastAPI, and PyWebview.

```



```
