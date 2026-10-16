import subprocess
from pathlib import Path

WORKSPACE = Path("./agent_workspace")
WORKSPACE.mkdir(exist_ok=True)

def write_file(path: str, content: str) -> str:
    target = WORKSPACE / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"Successfully wrote {len(content)} characters to {path}"

def read_file(path: str) -> str:
    target = WORKSPACE / path
    if not target.exists():
        return f"Error: File '{path}' does not exist."
    return target.read_text(encoding="utf-8")

def execute_code(command: str) -> str:
    try:
        res = subprocess.run(
            command,
            cwd=WORKSPACE,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        out = res.stdout.strip()
        err = res.stderr.strip()
        if res.returncode == 0:
            return out if out else "Command executed successfully with no output."
        return f"EXECUTION_ERROR (Exit Code {res.returncode}):\n{err if err else out}"
    except subprocess.TimeoutExpired:
        return "EXECUTION_ERROR: Process timed out after 10 seconds."

def execute_tool(tool_name: str, args: dict) -> str:
    if tool_name == "write_file":
        return write_file(args.get("path", ""), args.get("content", ""))
    elif tool_name == "read_file":
        return read_file(args.get("path", ""))
    elif tool_name == "execute_code":
        return execute_code(args.get("command", ""))
    elif tool_name == "finish":
        return "TASK_COMPLETE"
    else:
        return f"Error: Unknown tool '{tool_name}'."