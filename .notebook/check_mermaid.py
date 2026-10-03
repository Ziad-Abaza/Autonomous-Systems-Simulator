import re
from pathlib import Path

for f in list(Path("docs").rglob("*.md")) + [Path("README.md")]:
    if "master_audit" in str(f):
        continue
    t = f.read_text(encoding="utf-8")
    for m in re.finditer(r"```mermaid(.*?)```", t, re.S):
        block = m.group(1).strip()
        for i, line in enumerate(block.splitlines()):
            s = line.strip()
            if not s or s.startswith(("flowchart", "sequenceDiagram", "stateDiagram", "participant", "autonumber", "%%")):
                continue
            if s.count("[") != s.count("]") or s.count("(") != s.count(")"):
                print(f"{f} line {i}: {s}")
print("done")
