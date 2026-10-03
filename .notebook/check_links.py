import re
from pathlib import Path

files = ["README.md"]
for sub in Path("docs").rglob("*.md"):
    s = str(sub).replace("\\", "/")
    if "master_audit" in s:
        continue
    if sub.parent == Path("docs") and sub.name not in (
        "README.md", "DOCUMENTATION_AUDIT.md", "DOCUMENTATION_IMPLEMENTATION_REPORT.md"
    ):
        continue  # skip flat legacy docs
    files.append(s)

link_re = re.compile(r"\]\(([^)\s]+)\)|src=\"([^\"]+)\"")
bad_links, bad_imgs, anchors_bad, ok = [], [], [], 0

def anchors_of(path: Path):
    heads = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", line)
        if m:
            a = m.group(1).strip().lower()
            a = re.sub(r"[`*]", "", a)
            a = re.sub(r"[^a-z0-9_ \-]", "", a).replace(" ", "-")
            heads.append(a)
    return heads

anchor_cache = {}
for f in files:
    p = Path(f)
    if not p.exists():
        continue
    text = p.read_text(encoding="utf-8")
    for m in link_re.finditer(text):
        t = (m.group(1) or m.group(2)).strip()
        if t.startswith(("http", "mailto")):
            continue
        frag = ""
        if "#" in t:
            t, frag = t.split("#", 1)
        if not t:
            target = p
        else:
            target = p.parent / t
        if not target.exists():
            (bad_imgs if t.lower().endswith((".png", ".jpg", ".svg")) else bad_links).append(f"{f} -> {t}")
            continue
        ok += 1
        if frag:
            key = str(target.resolve())
            if key not in anchor_cache:
                anchor_cache[key] = anchors_of(target)
            if frag.lower() not in anchor_cache[key]:
                anchors_bad.append(f"{f} -> {t}#{frag}")

print(f"checked {len(files)} files, {ok} resolvable links/images")
print("BROKEN LINKS:" if bad_links else "no broken links")
for b in bad_links: print("  ", b)
print("BROKEN IMAGES:" if bad_imgs else "no broken images")
for b in bad_imgs: print("  ", b)
print("BAD ANCHORS:" if anchors_bad else "no bad anchors")
for b in anchors_bad: print("  ", b)
