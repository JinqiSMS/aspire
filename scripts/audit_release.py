"""Validate the tracked release surface before publication."""
from pathlib import Path
import ast
import json
import re
import subprocess
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=PROJECT_ROOT,
                            check=True, capture_output=True)
    names = [name.decode("utf-8") for name in result.stdout.split(b"\0") if name]
    if not names:
        raise SystemExit("No tracked files; stage the intended release before auditing")
    problems = []
    total = 0
    maximum = (0, "")
    text_extensions = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".html", ".csv", ".txt", ".ps1", ".svg"}
    for name in names:
        path = PROJECT_ROOT / name
        size = path.stat().st_size
        total += size
        maximum = max(maximum, (size, name))
        if size >= 100 * 1024 * 1024:
            problems.append(f"Oversized ordinary Git file: {name}")
        if path.suffix.lower() == ".html":
            problems.append(f"Unexpected HTML report: {name}")
        if name.split("/")[0] in {".venv", "cache", "tmp", "archive", "datasets", "results"}:
            problems.append(f"Runtime or private archive file tracked: {name}")
        if re.search(r"[\u4e00-\u9fff]", name):
            problems.append(f"Non-English filename: {name}")
        if path.suffix in text_extensions:
            content = path.read_text(encoding="utf-8-sig")
            if re.search(r"[\u4e00-\u9fff]", content):
                problems.append(f"Untranslated text: {name}")
            if re.search(r"C:[/\\]+Users[/\\]+13681", content, re.I):
                problems.append(f"Local user path in public file: {name}")
            if path.suffix == ".py":
                ast.parse(content, filename=name)
            if path.suffix == ".md":
                for target in re.findall(r"\]\(([^)]+)\)", content):
                    target = target.split("#", 1)[0].strip("<>")
                    if target and "://" not in target and not target.startswith("mailto:"):
                        resolved = (path.parent / target).resolve()
                        if not resolved.exists():
                            problems.append(f"Broken Markdown link in {name}: {target}")
    summary = {"tracked_files": len(names), "total_bytes": total,
               "largest_file": maximum[1], "largest_bytes": maximum[0],
               "problems": problems, "passed": not problems}
    print(json.dumps(summary, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
