"""Fail-closed classifier for the narrow Markdown prose fast lane."""

from __future__ import annotations

import difflib
import os
import re
import subprocess
import sys


ALLOWED_ACTIONS = {"opened", "synchronize", "reopened", "ready_for_review"}
SAFE_PATHS = re.compile(r"(?:docs/.+\.md|README\.md)\Z")
FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
HTML_OR_DIRECTIVE = re.compile(
    r"<!--|-->|<[^>]*>|\{\{|\}\}|\{%|%\}|<%|%>|\?\?|\?\{|\}"
)
IMAGE_EMBED = re.compile(r"!\[[^]]*\]\(")
INLINE_EXECUTION = re.compile(r"\{\s*(?:python|r|julia|ojs)(?:\s|[,}])", re.IGNORECASE)
INDENTED_CODE = re.compile(r"^(?: {4,}|\t)")
DIV_DIRECTIVE = re.compile(r":::")
YAML_DELIMITER = re.compile(r"^(?:\ufeff)?(?:---|\.\.\.)\s*$")


def git(*args: str, check: bool = True) -> bytes:
    return subprocess.run(
        ["git", *args], check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).stdout


def changed_paths(base: str, head: str) -> list[str] | None:
    try:
        raw = git("diff", "--name-only", "-z", base, head)
    except (subprocess.CalledProcessError, OSError):
        return None
    try:
        return [item.decode("utf-8") for item in raw.split(b"\0") if item]
    except UnicodeDecodeError:
        return None


def file_at(revision: str, path: str) -> str | None:
    try:
        entry = git("ls-tree", "-z", revision, "--", path)
        if not entry:
            return ""
        result = subprocess.run(
            ["git", "show", f"{revision}:{path}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return result.stdout.decode("utf-8")
    except (subprocess.CalledProcessError, UnicodeDecodeError, OSError):
        return None
    return None


def regular_markdown(revision: str, path: str) -> bool:
    try:
        entry = git("ls-tree", "-z", revision, "--", path)
    except (subprocess.CalledProcessError, OSError):
        return False
    if not entry:
        return True
    metadata = entry.split(b"\t", 1)[0].split()
    return len(metadata) == 3 and metadata[0] == b"100644" and metadata[1] == b"blob"


def prose_line_indices(text: str) -> set[int]:
    lines = text.splitlines()
    prose: set[int] = set()
    first_line = lines[0].lstrip("\ufeff").strip() if lines else ""
    in_frontmatter = first_line == "---"
    fence_char: str | None = None
    fence_length = 0
    for index, line in enumerate(lines):
        if in_frontmatter:
            if index > 0 and YAML_DELIMITER.fullmatch(line.strip()):
                in_frontmatter = False
            continue
        marker = FENCE.match(line)
        if fence_char is not None:
            if (
                marker
                and marker.group(1)[0] == fence_char
                and len(marker.group(1)) >= fence_length
                and not marker.group(2).strip()
            ):
                fence_char = None
            continue
        if marker:
            fence_char = marker.group(1)[0]
            fence_length = len(marker.group(1))
            continue
        if (
            line.startswith("\ufeff")
            or YAML_DELIMITER.fullmatch(line.strip())
            or INDENTED_CODE.match(line)
            or DIV_DIRECTIVE.search(line)
            or HTML_OR_DIRECTIVE.search(line)
            or IMAGE_EMBED.search(line)
            or INLINE_EXECUTION.search(line)
        ):
            continue
        prose.add(index)
    return prose


def is_prose_only_change(before: str, after: str) -> bool:
    protected_markup = (HTML_OR_DIRECTIVE, DIV_DIRECTIVE, INLINE_EXECUTION)
    if any(pattern.search(text) for text in (before, after) for pattern in protected_markup):
        return False
    old = before.splitlines()
    new = after.splitlines()
    old_prose = prose_line_indices(before)
    new_prose = prose_line_indices(after)
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
        a=old, b=new, autojunk=False
    ).get_opcodes():
        if tag == "equal":
            continue
        if any(index not in old_prose for index in range(i1, i2)):
            return False
        if any(index not in new_prose for index in range(j1, j2)):
            return False
    return True


def classify() -> str:
    event = os.environ.get("EVENT_NAME", "")
    action = os.environ.get("EVENT_ACTION", "")
    base = os.environ.get("BASE_SHA", "")
    head = os.environ.get("HEAD_SHA", "")
    if event == "pull_request":
        if action not in ALLOWED_ACTIONS:
            return "product"
    elif event != "push":
        return "product"
    if not re.fullmatch(r"[0-9a-f]{40}", base) or not re.fullmatch(
        r"[0-9a-f]{40}", head
    ):
        return "product"
    paths = changed_paths(base, head)
    if not paths or any(not SAFE_PATHS.fullmatch(path) for path in paths):
        return "product"
    for path in paths:
        if not regular_markdown(base, path) or not regular_markdown(head, path):
            return "product"
        before = file_at(base, path)
        after = file_at(head, path)
        if before is None or after is None or not is_prose_only_change(before, after):
            return "product"
    return "docs"


def main() -> int:
    lane = classify()
    output = os.environ.get("GITHUB_OUTPUT")
    if not output:
        print(f"lane={lane}")
        return 0
    with open(output, "a", encoding="utf-8") as stream:
        stream.write(f"lane={lane}\n")
    print(f"classified change scope: {lane}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
