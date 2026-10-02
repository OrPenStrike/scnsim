"""Prepare and verify the two source-bound, non-executing Pages renders."""

from __future__ import annotations

import argparse
from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import tomllib
from urllib.parse import unquote, urlsplit


SITE = "https://orpenstrike.github.io/scnsim/"
REPO = "https://github.com/OrPenStrike/scnsim"
SHA = re.compile(r"[0-9a-f]{40}\Z")
BRANCHES = ("main", "develop")


class References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value and name in ("href", "src"):
                self.paths.append(value)


def _regular_tree(root: Path) -> None:
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"Pages source/output is not a regular directory: {root}")
    for path in root.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError(f"Pages tree contains an unsafe entry: {path}")


def _version(source: Path) -> str:
    value = tomllib.loads((source / "pyproject.toml").read_text(encoding="utf-8"))
    version = value["project"]["version"]
    if not isinstance(version, str) or not version:
        raise ValueError("Pages source has no package version")
    return version


def _prepare(source: Path, branch: str, sha: str) -> None:
    if branch not in BRANCHES or not SHA.fullmatch(sha):
        raise ValueError("Pages branch/commit is invalid")
    _regular_tree(source)
    if (source / ".git").exists() or not (source / "_quarto.yml").is_file():
        raise ValueError("Pages source must be a clean exported Quarto project")
    if not (source / "_extensions/arfiligol/askr/_extension.yml").is_file():
        raise ValueError(f"{branch} is missing its own required Askr extension")
    if not ((source / "index.qmd").is_file() or (source / "README.md").is_file()):
        raise ValueError(f"{branch} has no source-backed home page")
    version = _version(source)
    profile = source / "_quarto-pages.yml"
    if profile.exists():
        raise ValueError("Pages render profile already exists in exported source")
    profile.write_text(
        "website:\n"
        f"  site-url: {json.dumps(SITE + branch + '/')}\n"
        f"  repo-branch: {json.dumps(sha)}\n"
        "  navbar:\n"
        "    right:\n"
        f"      - text: {json.dumps(branch + ' · ' + version)}\n"
        "        menu:\n"
        f"          - text: Version chooser\n            href: {json.dumps(SITE)}\n"
        f"          - text: main\n            href: {json.dumps(SITE + 'main/')}\n"
        f"          - text: develop\n            href: {json.dumps(SITE + 'develop/')}\n"
        "  page-footer:\n"
        f"    center: {json.dumps(branch + ' · ' + version + ' · [' + sha + '](' + REPO + '/tree/' + sha + ')')}\n",
        encoding="utf-8",
    )
    print(f"prepared {branch} {version} {sha}")


def _local_reference(site: Path, page: Path, value: str, branch: str) -> Path | None:
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or value.startswith("#"):
        return None
    path = unquote(parsed.path)
    if not path:
        return None
    prefix = f"/scnsim/{branch}/"
    if path.startswith(prefix):
        return site / path.removeprefix(prefix)
    if path.startswith("/"):
        raise ValueError(f"cross-site absolute reference in {branch}: {value}")
    return (page.parent / path).resolve()


def _inspect(source: Path, branch: str, sha: str) -> tuple[str, Path]:
    if branch not in BRANCHES or not SHA.fullmatch(sha):
        raise ValueError("Pages branch/commit is invalid")
    version = _version(source)
    site = source / "_site"
    _regular_tree(site)
    index = site / "index.html"
    search = site / "search.json"
    if not index.is_file() or not search.is_file():
        raise ValueError(f"{branch} render lacks home or branch-local search")
    json.loads(search.read_text(encoding="utf-8"))
    html = index.read_text(encoding="utf-8")
    for required in (branch, version, sha, SITE, SITE + "main/", SITE + "develop/", REPO):
        if required not in html:
            raise ValueError(f"{branch} home lacks branch/version/commit/navigation provenance: {required}")
    if f"/{sha}/" not in html:
        raise ValueError(f"{branch} home source link does not target its rendered commit")
    image = site / "docs/assets/readme-hero-orca-penguin.png"
    if (source / "docs/assets/readme-hero-orca-penguin.png").is_file() and not image.is_file():
        raise ValueError(f"{branch} render lacks its README hero image")
    notebook = site / "examples/engineer/chapter-01/chapter.ipynb"
    if (source / "examples/engineer/chapter-01/chapter.ipynb").is_file() and not notebook.is_file():
        raise ValueError(f"{branch} render lacks a linked Chapter notebook")
    for page in (index, site / "docs/index.html"):
        if not page.is_file():
            raise ValueError(f"{branch} render lacks a navigation entry page: {page}")
        refs = References()
        refs.feed(page.read_text(encoding="utf-8"))
        for value in refs.paths:
            target = _local_reference(site, page, value, branch)
            if target is not None and not target.is_file() and not target.is_dir():
                raise ValueError(f"{branch} render has a broken home/course asset: {value}")
    math_page = site / "docs/implementation/diagonal-root-numerical-procedure.html"
    if not math_page.is_file() or "mathjax" not in math_page.read_text(encoding="utf-8").lower():
        raise ValueError(f"{branch} render lacks declared MathJax support")
    return version, site


def _assemble(template: Path, main: Path, main_sha: str, develop: Path, develop_sha: str, output: Path) -> None:
    if output.exists():
        raise ValueError("Pages output must be a new isolated directory")
    versions = {}
    sites = {}
    for branch, source, sha in (("main", main, main_sha), ("develop", develop, develop_sha)):
        versions[branch], sites[branch] = _inspect(source, branch, sha)
    output.mkdir(parents=True)
    for branch in BRANCHES:
        shutil.copytree(sites[branch], output / branch, symlinks=False)
    html = template.read_text(encoding="utf-8")
    for token, value in {
        "MAIN_VERSION": versions["main"], "MAIN_SHA": main_sha,
        "DEVELOP_VERSION": versions["develop"], "DEVELOP_SHA": develop_sha,
    }.items():
        html = html.replace("{{" + token + "}}", escape(value))
    if "{{" in html:
        raise ValueError("Pages selector template has an unresolved token")
    (output / "index.html").write_text(html, encoding="utf-8")
    _regular_tree(output)
    print(f"assembled root + main/{versions['main']}@{main_sha} + develop/{versions['develop']}@{develop_sha}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--branch", choices=BRANCHES, required=True)
    prepare.add_argument("--sha", required=True)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("--template", type=Path, required=True)
    assemble.add_argument("--main", type=Path, required=True)
    assemble.add_argument("--main-sha", required=True)
    assemble.add_argument("--develop", type=Path, required=True)
    assemble.add_argument("--develop-sha", required=True)
    assemble.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        _prepare(args.source, args.branch, args.sha)
    else:
        _assemble(args.template, args.main, args.main_sha, args.develop, args.develop_sha, args.output)


if __name__ == "__main__":
    main()
