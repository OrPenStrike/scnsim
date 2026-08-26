"""Validate SCNSim package metadata against the selected repository line."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

VERSION_PATTERN = re.compile(
    r"^(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)\.(?:0|[1-9]\d*)"
    r"(?:(?P<dev>\.dev\d+)|(?P<rc>rc\d+))?$"
)


def _quoted_value(section: str, key: str) -> str:
    """Read one required quoted scalar from a known generated TOML section."""

    match = re.search(rf'(?m)^{re.escape(key)}\s*=\s*"([^"]+)"\s*$', section)
    if match is None:
        raise ValueError(f"missing quoted {key!r} value")
    return match.group(1)


def _project_section(text: str) -> str:
    """Return the PEP 621 project section without requiring Python 3.11."""

    marker = "[project]"
    try:
        section = text.split(marker, 1)[1]
    except IndexError as error:
        raise ValueError("pyproject.toml has no [project] section") from error
    return section.split("\n[", 1)[0]


def version_kind(version: str) -> str:
    """Return the SCNSim release class encoded by a PEP 440 version."""

    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"unsupported SCNSim version: {version!r}")
    if match.group("dev"):
        return "development"
    if match.group("rc"):
        return "release-candidate"
    return "stable"


def validate_line(version: str, line: str) -> None:
    """Require prereleases on develop and stable releases on main."""

    kind = version_kind(version)
    if line == "develop" and kind == "stable":
        raise ValueError("develop must use a development or release-candidate version")
    if line == "main" and kind != "stable":
        raise ValueError("main must use a stable version")


def repository_version(root: Path) -> str:
    """Return one version shared by pyproject.toml and uv.lock."""

    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    version = _quoted_value(_project_section(project), "version")

    lock = (root / "uv.lock").read_text(encoding="utf-8")
    locked = []
    for package in lock.split("[[package]]")[1:]:
        if _quoted_value(package, "name") != "scnsim":
            continue
        if re.search(
            r'(?m)^source\s*=\s*\{\s*editable\s*=\s*"\."\s*\}\s*$',
            package,
        ):
            locked.append(_quoted_value(package, "version"))
    if locked != [version]:
        raise ValueError(f"scnsim lock version {locked!r} does not match {version!r}")
    return version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--line", choices=("any", "develop", "main"), default="any")
    args = parser.parse_args()
    version = repository_version(Path(__file__).resolve().parents[1])
    if args.line == "any":
        version_kind(version)
    else:
        validate_line(version, args.line)
    print(f"scnsim {version} ({args.line})")


if __name__ == "__main__":
    main()
