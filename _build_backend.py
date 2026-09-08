"""Setuptools hooks bind public knowledge to each source or embedded-data build.

No SCNSim module is imported. A source checkout generates canonical resources;
an sdist verifies its complete embedded bundle before delegating to setuptools.
"""

from pathlib import Path
import runpy

from setuptools import build_meta as _setuptools


def _prepare():
    root = Path(__file__).parent
    generator = runpy.run_path(str(root / "scripts/generate_agent_knowledge.py"))
    generator["prepare_bundle"](root)


def get_requires_for_build_wheel(config_settings=None):
    _prepare()
    return _setuptools.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_sdist(config_settings=None):
    _prepare()
    return _setuptools.get_requires_for_build_sdist(config_settings)


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    _prepare()
    return _setuptools.prepare_metadata_for_build_wheel(
        metadata_directory, config_settings
    )


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare()
    return _setuptools.build_wheel(
        wheel_directory, config_settings, metadata_directory
    )


def build_sdist(sdist_directory, config_settings=None):
    _prepare()
    return _setuptools.build_sdist(sdist_directory, config_settings)


# Editable installs remain usable for development, but are not bundle transport.
def get_requires_for_build_editable(config_settings=None):
    _prepare()
    return _setuptools.get_requires_for_build_editable(config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    _prepare()
    return _setuptools.prepare_metadata_for_build_editable(
        metadata_directory, config_settings
    )


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    _prepare()
    return _setuptools.build_editable(
        wheel_directory, config_settings, metadata_directory
    )
