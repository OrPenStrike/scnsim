"""Managed JuliaPkg discovery entrypoint; all third-party probes stay owned.

Protocol uses a dedicated descriptor; JuliaPkg stdout remains ordinary output.
This helper has no Workspace or result-publication authority.
"""
from __future__ import annotations
import json
import os
import sys
import traceback


def _error_record(error):
    import importlib.util
    from pathlib import Path
    specification = importlib.util.spec_from_file_location(
        'scnsim_native_error_transport', Path(__file__).with_name('native_guard.py'))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.error_record(error)


def main():
    version = sys.argv[1]
    reply_fd = int(sys.argv[2])
    # The reply descriptor must not reach JuliaPkg probe executables.
    os.set_inheritable(reply_fd, False)
    phase = 'import'
    try:
        from juliapkg.compat import Compat
        from juliapkg.find_julia import find_julia
        from juliapkg.state import STATE
        phase = 'discovery'
        executable, discovered = find_julia(
            compat=Compat.parse(f'={version}'), prefix=STATE['install'],
            install=False, upgrade=False)
        reply = {'result': [str(executable), str(discovered)]}
    except BaseException as error:
        reply = {'error': _error_record(error), 'phase': phase}
    with os.fdopen(reply_fd, 'w', encoding='utf-8') as stream:
        json.dump(reply, stream)


if __name__ == '__main__':
    main()
