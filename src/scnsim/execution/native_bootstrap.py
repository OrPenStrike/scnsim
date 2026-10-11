"""Guard-owned exec barrier; never import SCNSim or inherit Kernel liveness.

The guardian registers OS descendant ownership before releasing this process.
The error descriptor closes on successful exec so native launch errors remain
launch errors rather than a manufactured native exit status.
"""
from __future__ import annotations
import json
import os
import sys


def _error_record(error):
    import importlib.util
    from pathlib import Path
    specification = importlib.util.spec_from_file_location(
        'scnsim_native_error_transport', Path(__file__).with_name('native_guard.py'))
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.error_record(error)


def main():
    configuration_fd, barrier_fd, error_fd = map(int, sys.argv[1:])
    os.set_inheritable(error_fd, False)
    try:
        with os.fdopen(configuration_fd, 'r', encoding='utf-8') as stream:
            configuration = json.load(stream)
        if os.read(barrier_fd, 1) != b'G':
            raise RuntimeError('native guardian did not release exec barrier')
        os.close(barrier_fd)
        os.execvpe(configuration['argv'][0], configuration['argv'], configuration['env'])
    except BaseException as error:
        record = _error_record(error)
        os.write(error_fd, json.dumps(record).encode('utf-8'))
        os.close(error_fd)
        raise


if __name__ == '__main__':
    main()
