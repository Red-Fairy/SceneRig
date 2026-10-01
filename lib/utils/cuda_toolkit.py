"""Point CUDA extension builds at the toolkit installed next to a backend.

gsplat and nvdiffrast compile against ``CUDA_HOME``. A system toolkit whose
major version differs from the interpreter's torch build fails to compile or
load, so scripts/install.sh places a matching conda toolkit beside each backend
that needs one. Standard library only: imported from every backend venv.
"""

from __future__ import annotations

import os
import sys


def _prepend(var: str, path: str) -> None:
    current = os.environ.get(var)
    os.environ[var] = path if not current else path + os.pathsep + current


def use_cuda_toolkit(prefix: str) -> bool:
    """Use the toolkit at ``prefix`` if it has nvcc. Returns whether it was applied."""
    # Backends run without venv activation; torch looks up ninja on PATH.
    _prepend("PATH", os.path.dirname(sys.executable))
    prefix = os.path.abspath(prefix)
    if not os.path.isfile(os.path.join(prefix, "bin", "nvcc")):
        return False
    os.environ["CUDA_HOME"] = prefix
    _prepend("PATH", os.path.join(prefix, "bin"))
    # Conda puts CUDA headers and libraries under targets/, not include/ and lib64/.
    targets = os.path.join(prefix, "targets", "x86_64-linux")
    if os.path.isdir(targets):
        _prepend("CPATH", os.path.join(targets, "include"))
        _prepend("LIBRARY_PATH", os.path.join(targets, "lib"))
    _prepend("LIBRARY_PATH", os.path.join(prefix, "lib"))
    # nvcc rejects host compilers newer than it supports; prefer the conda one.
    for var, name in (("CC", "x86_64-conda-linux-gnu-gcc"), ("CXX", "x86_64-conda-linux-gnu-g++")):
        exe = os.path.join(prefix, "bin", name)
        if os.path.isfile(exe):
            os.environ[var] = exe
    return True
