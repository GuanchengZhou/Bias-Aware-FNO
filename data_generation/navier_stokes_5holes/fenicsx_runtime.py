"""Runtime helpers for local FEniCSx scripts."""

from __future__ import annotations

import os
import subprocess
import sys


_BUILD_ENV_KEYS = [
    "CC",
    "CXX",
    "LDSHARED",
    "LD",
    "AR",
    "NM",
    "RANLIB",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "ARCHFLAGS",
    "SDKROOT",
    "MACOSX_DEPLOYMENT_TARGET",
    "CMAKE_OSX_ARCHITECTURES",
    "LTO_LIBRARY",
    "LTOFLAGS",
]


def _select_macos_fabric_iface() -> str:
    try:
        proc = subprocess.run(["ifconfig", "-a"], check=False, capture_output=True, text=True)
    except Exception:
        return "lo0"

    blocks: list[tuple[str, list[str]]] = []
    current_name = None
    current_lines: list[str] = []
    for raw_line in proc.stdout.splitlines():
        if raw_line and not raw_line[:1].isspace():
            if current_name is not None:
                blocks.append((current_name, current_lines))
            current_name = raw_line.split(":", 1)[0]
            current_lines = [raw_line]
        elif current_name is not None:
            current_lines.append(raw_line)
    if current_name is not None:
        blocks.append((current_name, current_lines))

    preferred: list[str] = []
    fallback: list[str] = []
    ignored_prefixes = ("utun", "awdl", "llw", "bridge", "gif", "stf", "anpi", "ap")
    for name, lines in blocks:
        if name.startswith(ignored_prefixes):
            continue
        text = "\n".join(lines)
        if "status: active" not in text:
            continue
        if "inet " not in text and "inet6 " not in text:
            continue
        if name.startswith("en"):
            preferred.append(name)
        elif name == "lo0":
            fallback.append(name)

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return "lo0"


def _apply_macos_mpi_env(env: dict[str, str]) -> list[str]:
    notes: list[str] = []
    if sys.platform != "darwin":
        return notes

    if "FI_PROVIDER" not in env:
        env["FI_PROVIDER"] = "tcp"
        notes.append("FI_PROVIDER=tcp")

    if "FI_TCP_IFACE" not in env:
        iface = _select_macos_fabric_iface()
        env["FI_TCP_IFACE"] = iface
        notes.append(f"FI_TCP_IFACE={iface}")

    return notes


def sanitize_current_process_build_env(verbose: bool = True) -> None:
    removed = []
    for key in _BUILD_ENV_KEYS:
        if key in os.environ:
            removed.append(key)
            os.environ.pop(key, None)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    mpi_notes = _apply_macos_mpi_env(os.environ)

    if sys.platform == "darwin" and os.path.exists("/usr/bin/clang"):
        os.environ["CC"] = "/usr/bin/clang"
        os.environ["CXX"] = "/usr/bin/clang++"
        os.environ["LDSHARED"] = "/usr/bin/clang -bundle -undefined dynamic_lookup"
        os.environ["BLDSHARED"] = "/usr/bin/clang -bundle -undefined dynamic_lookup"
        old_path = os.environ.get("PATH", "")
        if not old_path.startswith("/usr/bin:"):
            os.environ["PATH"] = f"/usr/bin:{old_path}"

    if verbose and removed:
        print(f"[env] removed build vars: {', '.join(removed)}")
    if verbose and mpi_notes:
        print(f"[env] set MPI vars: {', '.join(mpi_notes)}")
