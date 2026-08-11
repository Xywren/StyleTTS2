"""StyleTTS2 voice module installer.

Creates a Python venv, installs the correct PyTorch build for the current
platform/GPU, installs all remaining dependencies, and ensures espeak-ng is
available. Idempotent — safe to run repeatedly; skips work already done.

Usage:
    python setup.py install                      # full install
    python setup.py install --module-dir <path>   # override install location
"""

import argparse
import hashlib
import os
import platform
import shutil
import subprocess
import sys

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEPS_VERSION = 1


def _run(cmd, cwd=None, label=None):
    label = label or cmd[0]
    print(f"[setup] {label}: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd or MODULE_DIR,
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr + "\n")
        raise RuntimeError(f"{label} failed (exit {result.returncode})")
    return result.stdout


def _has_nvidia_gpu():
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _torch_install_args():
    if sys.platform == "darwin":
        return ["install", "-q", "torch", "torchaudio"]
    if _has_nvidia_gpu():
        return ["install", "-q", "torch", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu124"]
    return ["install", "-q", "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu"]


def _deps_stamp(torch_args):
    req_path = os.path.join(MODULE_DIR, "requirements.txt")
    if os.path.exists(req_path):
        req_hash = hashlib.sha256(open(req_path, "rb").read()).hexdigest()[:16]
    else:
        req_hash = "no-requirements"
    return f"v{DEPS_VERSION}|{' '.join(torch_args)}|{req_hash}"


def _find_espeak():
    if sys.platform == "darwin":
        candidates = [
            os.path.expanduser("~/ARI/tools/espeak-ng/lib/libespeak-ng.dylib"),
            "/opt/homebrew/lib/libespeak-ng.dylib",
            "/usr/local/lib/libespeak-ng.dylib",
        ]
    elif sys.platform == "win32":
        candidates = [
            os.path.join(os.environ.get("ProgramFiles", "C:\\Program Files"),
                         "eSpeak NG", "libespeak-ng.dll"),
            os.path.expanduser("~\\ARI\\tools\\espeak-ng\\libespeak-ng.dll"),
        ]
    else:
        candidates = [
            os.path.expanduser("~/ARI/tools/espeak-ng/lib/libespeak-ng.so.1"),
            "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",
            "/usr/lib/aarch64-linux-gnu/libespeak-ng.so.1",
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _ensure_espeak():
    if _find_espeak():
        print("[setup] espeak-ng found.", flush=True)
        return

    if sys.platform == "darwin":
        if shutil.which("brew"):
            print("[setup] Installing espeak-ng via Homebrew...", flush=True)
            _run(["brew", "install", "espeak-ng"], label="brew")
            return
    elif sys.platform != "win32":
        if shutil.which("apt-get"):
            print("[setup] Installing espeak-ng via apt...", flush=True)
            _run(["sudo", "apt-get", "install", "-y", "espeak-ng"], label="apt")
            return

    print("[setup] WARNING: espeak-ng not found. Install it manually:", flush=True)
    print("  macOS:   brew install espeak-ng", flush=True)
    print("  Linux:   sudo apt install espeak-ng", flush=True)
    print("  Windows: https://github.com/espeak-ng/espeak-ng/releases", flush=True)


def _msvc_check():
    if sys.platform != "win32":
        return
    try:
        subprocess.run(["cl"], capture_output=True, timeout=5)
    except FileNotFoundError:
        print("[setup] WARNING: MSVC build tools not found.", flush=True)
        print("  monotonic_align requires a C++ compiler.", flush=True)
        print("  Install 'Desktop development with C++' from:", flush=True)
        print("  https://visualstudio.microsoft.com/visual-cpp-build-tools/", flush=True)
        raise RuntimeError("MSVC build tools required on Windows")


def install(module_dir=None):
    install_dir = module_dir or os.path.join(os.path.expanduser("~"), "ARI",
                                              "voice-modules", "StyleTTS2")
    venv_dir = os.path.join(install_dir, "venv")
    marker_path = os.path.join(install_dir, ".deps-installed")

    os.makedirs(install_dir, exist_ok=True)

    is_win = sys.platform == "win32"
    pip = os.path.join(venv_dir, "Scripts\\pip.exe" if is_win else "bin/pip")
    venv_py = os.path.join(venv_dir, "Scripts\\python.exe" if is_win else "bin/python3")

    torch_args = _torch_install_args()
    stamp = _deps_stamp(torch_args)

    if os.path.exists(marker_path) and open(marker_path).read().strip() == stamp:
        print("[setup] StyleTTS2 already installed.", flush=True)
        return venv_py

    _msvc_check()

    if not os.path.isdir(venv_dir):
        print("[setup] Creating virtual environment...", flush=True)
        _run([sys.executable, "-m", "venv", venv_dir], label="venv")

    print("[setup] Upgrading pip...", flush=True)
    _run([venv_py, "-m", "pip", "install", "-q", "--upgrade", "pip"], label="pip")

    print("[setup] Installing PyTorch...", flush=True)
    _run([pip] + torch_args, label="torch")

    print("[setup] Installing requirements...", flush=True)
    req_path = os.path.join(MODULE_DIR, "requirements.txt")
    _run([pip, "install", "-q", "--prefer-binary", "-r", req_path],
         cwd=MODULE_DIR, label="requirements")

    print("[setup] Installing additional dependencies...", flush=True)
    _run([pip, "install", "-q", "flask", "phonemizer"], label="extras")

    _ensure_espeak()

    with open(marker_path, "w") as f:
        f.write(stamp)

    print("[setup] StyleTTS2 ready.", flush=True)
    return venv_py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")

    install_cmd = sub.add_parser("install")
    install_cmd.add_argument("--module-dir", default=None,
                             help="Override install location (default: ~/ARI/voice-modules/StyleTTS2)")

    args = parser.parse_args()
    if args.command == "install":
        install(args.module_dir)
    else:
        parser.print_help()
