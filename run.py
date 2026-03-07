"""
run.py — Bootstrap launcher for stableDiffusionEasyUI
------------------------------------------------------
Designed to be called by the bundled embeddable Python:

    python-3.10.6\python.exe run.py

On first run it will:
  1. Enable site-packages in the embedded Python
  2. Bootstrap pip into the embedded Python
  3. Install virtualenv
  4. Create a venv at ./venv/
  5. Detect the installed CUDA driver version and install the best
     matching PyTorch build (cu126 / cu128 / cu130 / CPU fallback)
  6. Install all remaining requirements into the venv

On every subsequent run it skips all of the above and launches app.py
directly with the venv Python.
"""

import os
import sys
import subprocess
import urllib.request
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
EMBEDDED_PYTHON = os.path.join(SCRIPT_DIR, "python-3.10.6", "python.exe")
PTH_FILE        = os.path.join(SCRIPT_DIR, "python-3.10.6", "python310._pth")
GET_PIP_PATH    = os.path.join(SCRIPT_DIR, "python-3.10.6", "get-pip.py")
VENV_DIR        = os.path.join(SCRIPT_DIR, "venv")
VENV_PYTHON     = os.path.join(VENV_DIR, "Scripts", "python.exe")
VENV_PIP        = os.path.join(VENV_DIR, "Scripts", "pip.exe")
SETUP_MARKER    = os.path.join(SCRIPT_DIR, ".setup_done")

# Packages installed straight into the venv (torch handled separately)
REQUIREMENTS = [
    "flask",
    "Pillow",
    "numpy",
    "diffusers",
    "transformers",
    "compel",
    "opencv-python",
    "requests",
    "accelerate",
    "matplotlib",
    "onnxruntime",
    "pandas",
]

# ---------------------------------------------------------------------------
# CUDA detection
# ---------------------------------------------------------------------------

def detect_cuda_version():
    """Return (major, minor) CUDA version from nvidia-smi, or None."""
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", result.stdout)
            if match:
                return (int(match.group(1)), int(match.group(2)))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def pick_torch_index_url(cuda_version):
    """
    Return the best PyTorch wheel index URL for the detected CUDA version.
    PyTorch CUDA builds run on drivers >= the build's CUDA version, so we
    pick the highest build that does not exceed the driver version.
    """
    if cuda_version is None:
        print("[setup] No NVIDIA GPU / CUDA detected — will install CPU-only PyTorch.")
        return None

    major, minor = cuda_version
    v = major * 100 + minor  # e.g. 1206, 1208, 1300
    print(f"[setup] Detected CUDA {major}.{minor}.")

    if v >= 1300:
        return "https://download.pytorch.org/whl/cu130"
    elif v >= 1208:
        return "https://download.pytorch.org/whl/cu128"
    else:
        # cu126 is the oldest wheel set offered; works on CUDA 12.6+
        return "https://download.pytorch.org/whl/cu126"


# ---------------------------------------------------------------------------
# Setup steps
# ---------------------------------------------------------------------------

def enable_site_packages():
    with open(PTH_FILE, "r") as f:
        content = f.read()
    if "#import site" in content:
        content = content.replace("#import site", "import site")
        with open(PTH_FILE, "w") as f:
            f.write(content)
        print("[setup] Enabled site-packages in embedded Python.")


def bootstrap_pip():
    if not os.path.exists(GET_PIP_PATH):
        print("[setup] Downloading get-pip.py ...")
        urllib.request.urlretrieve(
            "https://bootstrap.pypa.io/get-pip.py", GET_PIP_PATH
        )
    print("[setup] Bootstrapping pip into embedded Python ...")
    subprocess.run(
        [EMBEDDED_PYTHON, GET_PIP_PATH, "--no-warn-script-location"],
        check=True
    )


def install_virtualenv():
    print("[setup] Installing virtualenv into embedded Python ...")
    subprocess.run(
        [EMBEDDED_PYTHON, "-m", "pip", "install", "virtualenv",
         "--no-warn-script-location"],
        check=True
    )


def create_venv():
    print(f"[setup] Creating venv at: {VENV_DIR} ...")
    subprocess.run(
        [EMBEDDED_PYTHON, "-m", "virtualenv", VENV_DIR],
        check=True
    )


def install_torch():
    """
    Install torch + torchvision using the best matching CUDA index URL.
    Falls back through all remaining CUDA builds and finally CPU if needed.
    """
    cuda_version = detect_cuda_version()
    preferred_url = pick_torch_index_url(cuda_version)

    # Build ordered candidate list: preferred first, then other CUDA builds,
    # then CPU as last resort.
    all_cuda_urls = [
        "https://download.pytorch.org/whl/cu130",
        "https://download.pytorch.org/whl/cu128",
        "https://download.pytorch.org/whl/cu126",
    ]

    if preferred_url is not None:
        candidates = [preferred_url]
        for u in all_cuda_urls:
            if u != preferred_url:
                candidates.append(u)
        candidates.append(None)  # CPU last resort
    else:
        candidates = [None]  # CPU only

    for url in candidates:
        label = url if url else "CPU (default)"
        print(f"[setup] Installing PyTorch — index: {label} ...")
        cmd = [VENV_PIP, "install", "torch", "torchvision"]
        if url:
            cmd += ["--index-url", url]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            print(f"[setup] Install failed for {label}, trying next ...")
            continue

        # Quick sanity check
        verify = subprocess.run(
            [VENV_PYTHON, "-c",
             "import torch; print(torch.__version__); "
             "print('CUDA available:', torch.cuda.is_available())"],
            capture_output=True, text=True
        )
        if verify.returncode == 0:
            print(f"[setup] PyTorch OK — {verify.stdout.strip()}")
            return

    print("[setup] WARNING: Could not install PyTorch. "
          "Check your internet connection and NVIDIA drivers.")


def install_requirements():
    print("[setup] Installing remaining requirements ...")
    subprocess.run([VENV_PIP, "install"] + REQUIREMENTS, check=True)


def install_extension_requirements():
    """Install dependencies declared by extensions via requirements.txt."""
    extensions_dir = os.path.join(SCRIPT_DIR, "extensions")
    if not os.path.isdir(extensions_dir):
        return

    for name in sorted(os.listdir(extensions_dir)):
        ext_dir = os.path.join(extensions_dir, name)
        if not os.path.isdir(ext_dir):
            continue
        req_file = os.path.join(ext_dir, "requirements.txt")
        if not os.path.isfile(req_file):
            continue
        print(f"[extensions] Installing requirements for '{name}' ...")
        try:
            subprocess.run(
                [VENV_PIP, "install", "-r", req_file],
                check=True,
            )
            print(f"[extensions] '{name}' requirements installed.")
        except subprocess.CalledProcessError as exc:
            print(f"[extensions] WARNING: Failed to install requirements for '{name}': {exc}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def setup():
    print("=" * 60)
    print("  First-time setup — this will only run once.")
    print("=" * 60)
    try:
        enable_site_packages()
        bootstrap_pip()
        install_virtualenv()
        create_venv()
        install_torch()
        install_requirements()
    except Exception as exc:
        print(f"\n[setup] ERROR: {exc}")
        print("[setup] Fix the issue above and re-run.")
        sys.exit(1)

    with open(SETUP_MARKER, "w") as f:
        f.write("1")
    print("[setup] Setup complete!\n")


def main():
    if not os.path.exists(EMBEDDED_PYTHON):
        print("[error] Embedded Python not found:", EMBEDDED_PYTHON)
        print("[error] Make sure the 'python-3.10.6' folder is next to run.py.")
        sys.exit(1)

    needs_setup = (
        not os.path.exists(SETUP_MARKER)
        or not os.path.exists(VENV_PYTHON)
    )
    if needs_setup:
        setup()

    install_extension_requirements()

    print("[run] Launching app.py ...")
    app_path = os.path.join(SCRIPT_DIR, "app.py")
    try:
        result = subprocess.run([VENV_PYTHON, app_path])
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
