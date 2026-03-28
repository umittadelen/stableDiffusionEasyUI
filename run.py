import os, sys, subprocess, urllib.request, re

# Paths
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
EMBEDDED_PYTHON = os.path.join(SCRIPT_DIR, "python-3.10.6", "python.exe")
PTH_FILE        = os.path.join(SCRIPT_DIR, "python-3.10.6", "python310._pth")
GET_PIP_PATH    = os.path.join(SCRIPT_DIR, "python-3.10.6", "get-pip.py")
VENV_DIR        = os.path.join(SCRIPT_DIR, "venv")
VENV_PYTHON     = os.path.join(VENV_DIR, "Scripts", "python.exe")
VENV_PIP        = os.path.join(VENV_DIR, "Scripts", "pip.exe")
SETUP_MARKER    = os.path.join(SCRIPT_DIR, ".setup_done")

REQUIREMENTS = ["flask", "Pillow", "numpy", "diffusers", "transformers", "compel", "opencv-python", "requests", "accelerate", "matplotlib", "onnxruntime", "pandas"]

def detect_cuda_version():
    try:
        res = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", res.stdout)
        if m: return (int(m.group(1)), int(m.group(2)))
    except: pass
    return None

def pick_torch_index_url(cuda_v):
    if not cuda_v: return None
    v = cuda_v[0] * 100 + cuda_v[1]
    print(f"[setup] Detected CUDA {cuda_v[0]}.{cuda_v[1]}.")
    return "https://download.pytorch.org/whl/" + ("cu130" if v >= 1300 else "cu128" if v >= 1208 else "cu126")

def install_cuda_package(name, pkgs):
    """Generic installer for packages requiring specific CUDA wheels (Torch/Xformers)."""
    pref = pick_torch_index_url(detect_cuda_version())
    all_urls = [f"https://download.pytorch.org/whl/{u}" for u in ["cu130", "cu128", "cu126"]]
    candidates = ([pref] if pref else []) + [u for u in all_urls if u != pref] + [None]

    for url in candidates:
        label = url or "CPU (default)"
        print(f"[setup] Installing {name} — index: {label} ...")
        cmd = [VENV_PIP, "install"] + pkgs + (["--index-url", url] if url else [])
        if subprocess.run(cmd).returncode == 0:
            # Check logic: xformers doesn't have .cuda, so we check availability via torch if name is xformers
            check_attr = "cuda.is_available()" if name.lower() == "torch" else "is_available()"
            verify = subprocess.run([VENV_PYTHON, "-c", f"import {name.lower()}; print({name.lower()}.__version__); print('CUDA:', {name.lower()}.{check_attr})"], capture_output=True, text=True)
            if verify.returncode == 0:
                print(f"[setup] {name} OK — {verify.stdout.strip()}")
                return
    print(f"[setup] WARNING: Could not install {name}. Check internet/drivers.")

def enable_site_packages():
    if os.path.exists(PTH_FILE):
        with open(PTH_FILE, "r+") as f:
            c = f.read()
            if "#import site" in c:
                f.seek(0); f.write(c.replace("#import site", "import site")); f.truncate()
                print("[setup] Enabled site-packages.")

def setup():
    print("=" * 60 + "\n  First-time setup\n" + "=" * 60)
    if not os.path.exists(GET_PIP_PATH): urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", GET_PIP_PATH)
    
    steps = [
        ([EMBEDDED_PYTHON, GET_PIP_PATH, "--no-warn-script-location"], "pip"),
        ([EMBEDDED_PYTHON, "-m", "pip", "install", "virtualenv"], "virtualenv"),
        ([EMBEDDED_PYTHON, "-m", "virtualenv", VENV_DIR], "venv creation")
    ]
    enable_site_packages()
    for cmd, msg in steps:
        print(f"[setup] Installing {msg}...")
        subprocess.run(cmd, check=True)

    install_cuda_package("Torch", ["torch", "torchvision"])
    install_cuda_package("xformers", ["xformers"])
    
    print("[setup] Installing remaining requirements ...")
    subprocess.run([VENV_PIP, "install"] + REQUIREMENTS, check=True)
    with open(SETUP_MARKER, "w") as f: f.write("1")

def main():
    if not os.path.exists(EMBEDDED_PYTHON):
        print(f"[error] Embedded Python not found at {EMBEDDED_PYTHON}"); sys.exit(1)

    if not os.path.exists(SETUP_MARKER) or not os.path.exists(VENV_PYTHON):
        setup()

    ext_dir = os.path.join(SCRIPT_DIR, "extensions")
    if os.path.isdir(ext_dir):
        for name in sorted(os.listdir(ext_dir)):
            req = os.path.join(ext_dir, name, "requirements.txt")
            if os.path.isfile(req):
                print(f"[extensions] Installing requirements for '{name}'...")
                subprocess.run([VENV_PIP, "install", "-r", req])

    print("[run] Launching app.py ...")
    try:
        sys.exit(subprocess.run([VENV_PYTHON, os.path.join(SCRIPT_DIR, "app.py")]).returncode)
    except KeyboardInterrupt: sys.exit(0)

if __name__ == "__main__":
    main()