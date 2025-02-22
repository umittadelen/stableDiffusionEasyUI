import os
import subprocess
import sys

def check_virtualenv():
    return os.path.exists('venv')

def create_virtualenv():
    try:
        subprocess.check_call([sys.executable, '-m', 'venv', 'venv'])
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

def activate_virtualenv():
    if os.name == 'nt':
        activate_script = os.path.join('venv', 'Scripts', 'activate.bat')
    else:
        activate_script = os.path.join('venv', 'bin', 'activate')
    
    return activate_script

def install_requirements():
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def run_app():
    try:
        subprocess.check_call([sys.executable, './app.py'])
    except subprocess.CalledProcessError as e:
        print(f"Error running the application: {e}")
        sys.exit(1)

if __name__ == '__main__':
    if not check_virtualenv():
        print("Creating virtual environment...")
        create_virtualenv()
    
    print("Activating virtual environment...")
    activate_script = activate_virtualenv()
    if os.name == 'nt':
        subprocess.call([activate_script], shell=True)
    else:
        subprocess.call(['source', activate_script], shell=True)
    
    print("Installing requirements...")
    install_requirements()
    
    print("Running the application...")
    run_app()