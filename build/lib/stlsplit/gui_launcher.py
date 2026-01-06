import sys
import subprocess
from pathlib import Path

def main():
    gui_path = Path(__file__).parent / "gui.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(gui_path)])

if __name__ == "__main__":
    main()
