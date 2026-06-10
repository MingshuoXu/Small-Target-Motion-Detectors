from pathlib import Path
import subprocess
import sys

def main():
	script_path = Path(__file__).resolve().parent / 'demo' / 'inference_gui.py'
	result = subprocess.run([sys.executable, str(script_path)], check=False)
	return result.returncode


if __name__ == '__main__':
	raise SystemExit(main())

