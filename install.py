"""One-time setup: installs the Python packages Mosaic needs."""
import subprocess, sys
pkgs = ["numpy", "pandas", "scipy", "yfinance", "requests"]
print("Installing:", ", ".join(pkgs))
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", *pkgs])
print("\nDone. Next: run 'Collect Data.py', then 'Run Research.py'.")
