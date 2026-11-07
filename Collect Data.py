"""Pull the latest bars from Yahoo Finance and grow the local history. Run daily (a scheduled task is ideal)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic.config import load_settings
from mosaic import data

cfg = load_settings()
print("Collecting", cfg.symbols, cfg.intervals)
data.collect(cfg.symbols, cfg.intervals)
if len(sys.argv) > 3 and sys.argv[1] == "import":
    # python "Collect Data.py" import path.csv MES 5m
    data.import_csv(sys.argv[2], sys.argv[3], sys.argv[4])
print("Done. Tip: run this every day so the tester has more history to learn from.")
