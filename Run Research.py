"""Run the puzzle tester.  Options:  --demo (fake data, machinery check)   --retune (only replace the
saved strategy if the new one is better)   --symbol MNQ"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic.config import load_settings
from mosaic.run import run_research

args = sys.argv[1:]
cfg = load_settings()
symbol = args[args.index("--symbol") + 1] if "--symbol" in args else cfg.symbols[0]
run_research(symbol, cfg, demo="--demo" in args, retune="--retune" in args)
