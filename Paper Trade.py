"""Trade the saved strategy with FAKE money on live prices. Run during market hours and watch it."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic.live import run_trader
args = sys.argv[1:]
run_trader(symbol=args[args.index("--symbol") + 1] if "--symbol" in args else None, live=False)
