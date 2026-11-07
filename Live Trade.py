"""REAL orders on TopstepX. Needs: live_trading_enabled=true + API details in settings.json, a PASS
strategy, and you at the keyboard (Topstep requires supervised, local execution). Create STOP.txt to halt."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic.live import run_trader
args = sys.argv[1:]
run_trader(symbol=args[args.index("--symbol") + 1] if "--symbol" in args else None, live=True)
