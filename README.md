# Mosaic — strategy puzzle tester + supervised auto-trader (prop-firm futures)

Mosaic breaks day-trading strategies into puzzle pieces (entry, filters, exit, sizing), mixes them,
and keeps only combinations that hold up on data they never saw. If one survives every check, you
can paper trade it, and later trade it live on TopstepX with hard safety rails.

**Plain truth first:** nothing here is "mathematically proven." The most likely outcome of a research
run is `FAIL` or `WEAK`. That is the tool doing its job — it stops you paying for an evaluation with a
strategy that only looked good by luck. Not financial advice; you can lose money.

## Setup (Windows, Python 3.12+)
1. Unzip to `Downloads\Mosaic`.
2. In Command Prompt: `cd Downloads\Mosaic` then `python install.py`
3. `python "Collect Data.py"` — pulls MES and MNQ bars from Yahoo Finance (free).
   Yahoo only gives 60 days of 5-minute bars, so **run this every day** and the history keeps growing.
   (Task Scheduler → run daily after 5pm ET.) `settings.json` appears after the first run — edit it there.

### More history (recommended)
60 days is thin. Two ways to get years of 5-minute bars:
- Set `"research_interval": "1h"` in settings.json — Yahoo gives 2 years of hourly bars (coarser, but a
  much better test of whether an idea holds up across different markets).
- Import a CSV from a data vendor (Databento gives free starter credits; also FirstRate Data, Kibot):
  `python "Collect Data.py" import C:\path\bars.csv MES 5m`

## Run the puzzle tester
`python "Run Research.py"` (add `--symbol MNQ` for Nasdaq, `--demo` to check the machinery on fake data).

What happens, step by step:
1. **Stage 1** — each entry type is tested alone, walk-forward (tune on old data, judge on newer data).
2. **Stage 2** — best entries × every filter × every exit × sizing. Every template gets its own
   walk-forward; only the unseen-data results are ranked.
3. **Stage 3** — a second filter is tried on the leaders; kept only if unseen results improve.
4. **Checks on the finalists** (the first four are the homemade ones; the rest come from the research in
   the learning pack)
   - *Luck test*: random entries with the same filters/exit/frequency, 100 times. The real result must
     beat what a random search of the same size (~200 ideas) would have found. This is the check that
     kills most "amazing" backtests.
   - *Robustness*: every setting nudged one step. A real edge survives that; a fluke collapses.
   - *Slice consistency*: profitable in most time slices, including the most recent one.
   - *Market mood*: results split into trending vs choppy periods.
   - *PBO* (probability of backtest overfitting, Bailey/Lopez de Prado 2015): did the search pick
     noise? Computed across every template tried. Above 0.5 = FAIL.
   - *Deflated Sharpe*: chance the edge is real given how many ideas were tried.
   - *Stress*: rerun under 2x slippage, 1.5x fees and a 1-bar delay. Loses money = FAIL.
   - *Monte Carlo*: unseen trades reshuffled 2,000 times under the Topstep rules; odds of reaching the
     payout target before breaching the max-loss.
   - *Data quality*: gaps, duplicates, bad prints, timezone, partial sessions.
5. **Puzzle combos** — a regime-switcher (one strategy for trending markets, another for chop) and a
   2-of-3 vote ensemble, both judged on the final untouched slice.
6. **Verdict** `PASS` / `WEAK` / `FAIL` with the reasons in plain words. The winner is refit on all the
   data and saved to `strategies\best_MES.json`. Open `results\report_MES.html` for the full picture.

Runtime: roughly 2–6 minutes per symbol on a normal PC.

Keep it fresh: `python "Run Research.py" --retune` (weekly) only replaces the saved strategy when the new
one is better. The paper/live trader also has a drift monitor that pauses trading when live results fall
well below the backtest.

Every run also writes `results\ledger_*.csv`, the full list of ideas tried (so the search history is never
hidden), and the report labels each number as a research finding or a hypothesis. `PLAYBOOK.md` explains
what the famous quant firms do that transfers to this account, and what does not.

## Paper trade it (fake money, live prices)
`python "Paper Trade.py"` during market hours. Yahoo's futures prices are delayed ~15 minutes, so
paper results are approximate; once you have a TopstepX API key the trader uses its real-time bars.
Give it at least 2–4 weeks, then `python "Compare Paper vs Backtest.py"`: it checks whether the paper
results sit inside the backtest's expected range and says so in plain words.

## Live trade (TopstepX only)
Topstep allows bots through the TopstepX API ($29/mo, half off with code `topstep`) with rules:
**your own PC only (no VPS/cloud), you must supervise it, no high-frequency trading.** Read Topstep's
current rules yourself before starting; they change.
1. In settings.json set `topstepx_username`, `topstepx_api_key`, `topstepx_account_id`, and
   `"live_trading_enabled": true`.
2. Open `mosaic\broker.py` and confirm every line marked `VERIFY` against the API docs in your TopstepX
   dashboard (endpoint names and order-type numbers). There is no sandbox — the first order is real.
3. Best done on a Trading Combine (evaluation) account first, not a funded one.
4. `python "Live Trade.py"` — refuses unless the saved strategy is `PASS` and you type YES.

Safety rails in every mode: every entry gets a stop and target; no entries after 3:30pm ET; everything
flat by 3:55pm ET; daily loss limit → flat and locked for the day; max trades per day; create a file
named `STOP.txt` in the Mosaic folder to flatten and halt; Ctrl+C flattens too.

## Files
- `settings.json` — every knob (fees, slippage, Topstep limits, session times, search sizes).
- `mosaic\pieces.py` — the puzzle pieces. Add your own entry/filter/exit here; the search picks it up.
- `results\` — reports and CSV rankings. `strategies\` — saved winners. `logs\` — trade logs and state.
