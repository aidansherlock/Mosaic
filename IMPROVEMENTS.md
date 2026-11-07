# What was strengthened while building (and what to improve next)

Done in this build:
1. **Deflated luck test.** A single random comparison lets the best of 200 tries look real. Mosaic now
   estimates the best result a random search of the same size would find and demands the strategy beat it.
   On pure random data this correctly downgraded a "PASS" to a FAIL.
2. **Diversity for combos.** The vote ensemble originally picked three near-identical strategies, so votes
   always agreed. It now requires three different entry types.
3. **Fold consistency + recent-slice rule.** Profitable in most slices and in the newest one.
4. **Conservative fills.** Next-bar entries with slippage, stop-before-target when both hit, gap handling,
   fees both sides, forced flat before the close.
5. **Prop-firm rules inside the backtest.** Daily loss lockout and trailing max-loss (EOD or intraday
   style) — a breach is scored as a failure, not just a drawdown.
6. **Fail-closed live trader.** Refuses without PASS verdict, without the settings flag, without a typed
   YES; flattens if the stop order is rejected; kill-switch file; drift monitor.

Added 11 Oct 2026 from the firms brief + learning pack:
7. **PBO / CSCV** across all templates (Bailey, Borwein, Lopez de Prado, Zhu 2015). Random demo data: 0.14 on a
   search that the luck test already failed; a value above 0.5 is now an automatic FAIL.
8. **Deflated Sharpe** (Bailey & Lopez de Prado 2014) with skew/kurtosis and the number of ideas tried.
9. **Embargo gap** between training and test slices (TimeSeriesSplit-style gap).
10. **Stress tests**: 2x slippage, 1.5x fees, 1-bar delay, all three. Losing under "all three" = FAIL.
11. **Monte Carlo** of unseen trades under the account rules: P(target first) vs P(breach first).
12. **Data-quality gate**, **search ledger**, **evidence labels** in the report, **hypothesis + fails-when** on
    every entry piece, **R-multiple and streak/tail/autocorrelation metrics** (Lo 2002 caveat on Sharpe).
13. New pieces: EventDay filter, OvernightGapFade, ConfirmedBreakout (volume-confirmed, institutional flow).
14. `Compare Paper vs Backtest.py` reconciliation with a 95% range.

Next upgrades (in order of value):
- Get years of 5-minute data (Databento/CSV import). Sixty days is the biggest weakness right now.
- Replace EventDay's fixed clock windows with a dated economic calendar (FOMC/CPI/NFP).
- Portfolio mode: run MES and MNQ strategies together with a shared daily loss budget.
- Replace polling with the TopstepX real-time (SignalR) feed once the API key is in place.
- Monte-Carlo on the OOS trade list to estimate the odds of hitting the max-loss limit before a payout.
