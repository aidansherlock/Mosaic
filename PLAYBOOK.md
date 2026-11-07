# Playbook: what the exceptional firms teach a one-contract account

Source: the "Exceptional Trading Firms" brief and "AI Trading Learning Pack" (both 11 Oct 2026).
Evidence labels follow the pack: **primary** = the firm says it, **reported** = a publisher says it,
**inference** = my reading of it. Nothing here is a copied strategy; none of these firms publish one.

## The records, honestly read
| Firm | Reported record | What it actually tells you |
|---|---|---|
| Medallion | ~39% net/yr 1988-2018 (reported, Cornell via Zuckerman) | Thousands of small short-term bets across many markets, capacity-capped, closed to outsiders. Inference: the edge is breadth and infrastructure, not one signal. A single MES contract cannot copy this. |
| Druckenmiller | ~30%/yr, no losing year (primary, first-person, fee basis unknown) | Discretionary macro, multi-day holds. Lesson: concentrate when conviction is high, size down fast when wrong. |
| Thorp 1992-2002 | 20% net (primary paper) | 200 long / 200 short, ~10-day turnover; he reports costs explicitly (26% before fee, 20% net). The only case where the mechanism AND the cost drag are both published. |
| D.E. Shaw Oculus | 14.4% net/yr 2004-2025 (reported) | Many strategies blended; the headline is diversification, not a hero trade. |
| Citadel Wellington | 19.5%/yr 1990-2024 (reported) with ~-55% in 2008 | A great long-run record can include a near-death year. Drawdown control is a separate skill from return. |
| PDT, TGS, Jane Street, XTX, HRT | No comparable investor return series | Business descriptions only. Secrecy is not evidence. Revenue is not return. |

## What transfers (and where it lives in Mosaic)
1. **Breadth beats brilliance.** Medallion, Thorp and Shaw all run many small edges. Mosaic's version:
   run MES and MNQ, several pieces, and prefer a vote/regime combo over one hero template. Next step:
   portfolio mode with a shared daily loss budget.
2. **Count the costs first.** Thorp's 26% -> 20% gap is fees. Mosaic applies fees + slippage on every
   fill and now re-runs the winner under 2x slippage / 1.5x fees / 1-bar delay (`stress` in the report).
3. **The search itself is the enemy.** The pack's core warning: try 200 ideas and one looks great by
   chance. Mosaic answers with four independent checks: deflated luck test, PBO (Bailey et al. 2015),
   deflated Sharpe, and settings-nudge robustness. A full ledger of every idea tried is saved.
4. **Leakage.** Walk-forward with an embargo gap between training and test; indicators computed only
   from past bars; entries fill on the next bar.
5. **Survive first.** Citadel's 2008 and Kelly over-betting research: the prop-firm rules (daily loss,
   trailing max loss) are inside the backtest, and a Monte Carlo reports the odds of breaching before a payout.
6. **Data is the product (HRT).** Data-quality checks run before research and are printed in the report.
7. **Write the hypothesis before the test.** Every entry piece carries a mechanism, a hypothesis and a
   "fails when"; the report prints them, labelled as hypotheses.
8. **Reconcile paper to backtest** before any real order (`Compare Paper vs Backtest.py`).

## What does NOT transfer
- Their signals. 13F filings, interviews and websites cannot recover a quant fund's strategy.
- Their capital base, co-location, and data budgets. A retail 5-minute bar from Yahoo is not their feed.
- Multi-day macro holding (Duquesne). Prop-firm day-trading rules force flat by the close.

## Five open questions Mosaic cannot answer yet
1. Does any 5-minute edge survive once years of data (not 60 days) are in the tester?
2. How much of the MES/MNQ intraday return is just the overnight drift the day session doesn't capture?
3. Does the EventDay filter help once a real economic calendar replaces fixed clock windows?
4. What is the realistic slippage on TopstepX fills vs the 1-tick assumption?
5. Does the paper record match the backtest within its confidence range (the reconcile script decides)?
