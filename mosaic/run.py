"""Orchestrates one full research run and saves the winner."""
import json, os, time
import numpy as np
from datetime import datetime
import pandas as pd
from .config import CONTRACTS, RESULTS_DIR, STRAT_DIR, load_settings
from . import data, indicators, report, validate
from .research import Researcher
from .strategy import Strategy


def best_path(symbol):
    return os.path.join(STRAT_DIR, f"best_{symbol}.json")


def run_research(symbol, cfg=None, demo=False, retune=False, log=print):
    cfg = cfg or load_settings()
    spec = CONTRACTS[symbol]
    iv = cfg.research_interval
    if demo:
        from .synthetic import make_bars
        raw = make_bars(days=150, trend_days=False)
        log("DEMO MODE: random fake data. Anything that 'passes' here is luck - this only proves the machinery runs.")
    else:
        raw = data.load_bars(symbol, iv)
        if len(raw) == 0:
            log(f"No data for {symbol} {iv}. Run 'Collect Data.py' first (or import a CSV)."); return None
    iv_min = int(iv.rstrip("m")) if iv.endswith("m") else 60
    dq = validate.data_quality(raw, iv_min)
    for issue in dq:
        log(f"DATA QUALITY: {issue}")
    df = indicators.prepare(raw, cfg.rth_start_min)
    n_days = int(df["day_id"].max()) + 1
    log(f"{symbol} {iv}: {len(df):,} bars over {n_days} trading days ({df.index[0]:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d})")
    if n_days < 40:
        log("WARNING: under 40 days of data. Results will be noisy; keep collecting daily and re-run.")

    R = Researcher(df, spec, cfg, log)
    t0 = time.time()
    keep, stage1 = R.stage1_entries()
    results = R.stage2_templates(keep)
    n_templates = len(results)
    top = results[: cfg.top_combos]
    improved = R.stage3_second_filter(top[:5])
    if improved:
        n_templates += len(improved)
        top = sorted(top + improved, key=lambda r: r["oos"].metrics["score"], reverse=True)[: cfg.top_combos]

    log("Checks: luck test, robustness, market-mood split on the finalists...")
    for r in top:
        r["luck"] = R.luck_test(r, n_templates)
        r["robust_pf"], r["robust_frac"] = R.robustness(r)
        r["regime_split"] = R.regime_split(r)
        r["verdict"], r["verdict_reason"] = R.verdict(r["oos"].metrics, r["luck"], r["robust_pf"], r["robust_frac"], cfg, n_templates)
        m = r["oos"].metrics
        log(f"  {r['verdict']:4s} {r['strategy'].template():70s} PF={m['profit_factor']:.2f} net=${m['net']:,.0f} luck-margin={r['luck']['margin']:+.1f}sd nudged={r['robust_pf']:.2f}")

    # Probability of backtest overfitting across EVERYTHING stage 2 tried (final-fold settings)
    log("PBO: probability the search is picking noise (CSCV over all templates)...")
    cols = []
    for r in results:
        s = r["folds"][-1][0]
        cols.append(validate.slice_perf(df, s.signals(df, cfg, R.cache), s, spec, cfg, cfg.cscv_slices))
    pbo_val, _ = validate.pbo(np.array(cols).T, cfg.cscv_slices)
    log(f"  PBO = {pbo_val:.2f}  (0 = trustworthy selection, 0.5 = coin flip, >0.5 = overfit search)")

    extras = {}
    div = R.diverse(top, 5)
    rs = R.regime_switch(div)
    if rs:
        extras["Regime switch"] = {"desc": "one strategy when the market trends, another when it chops",
                                  "strategies": [rs["trend"], rs["chop"]], "final_slice": rs["final_slice"],
                                  "baseline": top[0]["folds"][-1][1].metrics}
    else:
        extras["Regime switch"] = None
    ev = R.ensemble_vote(R.diverse(top, 3))
    if ev:
        extras["Vote (2 of 3 must agree)"] = {"desc": "three different entries vote; trade only when two agree",
                                             "strategies": ev["strategies"], "final_slice": ev["final_slice"],
                                             "baseline": ev["lead_final_slice"].metrics}
    else:
        extras["Vote (2 of 3 must agree)"] = None

    # rank: PASS first, then WEAK, then by score
    order = {"PASS": 0, "WEAK": 1, "FAIL": 2}
    top.sort(key=lambda r: (order[r["verdict"]], -r["oos"].metrics["score"]))
    winner = top[0]
    dsr = validate.deflated_sharpe(winner["oos"].daily.values, n_templates)
    stress = validate.stress(df, winner["folds"], spec, cfg, R.cache)
    mc = validate.monte_carlo(winner["oos"].trades, cfg, runs=cfg.mc_runs)
    log(f"  Deflated Sharpe: p(real) = {dsr['dsr']:.0%}   stress: " + ", ".join(f"{k} ${v:,.0f}" for k, v in stress.items()))
    if mc:
        log(f"  Monte Carlo: hit ${cfg.profit_target:,.0f} target first {mc['p_target_first']:.0%}, breach max-loss first {mc['p_breach_first']:.0%}")
    # these can overturn the verdict
    extra_reasons = []
    if pbo_val > 0.5: extra_reasons.append(f"search is overfit (PBO {pbo_val:.2f} > 0.5)")
    if dsr["dsr"] < 0.5 and winner["verdict"] == "PASS": extra_reasons.append(f"deflated Sharpe says only {dsr['dsr']:.0%} chance the edge is real")
    if stress.get("all three", 0) <= 0: extra_reasons.append("loses money under stressed execution (2x slippage, 1.5x fees, 1-bar delay)")
    if mc and mc["p_breach_first"] > mc["p_target_first"]: extra_reasons.append("more likely to breach the max-loss than reach the payout target")
    if extra_reasons:
        winner["verdict"] = "FAIL" if (pbo_val > 0.5 or stress.get("all three", 0) <= 0) else "WEAK"
        winner["verdict_reason"] = "; ".join([winner["verdict_reason"]] + extra_reasons) if winner["verdict"] != "PASS" else winner["verdict_reason"]
    final = R.final_fit(winner["strategy"])
    info = {"symbol": symbol, "interval": iv, "verdict": winner["verdict"], "verdict_reason": winner["verdict_reason"],
            "oos_metrics": {k: v for k, v in winner["oos"].metrics.items() if k != "fold_nets"},
            "fold_nets": winner["oos"].metrics.get("fold_nets"), "luck": winner["luck"],
            "robust_pf": winner["robust_pf"], "robust_frac": winner["robust_frac"], "regime_split": winner["regime_split"],
            "data_span": [str(df.index[0]), str(df.index[-1])], "trained_at": datetime.now().isoformat(timespec="minutes"),
            "backtests": R.n_backtests, "minutes": round((time.time() - t0) / 60, 1), "demo": demo,
            "description": report.describe(final), "pbo": pbo_val, "dsr": dsr, "stress": stress, "monte_carlo": mc,
            "n_candidates": n_templates, "data_quality": dq,
            "evidence": "research finding: walk-forward backtest with embargo, fees, slippage and prop-firm rules; NOT a live record"}

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    rows = [{"pieces": r["strategy"].template(), **{k: v for k, v in r["oos"].metrics.items() if k != "fold_nets"},
             "luck_beats": r["luck"]["beats"], "luck_margin": r["luck"]["margin"], "robust_pf": r["robust_pf"], "verdict": r["verdict"]} for r in top]
    csv_path = os.path.join(RESULTS_DIR, f"research_{symbol}_{stamp}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    pd.DataFrame(R.ledger, columns=["strategy", "start_bar", "end_bar", "net", "trades"]).to_csv(
        os.path.join(RESULTS_DIR, f"ledger_{symbol}_{stamp}.csv"), index=False)
    html_path = os.path.join(RESULTS_DIR, f"report_{symbol}{'_demo' if demo else ''}.html")
    report.build(symbol, iv, df, stage1, top, extras, final, {**info, "oos_trades": winner["oos"].trades}, cfg, html_path)

    # save / retune
    path = best_path(symbol)
    keep_old = False
    if retune and os.path.exists(path) and not demo:
        _, old = Strategy.load(path)
        old_score = old.get("oos_metrics", {}).get("score", -1e9)
        old_v = order.get(old.get("verdict", "FAIL"), 2)
        if (old_v, -old_score) <= (order[winner["verdict"]], -winner["oos"].metrics["score"]):
            keep_old = True
            log(f"Retune: the current saved strategy ({old.get('verdict')}) still holds up - keeping it.")
    if not keep_old and not demo:
        final.save(path, info)
        log(f"Saved winner -> {path}")
    elif demo:
        final.save(os.path.join(STRAT_DIR, f"demo_{symbol}.json"), info)

    log(f"\nVERDICT: {winner['verdict']} - {winner['verdict_reason']}")
    for line in info["description"]:
        log("  " + line)
    log(f"Report: {html_path}\nRan {R.n_backtests:,} backtests in {info['minutes']} min.")
    return info
