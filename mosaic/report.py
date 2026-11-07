"""HTML report + plain-English strategy descriptions."""
import html
import numpy as np
import pandas as pd


def describe(strat):
    """Plain-English description of a strategy, one line per piece."""
    lines = []
    for p in strat.pieces():
        doc = (type(p).__doc__ or "").strip().split("\n")[0]
        ps = ", ".join(f"{k} {v}" for k, v in p.params.items())
        lines.append(f"{p.kind.upper():6s} {p.name} ({ps}) - {doc}")
    return lines


def hypotheses(strat):
    out = []
    for p in strat.pieces():
        if p.hypothesis:
            out.append((p.name, p.mechanism, p.hypothesis, p.fails_when))
    return out


def equity_svg(trades, width=760, height=220):
    if trades is None or len(trades) == 0:
        return "<p>No trades.</p>"
    eq = np.concatenate([[0.0], np.cumsum(trades["pnl"].values)])
    lo, hi = eq.min(), eq.max()
    span = (hi - lo) or 1.0
    pts = []
    for i, v in enumerate(eq):
        x = 10 + i / max(len(eq) - 1, 1) * (width - 20)
        y = 10 + (hi - v) / span * (height - 20)
        pts.append(f"{x:.1f},{y:.1f}")
    zero_y = 10 + (hi - 0) / span * (height - 20)
    color = "#16a34a" if eq[-1] >= 0 else "#dc2626"
    return (f'<svg width="{width}" height="{height}" style="background:#0b0f19;border-radius:8px">'
            f'<line x1="10" x2="{width-10}" y1="{zero_y:.1f}" y2="{zero_y:.1f}" stroke="#334" stroke-dasharray="4"/>'
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(pts)}"/>'
            f'<text x="14" y="20" fill="#9ca3af" font-size="12">unseen-data equity, ${eq[-1]:,.0f} over {len(trades)} trades</text></svg>')


def money(v):
    return f"${v:,.0f}"


def table(rows, headers):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in headers)
    body = "".join("<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>"


def build(symbol, interval, df, stage1, top, extras, best, best_info, cfg, path):
    css = """body{font-family:Segoe UI,Arial,sans-serif;background:#f6f7fb;color:#111;margin:0;padding:24px;max-width:960px}
    h1{margin:0 0 4px}h2{margin-top:28px;border-bottom:2px solid #ddd;padding-bottom:4px}
    table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}th,td{border:1px solid #e5e7eb;padding:6px 8px;text-align:left}
    th{background:#eef}code{background:#eef;padding:1px 4px}.pass{color:#16a34a;font-weight:bold}.weak{color:#d97706;font-weight:bold}.fail{color:#dc2626;font-weight:bold}
    .box{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:12px 16px;margin:10px 0}"""
    span = f"{df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} ({df['day_id'].max()+1} trading days, {len(df):,} bars)"
    out = [f"<html><head><meta charset='utf-8'><title>Mosaic report {symbol}</title><style>{css}</style></head><body>",
           f"<h1>Mosaic research report: {symbol} {interval}</h1><div>Data: {span}. Fees, slippage and "
           f"{cfg.trailing_mode} trailing max-loss (${cfg.max_loss_limit:,.0f}) / daily loss (${cfg.daily_loss_limit:,.0f}) rules applied.</div>"]

    out.append("<div class='box' style='font-size:13px'><b>How to read this report (evidence labels):</b> every number below is a "
               "<b>research finding</b> from a walk-forward backtest with fees, slippage and prop-firm rules on the stated data. "
               "None of it is a live trading record. Hypotheses are labelled as hypotheses. Where a number could not be computed it says so.</div>")
    dq = best_info.get("data_quality") or []
    out.append("<div class='box'><b>Data quality:</b> " + ("no problems found" if not dq else "; ".join(html.escape(x) for x in dq)) + "</div>")
    v, why = best_info["verdict"], best_info["verdict_reason"]
    out.append(f"<h2>Verdict on the best combination</h2><div class='box'><span class='{v.lower()}'>{v}</span> - {html.escape(why)}")
    out.append("<br><br><b>The strategy, piece by piece:</b><ul>" + "".join(f"<li>{html.escape(l)}</li>" for l in describe(best)) + "</ul>")
    m = best_info["oos_metrics"]
    out.append(table([[m["trades"], f"{m['profit_factor']:.2f}", money(m["net"]), money(m["expectancy"]), f"{m['win_rate']:.0%}",
                       money(m["max_dd"]), f"{m['sharpe']:.2f}", f"{m.get('folds_profitable',0):.0%}", f"{best_info['luck']['margin']:+.1f} sd",
                       f"{best_info['robust_pf']:.2f} / {best_info['robust_frac']:.0%}"]],
                     ["unseen trades", "profit factor", "net", "per trade", "win rate", "max drawdown", "sharpe",
                      "slices profitable", "beats random search by", "nudged settings PF / % profitable"]))
    out.append("<p style='font-size:12px;color:#555'>Profit factor = $ won / $ lost (need well above 1). 'Beats random search by' = how many standard deviations the result "
               "sits above the best a random search of the same size would find (must be above 0). Nudged = median profit factor when each setting is moved one step (a real edge survives that).</p>")
    out.append(equity_svg(best_info["oos_trades"]))
    if "avg_win_R" in m:
        out.append(f"<p><b>In risk units (R):</b> average win {m['avg_win_R']:+.2f}R, average loss {m['avg_loss_R']:+.2f}R, "
                   f"expectancy {m['expectancy_R']:+.2f}R per trade = win rate x avg win - loss rate x avg loss, after costs. "
                   f"Longest stretch under a prior equity high: {m.get('longest_underwater_trades', 0)} trades; "
                   f"worst losing streak {m.get('max_consecutive_losses', 0)}; worst 5% of trades average {money(m.get('tail_5pct_avg', 0))}.</p>")
    pbo = best_info.get("pbo"); dsr = best_info.get("dsr") or {}; st = best_info.get("stress") or {}; mc = best_info.get("monte_carlo") or {}
    out.append("<h3>Overfitting and friction checks</h3>")
    out.append(table([[f"{pbo:.2f}" if pbo == pbo else "n/a", f"{dsr.get('dsr', 0):.0%}", f"{dsr.get('sr_annual', 0):.2f}", best_info.get("n_candidates", "")]],
                     ["PBO (prob. the search is overfit; want < 0.5)", "deflated Sharpe: chance the edge is real", "annualized Sharpe", "ideas tried"]))
    out.append(table([[k, money(v)] for k, v in st.items()], ["execution assumption", "net on unseen data"]))
    if mc:
        out.append(f"<p><b>Monte Carlo under the account rules</b> ({mc['horizon_days']} trading days, trades reshuffled by day): "
                   f"reach the ${cfg.profit_target:,.0f} target first <b>{mc['p_target_first']:.0%}</b>, breach the max-loss first <b>{mc['p_breach_first']:.0%}</b>, "
                   f"neither {mc['p_neither']:.0%}; median {mc['median_days_to_target']:.0f} days to target; 95th-percentile drawdown {money(mc['dd_95pct'])}.</p>")
    hs = hypotheses(best)
    if hs:
        out.append("<h3>Why it should work (hypotheses, not findings)</h3>" + table(
            [[n, mech, h, f] for n, mech, h, f in hs], ["piece", "mechanism", "hypothesis", "fails when"]))
    rs = best_info.get("regime_split") or {}
    if rs:
        out.append("<p><b>By market mood (unseen data):</b> " + "; ".join(
            f"{k}: {v['trades']} trades, {money(v['net'])}, PF {v['pf']:.2f}" for k, v in rs.items()) + "</p></div>")
    else:
        out.append("</div>")

    out.append("<h2>Top combinations (ranked on unseen data only)</h2>")
    rows = []
    for r in top:
        mm = r["oos"].metrics
        rows.append([r["strategy"].template().replace("|", " + "), mm["trades"], f"{mm['profit_factor']:.2f}", money(mm["net"]),
                     money(mm["max_dd"]), f"{mm.get('folds_profitable',0):.0%}", f"{r['luck']['margin']:+.1f} sd" if isinstance(r.get('luck'), dict) else "",
                     f"{r.get('robust_pf', float('nan')):.2f}", r.get("verdict", "")])
    out.append(table(rows, ["pieces", "trades", "PF", "net", "max DD", "slices +", "vs random search", "nudged PF", "verdict"]))

    out.append("<h2>Puzzle combinations</h2>")
    for name, ex in extras.items():
        if ex is None:
            out.append(f"<div class='box'><b>{name}</b>: not enough variety to build one.</div>")
            continue
        fm = ex["final_slice"].metrics
        base = ex.get("baseline")
        txt = f"final untouched slice: {fm['trades']} trades, PF {fm['profit_factor']:.2f}, net {money(fm['net'])}, max DD {money(fm['max_dd'])}"
        if base is not None:
            txt += f" &nbsp;|&nbsp; best single strategy on the same slice: net {money(base['net'])}, PF {base['profit_factor']:.2f}"
        out.append(f"<div class='box'><b>{name}</b> - {ex['desc']}<br>{txt}<br><ul>" +
                   "".join(f"<li>{html.escape(s.label())}</li>" for s in ex["strategies"]) + "</ul></div>")

    out.append("<h2>Stage 1: each entry on its own</h2>")
    out.append(table([[E.name, mm["trades"], f"{mm['profit_factor']:.2f}", money(mm["net"]), money(mm["max_dd"])] for E, mm in stage1],
                     ["entry", "unseen trades", "PF", "net", "max DD"]))
    out.append("<p style='font-size:12px;color:#555'>Mosaic is a research tool, not financial advice. Past results, even on unseen data, do not guarantee future profits.</p></body></html>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path
