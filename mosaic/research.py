"""The puzzle tester.

Stage 1  score every ENTRY on its own (walk-forward). Keep the best few.
Stage 2  build templates = entry x filter x exit x sizer. For each template run a walk-forward:
         on each slice's TRAIN data a greedy search picks the piece settings, then the settings are
         judged on the UNSEEN test slice. Only unseen results are ranked.
Stage 3  try adding a second filter to the best templates (kept only if unseen results improve).
Checks   luck test (random entries, same frequency), robustness (nudge each setting), regime split.
Combos   regime-switching strategy and a voting ensemble, validated on the final untouched slice.
Final    refit the winning template on all data and save it for paper/live trading.
"""
import itertools
import time
import numpy as np
import pandas as pd
from . import pieces as P
from . import engine
from .strategy import Strategy

BASE_FILTER = P.SessionWindow(start="09:30", end="15:30")


def fold_bounds(df, k):
    """Split bars into k equal date-based slices -> list of (start, end) bar indexes."""
    days = df["day_id"].values
    n_days = days.max() + 1
    edges = np.linspace(0, n_days, k + 1).astype(int)
    bounds = []
    for a, b in zip(edges[:-1], edges[1:]):
        s = int(np.searchsorted(days, a)); e = int(np.searchsorted(days, b))
        bounds.append((s, e))
    return bounds


def combine_oos(results, cfg):
    trades = pd.concat([r.trades for r in results], ignore_index=True) if results else pd.DataFrame()
    daily = pd.concat([r.daily for r in results]) if results else pd.Series(dtype=float)
    breached = any(r.breached for r in results)
    return engine.Result(trades, daily, breached, cfg)


class Researcher:
    def __init__(self, df, spec, cfg, log=print):
        self.df, self.spec, self.cfg, self.log = df, spec, cfg, log
        self.cache = {}
        self.bounds = fold_bounds(df, cfg.folds)
        self.n_backtests = 0
        self.ledger = []      # every candidate tried, so the search history is never hidden

    # ------------------------------------------------------------------ helpers
    def bt(self, strat, start, end):
        self.n_backtests += 1
        sig = strat.signals(self.df, self.cfg, self.cache)
        res = engine.run(self.df, sig, strat.exit.rules(), strat.size, self.spec, self.cfg, start, end)
        self.ledger.append((strat.label(), start, end, res.metrics["net"], res.metrics["trades"]))
        return res

    def greedy_fit(self, strat, start, end):
        """Coordinate search on TRAIN data only: exit -> filters -> entry -> size, one pass each."""
        best = strat
        best_s = self.bt(best, start, end).metrics["score"]

        def try_variants(make):
            nonlocal best, best_s
            for cand in make(best):
                s = self.bt(cand, start, end).metrics["score"]
                if s > best_s:
                    best, best_s = cand, s

        try_variants(lambda b: [b.with_(exit_=v) for v in type(b.exit).variants()])
        for fi in range(len(best.filters)):
            def mk(b, fi=fi):
                out = []
                for v in type(b.filters[fi]).variants():
                    fl = list(b.filters); fl[fi] = v
                    out.append(b.with_(filters=fl))
                return out
            try_variants(mk)
        try_variants(lambda b: [b.with_(entry=v) for v in type(b.entry).variants()])
        try_variants(lambda b: [b.with_(size=v) for v in type(b.size).variants()])
        return best, best_s

    def walk_forward(self, strat):
        """Returns (oos Result, per-fold list of (chosen strategy, oos Result)).
        An embargo of cfg.embargo_days sits between train and test so no trade straddles the line."""
        folds = []
        days = self.df["day_id"].values
        for j in range(1, len(self.bounds)):
            train_s, train_e = self.bounds[0][0], self.bounds[j - 1][1]
            test_s, test_e = self.bounds[j]
            cut_day = days[train_e - 1] - self.cfg.embargo_days + 1
            train_e = int(np.searchsorted(days, cut_day))
            chosen, _ = self.greedy_fit(strat, train_s, train_e)
            res = self.bt(chosen, test_s, test_e)
            res.fold = (test_s, test_e)
            folds.append((chosen, res))
        oos = combine_oos([r for _, r in folds], self.cfg)
        nets = [r.metrics["net"] for _, r in folds]
        oos.metrics["folds_profitable"] = float(np.mean([n > 0 for n in nets])) if nets else 0.0
        oos.metrics["fold_nets"] = nets
        return oos, folds

    # ------------------------------------------------------------------ stages
    def stage1_entries(self):
        self.log("Stage 1: scoring each entry on its own (walk-forward)...")
        rows = []
        for E in P.ENTRIES:
            strat = Strategy(E(), [BASE_FILTER], P.FixedBracket(), P.FixedContracts(n=1))
            oos, folds = self.walk_forward(strat)
            m = oos.metrics
            rows.append((E, m))
            self.log(f"  {E.name:22s} OOS trades={m['trades']:4d} PF={m['profit_factor']:.2f} "
                     f"net=${m['net']:,.0f} maxDD=${m['max_dd']:,.0f} score={m['score']:.1f}")
        rows.sort(key=lambda r: r[1]["score"], reverse=True)
        keep = [E for E, _ in rows[: self.cfg.top_entries]]
        self.log(f"  keeping: {', '.join(E.name for E in keep)}")
        return keep, rows

    def stage2_templates(self, entries):
        self.log("Stage 2: mixing entries x filters x exits x sizing (walk-forward each)...")
        results = []
        filters = [None] + list(P.FILTERS)
        combos = list(itertools.product(entries, filters, P.EXITS, P.SIZES))
        t0 = time.time()
        for n, (E, F, X, S) in enumerate(combos, 1):
            fl = [BASE_FILTER] if F is None else [BASE_FILTER, F()]
            if F is P.SessionWindow:
                fl = [F()]  # replaces the base window instead of stacking two windows
            strat = Strategy(E(), fl, X(), S())
            oos, folds = self.walk_forward(strat)
            results.append({"strategy": strat, "oos": oos, "folds": folds})
            if n % 10 == 0 or n == len(combos):
                el = time.time() - t0
                self.log(f"  {n}/{len(combos)} templates done ({el/60:.1f} min, ~{el/n*(len(combos)-n)/60:.1f} min left)")
        results.sort(key=lambda r: r["oos"].metrics["score"], reverse=True)
        return results

    def stage3_second_filter(self, top):
        self.log("Stage 3: trying a second filter on the best templates...")
        improved = []
        for r in top:
            base = r["strategy"]
            have = {type(f) for f in base.filters}
            for F in P.FILTERS:
                if F in have or len(base.filters) + 3 >= self.cfg.max_pieces + 1:
                    continue
                strat = base.with_(filters=base.filters + [F()])
                oos, folds = self.walk_forward(strat)
                if oos.metrics["score"] > r["oos"].metrics["score"]:
                    improved.append({"strategy": strat, "oos": oos, "folds": folds})
                    self.log(f"  + {F.name} improved {base.template()} "
                             f"(score {r['oos'].metrics['score']:.1f} -> {oos.metrics['score']:.1f})")
        return improved

    # ------------------------------------------------------------------ checks
    def luck_test(self, r, n_searched=1):
        """Random entries (same filters, exit, frequency), many times. Two answers:
        beats  = share of single random runs that matched the real result
        margin = how far the real result sits above the BEST result a random search of n_searched
                 tries would be expected to find (in standard deviations). Below 0 = a random search
                 would have found something just as good, so the edge is probably a fluke."""
        rng = np.random.default_rng(7)
        real = r["oos"].metrics["net"]
        fold_info = []
        for chosen, oos in r["folds"]:
            ts, te = self._fold_of(oos)
            sig = chosen.signals(self.df, self.cfg, self.cache)
            n = int((sig[ts:te] != 0).sum())
            ok = np.where(self._allowed_mask(chosen)[ts:te])[0] + ts
            fold_info.append((chosen, ts, te, n, ok))
        nets = []
        for _ in range(self.cfg.luck_runs):
            fold_res = []
            for chosen, ts, te, n, ok in fold_info:
                rand = np.zeros(len(self.df), dtype=np.int8)
                if n and len(ok):
                    pick = rng.choice(ok, size=min(n, len(ok)), replace=False)
                    rand[pick] = rng.choice([-1, 1], size=len(pick))
                fold_res.append(engine.run(self.df, rand, chosen.exit.rules(), chosen.size,
                                           self.spec, self.cfg, ts, te))
            nets.append(combine_oos(fold_res, self.cfg).metrics["net"])
        nets = np.array(nets)
        mu, sd = float(nets.mean()), float(nets.std()) or 1.0
        expected_max = mu + sd * np.sqrt(2 * np.log(max(n_searched, 2)))
        return {"beats": float((nets >= real).mean()), "mean": mu, "std": sd,
                "expected_max": float(expected_max), "margin": float((real - expected_max) / sd)}

    def _fold_of(self, oos):
        return getattr(oos, "fold", self.bounds[-1])

    def _allowed_mask(self, strat):
        ones = np.ones(len(self.df), dtype=np.int8)
        for f in strat.filters:
            ones = f.apply(ones, self.df, self.cfg)
        return ones != 0

    def robustness(self, r):
        """Nudge each entry/exit setting one step. If neighbours fall apart, the result is fragile."""
        final = r["folds"][-1][0]
        pfs = []
        for cand in ([final.with_(entry=v) for v in final.entry.neighbours()] +
                     [final.with_(exit_=v) for v in final.exit.neighbours()]):
            res = combine_oos([self.bt(cand, s, e) for (s, e) in self.bounds[1:]], self.cfg)
            pfs.append(res.metrics["profit_factor"])
        if not pfs:
            return 1.0, 1.0
        return float(np.median(pfs)), float(np.mean(np.array(pfs) > 1.0))

    def regime_split(self, r):
        t = r["oos"].trades
        if len(t) == 0:
            return {}
        regime = self.regime_labels()
        lab = regime.reindex(pd.DatetimeIndex(t.entry_time)).values
        out = {}
        for name in ("trend", "chop"):
            p = t.pnl.values[lab == name]
            out[name] = {"trades": int(len(p)), "net": float(p.sum()),
                         "pf": float(p[p > 0].sum() / -p[p <= 0].sum()) if (p <= 0).any() and (p <= 0).sum() and -p[p <= 0].sum() > 0 else 0.0}
        return out

    def regime_labels(self):
        return pd.Series(np.where(self.df["adx"].values >= 25, "trend", "chop"), index=self.df.index)

    # ------------------------------------------------------------------ combos
    def regime_switch(self, top):
        """Best strategy for trending bars + best for choppy bars, stitched together.
        Chosen on the earlier test slices, then judged on the FINAL slice only (never seen in the choice)."""
        if len(top) < 2:
            return None
        early = self.bounds[1:-1]; final = self.bounds[-1]
        regime = self.regime_labels().values
        best = {}
        for name in ("trend", "chop"):
            scored = []
            for r in top:
                folds = r["folds"][:-1]
                trades = pd.concat([f[1].trades for f in folds], ignore_index=True) if folds else pd.DataFrame()
                if len(trades) == 0:
                    continue
                lab = pd.Series(regime, index=self.df.index).reindex(pd.DatetimeIndex(trades.entry_time)).values
                p = trades.pnl.values[lab == name]
                if len(p) < max(5, self.cfg.min_trades // 3):
                    continue
                scored.append((p.mean() * np.sqrt(len(p)), r))
            if scored:
                best[name] = max(scored, key=lambda x: x[0])[1]
        if len(best) < 2 or best["trend"]["strategy"].template() == best["chop"]["strategy"].template():
            return None
        s_tr, s_ch = best["trend"]["folds"][-1][0], best["chop"]["folds"][-1][0]
        sig_tr = s_tr.signals(self.df, self.cfg, self.cache); sig_ch = s_ch.signals(self.df, self.cfg, self.cache)
        sig = np.where(regime == "trend", sig_tr, sig_ch).astype(np.int8)
        # exits/sizing follow whichever strategy fired; approximate with the trend one's rules on trend bars
        res_tr = engine.run(self.df, np.where(regime == "trend", sig, 0).astype(np.int8), s_tr.exit.rules(), s_tr.size, self.spec, self.cfg, *final)
        res_ch = engine.run(self.df, np.where(regime == "chop", sig, 0).astype(np.int8), s_ch.exit.rules(), s_ch.size, self.spec, self.cfg, *final)
        combined = combine_oos([res_tr, res_ch], self.cfg)
        return {"trend": s_tr, "chop": s_ch, "final_slice": combined}

    def ensemble_vote(self, top, m=2):
        """Trade only when at least m of the top 3 strategies agree on direction (like 'all three line up')."""
        if len(top) < 3:
            return None
        final = self.bounds[-1]
        strats = [r["folds"][-1][0] for r in top[:3]]
        sigs = np.stack([s.signals(self.df, self.cfg, self.cache) for s in strats])
        # a signal counts for 3 bars so near-simultaneous agreement is caught
        smear = sigs.copy()
        for k in (1, 2):
            smear = np.where(smear == 0, np.roll(sigs, k, axis=1), smear)
        votes = smear.sum(axis=0)
        sig = np.where(votes >= m, 1, np.where(votes <= -m, -1, 0)).astype(np.int8)
        sig = np.where(sigs[0] != 0, sig, 0).astype(np.int8)  # fire on the lead strategy's bar
        res = engine.run(self.df, sig, strats[0].exit.rules(), strats[0].size, self.spec, self.cfg, *final)
        lead = top[0]["folds"][-1][1]
        return {"strategies": strats, "min_votes": m, "final_slice": res, "lead_final_slice": lead}

    @staticmethod
    def diverse(results, n=3):
        """Top results but with DIFFERENT entry types, so voting/switching mixes real variety."""
        out, seen = [], set()
        for r in results:
            e = r["strategy"].entry.name
            if e in seen:
                continue
            seen.add(e); out.append(r)
            if len(out) == n:
                break
        return out

    @staticmethod
    def verdict(m, luck, robust_pf, robust_frac, cfg, n_templates):
        """PASS / WEAK / FAIL in plain words. Strict on purpose: we tried ~n_templates ideas, so the
        result must beat what a random search of that size would have stumbled on."""
        reasons = []
        nets = m.get("fold_nets") or []
        if nets and nets[-1] <= 0: reasons.append("lost money on the most recent slice")
        if m["breached"]: reasons.append("would have breached the max-loss rule")
        if m["trades"] < cfg.min_trades: reasons.append(f"too few unseen trades ({m['trades']} < {cfg.min_trades})")
        if m["profit_factor"] < 1.15: reasons.append(f"profit factor {m['profit_factor']:.2f} is too thin after fees")
        if m.get("folds_profitable", 0) < 0.75: reasons.append("not profitable in most time slices")
        if luck["margin"] < 0.0:
            reasons.append(f"a random search of {n_templates} ideas would find something this good "
                           f"(real ${m['net']:,.0f} vs random-search best ~${luck['expected_max']:,.0f})")
        elif luck["beats"] > 0.02:
            reasons.append(f"random entries matched it {luck['beats']:.0%} of the time")
        if robust_pf < 1.05 or robust_frac < 0.6: reasons.append("falls apart when settings are nudged (fragile)")
        if not reasons:
            return "PASS", "survived every check on unseen data - worth paper trading"
        fatal = m["breached"] or luck["margin"] < 0.0
        if len(reasons) <= 1 and not fatal and m["profit_factor"] > 1.0:
            return "WEAK", "; ".join(reasons)
        return "FAIL", "; ".join(reasons)

    def final_fit(self, strat):
        """Refit the winning template on ALL data - these are the settings that go live."""
        chosen, _ = self.greedy_fit(strat, 0, len(self.df))
        return chosen
