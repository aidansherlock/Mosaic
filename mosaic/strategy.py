"""A Strategy = one entry + 0..N filters + one exit + one sizer, assembled from pieces."""
import json
import numpy as np
from . import pieces as P


class Strategy:
    def __init__(self, entry, filters, exit_, size):
        self.entry, self.filters, self.exit, self.size = entry, list(filters), exit_, size

    def signals(self, df, cfg, cache=None):
        key = self.entry.label()
        if cache is not None and key in cache:
            sig = cache[key]
        else:
            sig = self.entry.signals(df, cfg)
            if cache is not None:
                cache[key] = sig
        for f in self.filters:
            sig = f.apply(sig, df, cfg)
        return sig

    def pieces(self):
        return [self.entry, *self.filters, self.exit, self.size]

    def label(self):
        return " + ".join(p.label() for p in self.pieces())

    def template(self):
        return "|".join([self.entry.name, *(f.name for f in self.filters), self.exit.name, self.size.name])

    def spec(self):
        return {"entry": self.entry.spec(), "filters": [f.spec() for f in self.filters],
                "exit": self.exit.spec(), "size": self.size.spec()}

    @classmethod
    def from_spec(cls, s):
        return cls(P.from_spec(s["entry"]), [P.from_spec(f) for f in s["filters"]],
                   P.from_spec(s["exit"]), P.from_spec(s["size"]))

    def with_(self, **kw):
        d = dict(entry=self.entry, filters=self.filters, exit_=self.exit, size=self.size)
        d.update(kw)
        return Strategy(**d)

    def save(self, path, extra=None):
        with open(path, "w") as f:
            json.dump({"strategy": self.spec(), **(extra or {})}, f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        return cls.from_spec(d["strategy"]), d
