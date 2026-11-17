"""
bayes.py
========
Naive-Bayes net:  Outcome (Up/Down)  ->  Regime, Trend, RSI, MACD, Volume

Job: given today's labels, return p_up, mu, sigma, n.
It does NOT download data, build labels, or decide BUY/SELL.

INPUTS
------
fit(evidence, labels, fwd_ret)
    evidence : DataFrame (index = dates), text labels. Columns from
               features.make_evidence() + "Regime" from hmm.py
    labels   : Series "Up"/"Down"     (features.make_labels(df, edges))
    fwd_ret  : Series of 5-day return (features.make_forward_returns(df))
    Pass ONLY training dates (after embargo). The three are aligned by date.

predict(row)  -> dict(p_up, mu, sigma, n)      one day, evidence only
predict_df(evidence) -> DataFrame of the same 4 columns, one row per day

OUTPUTS (all on the same horizon as the labels, i.e. 5 days)
-------
p_up  : probability of "Up" (move above the stock's own threshold)
mu    : expected 5-day return
sigma : std of the 5-day return
n     : training days with exactly today's evidence (CSP uses n >= 20)
"""

from collections import Counter

import numpy as np
import pandas as pd

MIN_TRAIN_ROWS = 300
DEFAULT_FEATURES = ("Regime", "Trend", "RSI", "MACD", "Volume")


class BayesNet:
    def __init__(self, features=DEFAULT_FEATURES, alpha=1.0, p_clip=(0.05, 0.95)):
        self.features = list(features)
        self.alpha = alpha            # Laplace smoothing (add alpha to every count)
        self.p_clip = p_clip          # naive Bayes is over-confident, so limit p_up
        self.classes = ["Down", "Up"]
        self.fitted = False

    # ------------------------------------------------------------------
    # Training = counting
    # ------------------------------------------------------------------
    def fit(self, evidence: pd.DataFrame, labels: pd.Series, fwd_ret: pd.Series):
        missing = [f for f in self.features if f not in evidence.columns]
        if missing:
            raise ValueError(f"evidence is missing columns: {missing}")

        d = (evidence[self.features]
             .join(labels.rename("y"), how="inner")
             .join(fwd_ret.rename("fwd"), how="inner")
             .dropna())

        if len(d) < MIN_TRAIN_ROWS:
            raise ValueError(f"Insufficient data for BayesNet: {len(d)} rows "
                             f"(need {MIN_TRAIN_ROWS}).")
        if not set(d["y"].unique()).issubset(self.classes):
            raise ValueError("labels must be 'Up' or 'Down'.")
        if d["y"].nunique() < 2:
            raise ValueError("Only one class in the training labels.")

        N = len(d)
        a = self.alpha
        self.n_train = N
        self.base_rate = float((d["y"] == "Up").mean())

        # 1) prior  P(Up), P(Down)
        self.prior = {c: ((d["y"] == c).sum() + a) / (N + 2 * a)
                      for c in self.classes}

        # 2) tables  P(feature = value | class)
        self.values, self.cpt = {}, {}
        for f in self.features:
            vals = sorted(d[f].unique())
            k = len(vals)
            self.values[f] = vals
            self.cpt[f] = {}
            for c in self.classes:
                sub = d.loc[d["y"] == c, f]
                self.cpt[f][c] = {v: ((sub == v).sum() + a) / (len(sub) + a * k)
                                  for v in vals}

        # 3) 5-day return mean / std for each class
        self.stats = {}
        for c in self.classes:
            r = d.loc[d["y"] == c, "fwd"]
            sd = r.std(ddof=1)
            self.stats[c] = (float(r.mean()), float(0.0 if np.isnan(sd) else sd))

        # 4) exact-combination counts (this gives n)
        self._combo_counts = Counter(map(tuple, d[self.features].to_numpy()))

        self.fitted = True
        return self

    # ------------------------------------------------------------------
    # Inference = Bayes theorem (in log space)
    # ------------------------------------------------------------------
    def predict(self, row) -> dict:
        if not self.fitted:
            raise RuntimeError("Call fit() first.")
        for f in self.features:
            if f not in row or pd.isna(row[f]):
                raise ValueError(f"Missing evidence value for '{f}'.")

        logp = {c: np.log(self.prior[c]) for c in self.classes}
        for f in self.features:
            v = row[f]
            for c in self.classes:
                # value never seen in training -> uniform (no information)
                p = self.cpt[f][c].get(v, 1.0 / len(self.values[f]))
                logp[c] += np.log(p)

        m = max(logp.values())
        e = {c: np.exp(logp[c] - m) for c in self.classes}
        p_up = float(np.clip(e["Up"] / (e["Up"] + e["Down"]), *self.p_clip))

        # mix the Up-day and Down-day return stats using p_up
        (mu_d, sd_d), (mu_u, sd_u) = self.stats["Down"], self.stats["Up"]
        mu = p_up * mu_u + (1 - p_up) * mu_d
        var = (p_up * (sd_u ** 2 + mu_u ** 2)
               + (1 - p_up) * (sd_d ** 2 + mu_d ** 2) - mu ** 2)
        sigma = float(np.sqrt(max(var, 1e-12)))

        key = tuple(row[f] for f in self.features)
        n = int(self._combo_counts.get(key, 0))

        return {"p_up": p_up, "mu": float(mu), "sigma": sigma, "n": n}

    def predict_df(self, evidence: pd.DataFrame) -> pd.DataFrame:
        rows = [self.predict(r) for _, r in evidence[self.features].iterrows()]
        return pd.DataFrame(rows, index=evidence.index)

    # ------------------------------------------------------------------
    # Explainability (useful for the report / explainer)
    # ------------------------------------------------------------------
    def explain(self, row) -> pd.DataFrame:
        """Likelihood ratio P(value|Up) / P(value|Down) for each feature.
        > 1 pushes toward Up, < 1 pushes toward Down."""
        out = []
        for f in self.features:
            v = row[f]
            pu = self.cpt[f]["Up"].get(v, 1.0 / len(self.values[f]))
            pd_ = self.cpt[f]["Down"].get(v, 1.0 / len(self.values[f]))
            out.append({"feature": f, "value": v,
                        "P(v|Up)": round(pu, 3), "P(v|Down)": round(pd_, 3),
                        "ratio": round(pu / pd_, 2)})
        return pd.DataFrame(out)


# ----------------------------------------------------------------------
# Self-test:   python -m core.models.bayes     (run from the backend/ folder)
# Random synthetic prices, so there is no real signal to find;
# this only proves the code runs and the numbers are sane.
# ----------------------------------------------------------------------
if __name__ == "__main__":
    from core.features import (_synthetic_prices, fit_bins, make_evidence,
                               make_labels, make_forward_returns,
                               train_index_with_embargo)

    raw = _synthetic_prices(1200)
    tr_end = raw.index[int(len(raw) * 0.7) - 1]

    edges = fit_bins(raw[raw.index <= tr_end])          # train only
    ev = make_evidence(raw, edges)
    labels = make_labels(raw, edges)
    fwd = make_forward_returns(raw)

    # Regime comes from hmm.py; until it exists, test without it
    feats = ("Trend", "RSI", "MACD", "Volume")
    tr_idx = train_index_with_embargo(ev.index, tr_end)
    bn = BayesNet(features=feats).fit(ev.loc[tr_idx], labels, fwd)

    test_idx = ev.index[ev.index > tr_end].intersection(labels.index)
    out = bn.predict_df(ev.loc[test_idx])
    y = (labels.loc[test_idx] == "Up").astype(int)

    # --- checks ---
    assert out["p_up"].between(*bn.p_clip).all()
    assert (out["sigma"] > 0).all()
    assert (out["n"] >= 0).all()
    assert tr_idx[-1] < tr_end                      # embargo applied

    brier = ((out["p_up"] - y) ** 2).mean()
    base = ((bn.base_rate - y) ** 2).mean()
    acc = ((out["p_up"] > 0.5).astype(int) == y).mean()

    print(out.tail(3).round(4))
    print(f"\ntrain rows={bn.n_train}  train Up rate={bn.base_rate:.2%}")
    print(f"mean p_up (test)={out['p_up'].mean():.3f}   actual Up rate (test)={y.mean():.3f}")
    print(f"Brier model={brier:.4f}   baseline={base:.4f}   accuracy={acc:.3f}")

    bins = pd.cut(out["p_up"], [0, .15, .25, .35, .45, .55, 1.0])
    print("\ncalibration:")
    print(pd.DataFrame({"pred": out["p_up"], "actual": y})
          .groupby(bins, observed=True).agg(n=("pred", "size"),
                                            mean_pred=("pred", "mean"),
                                            actual_up=("actual", "mean")).round(3))

    print("\nwhy today's p_up:")
    print(bn.explain(ev.iloc[-1]))
    print("\nALL CHECKS PASSED")
