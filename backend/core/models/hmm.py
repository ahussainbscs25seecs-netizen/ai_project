# backend/core/models/hmm.py
# Discrete Hidden Markov Model written from scratch (numpy only).
#
#   hidden states : N market regimes (we use 3: bear / sideways / bull)
#   observations  : integer symbols 0..M-1 (the bucketed return/volatility from features_regime.py)
#
#   pi[i]   = P(first state is i)
#   A[i,j]  = P(next state is j | current state is i)
#   B[i,k]  = P(we see symbol k | state is i)

import numpy as np

EPS = 1e-12   # tiny number to avoid log(0) and division by zero


class DiscreteHMM:
    def __init__(self, n_states=3, n_symbols=9, seed=0):
        self.N = n_states                        # number of hidden states
        self.M = n_symbols                       # number of distinct observation symbols
        rng = np.random.default_rng(seed)        # seeded so runs are repeatable
        # random start: every row is a probability distribution (sums to 1)
        self.pi = rng.dirichlet(np.ones(self.N))
        self.A = rng.dirichlet(np.ones(self.N), size=self.N)
        self.B = rng.dirichlet(np.ones(self.M), size=self.N)

    
    # Day 3: forward algorithm (with scaling) and the forward FILTER
    
    def _forward(self, obs):
        """Returns alpha (T x N, each row sums to 1), scale factors c (T), log-likelihood.
        alpha[t] = P(state_t | obs_0..t)  -> this IS the filtered distribution."""
        T = len(obs)
        alpha = np.zeros((T, self.N))
        c = np.zeros(T)                                   # scale factor per step
        a = self.pi * self.B[:, obs[0]]                   # start: pi * P(first symbol | state)
        c[0] = a.sum() + EPS
        alpha[0] = a / c[0]                               # normalize so the row sums to 1
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * self.B[:, obs[t]]   # predict next state, then weigh by evidence
            c[t] = a.sum() + EPS
            alpha[t] = a / c[t]                           # scaling stops numbers underflowing to 0
        loglik = np.log(c).sum()                          # log P(obs) = sum of log scale factors
        return alpha, c, loglik

    def filter(self, obs):
        """P(state_t | observations up to t). Use THIS for live decisions (no look-ahead)."""
        alpha, _, _ = self._forward(np.asarray(obs))
        return alpha

    def loglik(self, obs):
        return self._forward(np.asarray(obs))[2]

   
    # Day 4: backward pass + smoothing + Baum-Welch (training)
    
    def _backward(self, obs, c):
        """Scaled backward pass, using the same scale factors c as the forward pass."""
        T = len(obs)
        beta = np.zeros((T, self.N))
        beta[-1] = 1.0                                    # at the last step there is nothing after it
        for t in range(T - 2, -1, -1):
            beta[t] = (self.A @ (self.B[:, obs[t + 1]] * beta[t + 1])) / c[t + 1]
        return beta

    def smooth(self, obs):
        """P(state_t | ALL observations). Uses the future, so use for analysis/plots only."""
        obs = np.asarray(obs)
        alpha, c, _ = self._forward(obs)
        beta = self._backward(obs, c)
        gamma = alpha * beta
        return gamma / (gamma.sum(axis=1, keepdims=True) + EPS)

    def _em_step(self, seqs):
        """One Baum-Welch iteration over a list of sequences. Returns total log-likelihood
        (measured BEFORE the update) so we can watch it improve."""
        pi_new = np.zeros(self.N)
        A_num = np.zeros((self.N, self.N))
        A_den = np.zeros(self.N)
        B_num = np.zeros((self.N, self.M))
        B_den = np.zeros(self.N)
        total_ll = 0.0
        for obs in seqs:
            alpha, c, ll = self._forward(obs)
            beta = self._backward(obs, c)
            total_ll += ll
            gamma = alpha * beta
            gamma /= gamma.sum(axis=1, keepdims=True) + EPS          # P(state_t | all data)
            pi_new += gamma[0]
            for t in range(len(obs) - 1):
                # xi[i,j] = P(state_t = i and state_t+1 = j | all data)
                xi = alpha[t][:, None] * self.A * (self.B[:, obs[t + 1]] * beta[t + 1])[None, :]
                xi /= xi.sum() + EPS
                A_num += xi
            A_den += gamma[:-1].sum(axis=0)
            for k in range(self.M):
                B_num[:, k] += gamma[obs == k].sum(axis=0)           # count of state i emitting symbol k
            B_den += gamma.sum(axis=0)
        # re-estimate; the 1e-6 smoothing keeps any probability from becoming exactly 0
        self.pi = (pi_new + 1e-6) / (pi_new + 1e-6).sum()
        self.A = (A_num + 1e-6) / ((A_num + 1e-6).sum(axis=1, keepdims=True))
        self.B = (B_num + 1e-6) / ((B_num + 1e-6).sum(axis=1, keepdims=True))
        return total_ll

    def fit(self, sequences, n_iter=100, tol=1e-4, n_restarts=5, seed=0):
        """Baum-Welch with random restarts. sequences = one array or a list of arrays of symbols.
        Keeps the best restart (highest log-likelihood). Returns that log-likelihood."""
        if isinstance(sequences, np.ndarray) and sequences.ndim == 1:
            sequences = [sequences]
        seqs = [np.asarray(s) for s in sequences]
        best = None
        for r in range(n_restarts):
            trial = DiscreteHMM(self.N, self.M, seed=seed + r)       # fresh random start each time
            prev = -np.inf
            history = []
            for _ in range(n_iter):
                ll = trial._em_step(seqs)
                history.append(ll)
                if abs(ll - prev) < tol:                             # converged
                    break
                prev = ll
            final = sum(trial.loglik(s) for s in seqs)
            if best is None or final > best[0]:
                best = (final, trial, history)
        self.pi, self.A, self.B = best[1].pi, best[1].A, best[1].B
        self.history = best[2]                                       # log-likelihood per iteration (for your test)
        return best[0]

    
    # Day 5: Viterbi (best path; analysis only) and state labelling
    
    def viterbi(self, obs):
        """Single most likely STATE PATH. Uses the whole sequence, so it looks ahead:
        never use it for live decisions."""
        obs = np.asarray(obs)
        T = len(obs)
        logA, logB = np.log(self.A + EPS), np.log(self.B + EPS)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)                       # back-pointers
        delta[0] = np.log(self.pi + EPS) + logB[:, obs[0]]
        for t in range(1, T):
            scores = delta[t - 1][:, None] + logA                    # scores[i,j] = best path ending i then moving to j
            psi[t] = scores.argmax(axis=0)
            delta[t] = scores.max(axis=0) + logB[:, obs[t]]
        path = np.zeros(T, dtype=int)
        path[-1] = delta[-1].argmax()
        for t in range(T - 2, -1, -1):
            path[t] = psi[t + 1][path[t + 1]]                        # walk the pointers backwards
        return path

    def reorder(self, order):
        """Permute states so new state k = old state order[k]. Fixes 'label switching'."""
        order = np.asarray(order)
        self.pi = self.pi[order]
        self.A = self.A[np.ix_(order, order)]
        self.B = self.B[order]

    def label_by_values(self, obs, values, names=("bear", "sideways", "bull")):
        """Sort states by their average `values` (e.g. daily return) so state 0 = lowest = 'bear'.
        values must be aligned with obs (same length). Returns the list of state names."""
        gamma = self.smooth(obs)
        means = (gamma * np.asarray(values)[:, None]).sum(axis=0) / (gamma.sum(axis=0) + EPS)
        self.reorder(np.argsort(means))                              # lowest average return first
        return list(names)[: self.N]

    
    # saving / loading for artifacts/hmm_params.json
    
        return {"pi": self.pi.tolist(), "A": self.A.tolist(), "B": self.B.tolist()}

    @classmethod
    def from_dict(cls, d):
        pi = np.array(d["pi"])
        m = cls(n_states=len(pi), n_symbols=np.array(d["B"]).shape[1])
        m.pi, m.A, m.B = pi, np.array(d["A"]), np.array(d["B"])
        return m