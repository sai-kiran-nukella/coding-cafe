import numpy as np


class Net:
    def __init__(self, sizes, seed=0, l2=1e-4):
        r = np.random.default_rng(seed)
        self.W = [r.standard_normal((a, b)) * np.sqrt(2 / a) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros((1, b)) for b in sizes[1:]]
        self.mW, self.vW = [np.zeros_like(w) for w in self.W], [np.zeros_like(w) for w in self.W]
        self.mb, self.vb = [np.zeros_like(b) for b in self.b], [np.zeros_like(b) for b in self.b]
        self.t, self.l2, self.losses = 0, l2, []

    def forward(self, X, keep=False):
        acts, a, last = [X], X, len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            a = z if i == last else np.maximum(0, z)
            acts.append(a)
        return acts if keep else a

    def step(self, X, Y, lr):
        acts = self.forward(X, True)
        err = acts[-1] - Y
        d = err / len(X)
        self.t += 1
        for i in reversed(range(len(self.W))):
            gW, gb = acts[i].T @ d + self.l2 * self.W[i], d.sum(0, keepdims=True)
            if i:
                d = (d @ self.W[i].T) * (acts[i] > 0)
            for P, G, m, v in ((self.W[i], gW, self.mW[i], self.vW[i]), (self.b[i], gb, self.mb[i], self.vb[i])):
                m *= 0.9; m += 0.1 * G
                v *= 0.999; v += 0.001 * G * G
                P -= lr * (m / (1 - 0.9 ** self.t)) / (np.sqrt(v / (1 - 0.999 ** self.t)) + 1e-8)
        return float(np.mean(np.sum(err ** 2, axis=1)) / 2)

    def epoch(self, X, Y, lr, batch=64, rng=None):
        rng, Y = rng or np.random.default_rng(0), Y.reshape(len(X), -1)
        idx = rng.permutation(len(X))
        ls = [self.step(X[j], Y[j], lr) for j in (idx[s:s + batch] for s in range(0, len(X), batch))]
        self.losses.append(float(np.mean(ls)))
        return self.losses[-1]

    def fit(self, X, Y, epochs=120, lr=0.01, batch=64, rng=None):
        rng, Y = rng or np.random.default_rng(0), Y.reshape(len(X), -1)
        for e in range(epochs):
            self.epoch(X, Y, lr * (0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * e / epochs))), batch, rng)


class Ensemble:
    def __init__(self, sizes, n=5, seed=0):
        self.seed, self.nets = seed, [Net(sizes, seed + i) for i in range(n)]

    def fit(self, X, Y, epochs=120, lr=0.01):
        r = np.random.default_rng(self.seed)
        for n in self.nets:
            j = r.integers(0, len(X), len(X))
            n.fit(X[j], Y[j], epochs, lr, rng=r)

    def predict(self, X):
        P = np.stack([n.forward(X) for n in self.nets])
        return P.mean(0), P.std(0)


class Linear:
    def __init__(self, X, Y):
        self.w = np.linalg.lstsq(np.c_[X, np.ones(len(X))], Y.reshape(len(X), -1), rcond=None)[0]

    def predict(self, X):
        return np.c_[X, np.ones(len(X))] @ self.w
