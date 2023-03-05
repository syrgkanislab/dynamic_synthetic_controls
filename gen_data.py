
import numpy as np

class SynthDGP:

    def __init__(self, N, T, K, L, T0):
        self.N = N # n units
        self.T = T # n overall time periods
        self.K = K # n actions (for now has to be true)
        self.L = L # number of lags that impact current outcome
        self.T0 = T0 # pre-treatment period length

    def init_instance(self, random_seed=None):
        np.random.seed(random_seed)
        self.u_ = np.random.normal(0, 1, size=(self.N, 1)) # unit latent factors
        self.v_ = np.random.normal(0, 1, size=(self.K, self.L + 1, self.T, 1)) # (action, time) latent factors
        self.W_ = np.einsum('ij,jltk->iltk', self.u_, self.v_.T) # true mean potential blips for each unit and period and lag
        return self

    def sample_data(self, random_seed=None):
        np.random.seed(random_seed)
        W = self.W_
        Z = W + np.random.normal(0, .4, size=(self.N, self.T, self.L + 1, self.K)) # random potential blips for each unit, period and lag

        t0 = np.random.choice(np.arange(self.T0, self.T + 1), size=self.N, replace=True) # choose random rollout time after T0
        time = np.tile(np.arange(self.T), (self.N, 1)) # helper matrix
        A = np.random.choice(np.arange(1, self.K), size=(self.N, self.T), replace=True)
        A = (time >= np.tile(t0.reshape(-1, 1), (1, self.T))) * A # set treatment to 1 after rollout

        Zobs = np.zeros(Z.shape[:2]) # building observed noisy outcomes
        Wobs = np.zeros(W.shape[:2]) # building observed true mean outcomes
        for ell in range(self.L + 1): # for each lag period
            Aell = np.roll(A, ell) # we find the lag treatment for each period
            Aell[:, :ell] = 0
            for k in range(self.K):
                Zobs += Z[:, :, ell, k] * (Aell == k) # we add the lag blip effect of that lag treatment
                Wobs += W[:, :, ell, k] * (Aell == k) # we add the lag blip effect of that lag treatment

        return W, Z, Wobs, Zobs, A

    def get_counterfactual(self, unit, Atarget):
        '''Counterfactual post-T0 mean potential outcome trajectories for unit `unit`;
        `Atarget` contains target treatments in periods {T0, ..., T}.'''
        T, L, K, T0 = self.T, self.L, self.K, self.T0
        W = self.W_

        Wtarget = np.zeros(T - T0) * np.nan
        t1 = T0 + L
        for t in np.arange(t1, T):
            Wtarget[t - T0] = 0
            for ell in range(L + 1): # for each lag period
                for k in range(K):
                    # we add the lag blip effect of that lag treatment
                    Wtarget[t - T0] += W[unit, t, ell, k] * (Atarget[t - T0 - ell] == k)
        return Wtarget
    