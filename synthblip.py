
from joblib import Parallel, delayed
import numpy as np
from pcr import PCR
from sklearn.preprocessing import OneHotEncoder

def donor_weights(i, Zobs, A, T0, K):
    N, T = Zobs.shape
    X = Zobs[:, :T0].T
    y = Zobs[i, :T0]
    # calculate mean counterfactual outcome for each period and each potential treatment
    Beta = np.zeros((K, T - T0, N))
    for k in np.arange(K):
        for t in np.arange(T0, T):
            # find units that received treatment k in period t, as their first treatment
            # in this post-treatment period
            donors = (A[:, t] == k) & np.all(A[:, :t] == 0, axis=1)
            est = PCR().fit(X[:, donors], y) # find coefficients to donor units using PCR
            Beta[k, t - T0, donors] = est.coef_ # store the unit weights in the matrix Beta
    return Beta


class SyntheticBlip:

    def fit_weights(self, Zobs, A, T0, K):
        N, T = Zobs.shape
        # The matrix Beta will be of shape (n, K, T - T0, n). Each entry (i, k, t, :)
        # will contain the donor weights with target unit i, among donors which received
        # treatment k, as their first treatment and at period t.
        self.Beta_ = np.array(Parallel(n_jobs=-1, verbose=3)(delayed(donor_weights)(i, Zobs, A, T0, K)
                                                             for i in range(N)))
        return self
    
    
    def fit_blips(self, Zobs, A, T0, K, L):
        if not hasattr(self, 'Beta_'):
            raise AttributeError("You must first call fit_weights to construct donor weights")
        Beta = self.Beta_
        N, T = Zobs.shape

        base = np.nan * np.zeros((N, T)) # baseline response for each unit and period
        blip = np.nan * np.zeros((N, T, L + 1, K)) # blip effects for each unit i, period t, lag ell and action k

        base_var = np.nan * np.zeros((N, T)) # TRIAL
        blip_var = np.nan * np.zeros((N, T, L + 1, K)) # TRIAL

        # We start producing counterfactuals starting from period T0 + L, as for earlier periods
        # we don't have enough "lags"
        for t in np.arange(T0 + L, T):

            # for each post intervention period and unit, estimate the mean baseline response
            base[:, t] = Beta[:, 0, t - T0, :] @ Zobs[:, t] # sum_{j\in I_t^0} \beta_j^{i, I_t^0} Y_{j, t}
            
            ### DRAFT ATTEMPT: Variance of estimate calculation
            donors = np.abs(Beta[:, 0, t - T0, :]) > 0
            var_base = np.mean(np.tile((Zobs[:, t] - base[:, t])**2, (N, 1)), axis=1, where=donors)
            base_var[:, t] = var_base * np.sum(Beta[:, 0, t - T0, :]**2, axis=1)
            ###################################################

            for ell in range(L + 1):  # we construct the blip effects for lag ell
                for k in range(K):   # and for each action k, i.e. gamma_{j, t, t-ell}(k)

                    # we build the "obesrved" blip effects; we will actually for a moment pretend that
                    # every unit is in the I_{t-ell}^k, but then all the "wrong" entries will be corrected
                    # by taking the inner product with the donor entries and since donor weights will only
                    # be supported on elements in I_{t-ell}^k. This is more convenient for coding purposes
                    observed_blips = Zobs[:, t] - base[:, t] # we subtract the baseline response Y_{j, t} - b_{j, t}

                    for ellp in range(ell): # for each smaller lag, i.e. period t - ellp, with ellp < ell
                        # we subtract the blip effect of the treatment that each unit received at period t - ellp
                        # this subtracts gamma_{j, t, t - ellp}(A_{j, t - ell})
                        ohe = OneHotEncoder(sparse=False, categories=[np.arange(K)])
                        lagAohe = ohe.fit_transform(A[:, [t - ellp]]) # this is the treatment at t-ellp
                        observed_blips -= np.sum(blip[:, t, ellp, :] * lagAohe, axis=1)

                    # now that we have constructed the observed blip effects for all donor units
                    # we can impute the blip effects for all units, using the donor weights
                    # we will in fact even replace the blip effects of the donor units, with their
                    # corresponding averages, which will induce variance reduction
                    blip[:, t, ell, k] = Beta[:, k, t - ell - T0, :] @ observed_blips

                    ### DRAFT ATTEMPT: Variance of estimate calculation
                    donors = np.abs(Beta[:, k, t - ell - T0, :]) > 0
                    donor_wsq = Beta[:, k, t - ell - T0, :]**2
                    # variance of current blip effect
                    var_blips = np.mean(np.tile((observed_blips - blip[:, t, ell, k])**2, (N, 1)), axis=1, where=donors)
                    blip_var[:, t, ell, k] = var_blips * np.sum(donor_wsq, axis=1)
                    # influence from variance of future blip effect estimation and base response estimation
                    # TODO. This influence part is wrong. The different future blip and base quantities
                    # are not independent variables with each other across units; this formula treats them as such. 
                    blip_var[:, t, ell, k] += donor_wsq @ base_var[:, t]
                    for ellp in range(ell):
                        ohe = OneHotEncoder(sparse=False, categories=[np.arange(K)])
                        lagAohe = ohe.fit_transform(A[:, [t - ellp]]) # this is the treatment at t-ellp
                        blip_var[:, t, ell, k] += donor_wsq @ np.sum(blip_var[:, t, ellp, :] * lagAohe, axis=1)
                    ###################################################
        
        self.base_ = base
        self.blip_ = blip
        self.base_var_ = base_var
        self.blip_var_ = blip_var
        self.T0_ = T0
        self.T_ = T
        self.L_ = L
        self.K_ = K
        return self
    
    def fit(self, Zobs, A, T0, K, L):
        self.fit_weights(Zobs, A, T0, K)
        self.fit_blips(Zobs, A, T0, K, L)
        return self
    

    def predict_counterfactual(self, unit, Atarget):
        T, T0, L, K = self.T_, self.T0_, self.L_, self.K_
        base, blip = self.base_, self.blip_
        base_var, blip_var = self.base_var_, self.blip_var_

        pred = np.zeros(T - T0) * np.nan
        pred_var = np.zeros(T - T0) * np.nan
        t1 = T0 + L
        for t in np.arange(t1, T):
            pred[t - T0] = base[unit, t]
            pred_var[t - T0] = base_var[unit, t]
            for ell in range(L + 1): # for each lag period
                for k in range(K):
                    # we add the lag blip effect of that lag treatment
                    pred[t - T0] += blip[unit, t, ell, k] * (Atarget[t - T0 - ell] == k)
                    pred_var[t - T0] += blip_var[unit, t, ell, k] * (Atarget[t - T0 - ell] == k)

        return pred, pred_var