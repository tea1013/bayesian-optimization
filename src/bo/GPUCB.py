import numpy as np
from bo.bo import BO
from scipy.stats import norm

class GPUCB(BO):

    def __init__(self, gp_model, f, t):
        super().__init__(gp_model, f)
        self.f = f
        self.t = t

    def acquire(self, x):
        x = x.reshape(-1, self.f.dim)
        pred_mean, pred_var = self.gp_model.predict_noiseless(x)
        pred_std = np.sqrt(pred_var)
        res = pred_mean + np.sqrt(self.beta_func()) * pred_std
        return res


    def beta_func(self):
        beta = np.log((self.t**2) + 0.1)
        return beta
