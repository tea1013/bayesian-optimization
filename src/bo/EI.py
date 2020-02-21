import numpy as np
from bo.bo import BO
from scipy.stats import norm

class EI(BO):

    def __init__(self, gp_model, f, y_max, xi=0.01):
        super().__init__(gp_model, f)
        self.f = f
        self.y_max = y_max
        self.xi = xi

    def acquire(self, x):
        x = x.reshape(-1, self.f.dim)
        pred_mean, pred_var = self.gp_model.predict_noiseless(x)
        pred_std = np.sqrt(pred_var)
        Z = (pred_mean - self.y_max - self.xi) / pred_std
        res = ((Z * pred_std) * norm.cdf(Z) + pred_std * norm.pdf(Z)).ravel()
        return res

