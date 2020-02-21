import numpy as np
from bo.bo import BO
from scipy.stats import norm

class PI(BO):

    def __init__(self, gp_model, f, y_max, xi=0.01):
        super().__init__(gp_model, f)
        self.y_max = y_max
        self.xi = xi

    def acquire(self, x):
        if x[0] in self.gp_model.X:
            return 0

        x = x[:, None]
        pred_mean, pred_var = self.gp_model.predict_noiseless(x)
        pred_std = np.sqrt(pred_var)
        Z = (pred_mean - self.y_max - self.xi) / pred_std
        res = norm.cdf(Z, loc=0, scale=1).ravel()[0]
        return res
