from abc import ABCMeta, abstractmethod
import numpy as np
import GPy

class GPRegression(GPy.models.GPRegression):

    def __init__(self, X, Y, kernel, noise_var=1e-6, normalizer=False, opt_params=False, num_restarts=10):
        super().__init__(X=X, Y=Y, kernel=kernel, noise_var=noise_var, normalizer=normalizer)
        self.noise_var = noise_var
