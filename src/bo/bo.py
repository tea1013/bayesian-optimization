from abc import ABCMeta, abstractmethod
import numpy as np
import GPy
from scipydirect import minimize

class BO(object):
    __metaclass__ = ABCMeta

    def __init__(self, gp_model, f):
        self.gp_model = gp_model
        self.f = f

    @abstractmethod
    def acquire(self, x):
        pass

    def acquire_minus(self, x):
        res = -1 * self.acquire(x)
        return res

    def next_input(self):
        res = minimize(self.acquire_minus, self.f.bounds, maxf=self.f.dim * 1000, algmethod=1)
        return res['x']
