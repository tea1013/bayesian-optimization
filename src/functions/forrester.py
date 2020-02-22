import numpy as np
from .function import Function
from scipydirect import minimize

class Forrester(Function):

    def __init__(self):
        self.dim = 1
        self.bounds = [[0, 1]]
        res = minimize(self.value, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x'][0]
        self.y_opt = -self.value(self.x_opt)

    def value(self, x):
        res = (6 * x - 2)**2 * np.sin(12 * x - 4)
        return res
