import numpy as np
from .function import Function
from scipydirect import minimize

class Sphere1d(Function):

    def __init__(self):
        self.dim = 1
        self.bounds = [[-5, 5]]
        res = minimize(self.value, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x'][0]
        self.y_opt = -self.value(self.x_opt)

    def value(self, x):
        res = x**2
        return res

