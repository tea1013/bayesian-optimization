import numpy as np
from .function import Function
from scipydirect import minimize

class SixHumpCamel(Function):

    def __init__(self):
        self.dim =2
        self.bounds = [[-3, 3], [-2, 2]]
        res = minimize(self.value, self.bounds, maxf=self.dim * 1000, algmethod=1)
        self.x_opt = res['x']
        self.y_opt = -self.value(self.x_opt)

    def value(self, x):
        x1 = x[0]
        x2 = x[1]
        res = (4 - 2.1 * x1**2 + x1**4 / 3) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2

        return res

