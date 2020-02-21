import sys
sys.path.append('../')
import matplotlib.pyplot as plt

import numpy as np
from bo.PI import PI
from bo.EI import EI
from gp.kernel.RBF import RBF
from gp.gp import GPRegression
from functions.forrester import Forrester
from functions.sphere1d import Sphere1d

f = Sphere1d()
f = Forrester()
X = np.random.uniform(0, 1, 1)
Y = -f.value(X)

for i in range(100):
    kernel = RBF(f.dim, lengthscale=1)
    gp_model = GPRegression(X.reshape(-1, f.dim), Y[:, None], kernel.kernel)
    gp_model.optimize()

    pi = PI(gp_model, f, np.max(Y))
    next_x = pi.next_input()
    next_y = -f.value(next_x)
    X = np.append(X, next_x)
    Y = np.append(Y, next_y)
    print('regret = {}'.format(f.y_opt - np.max(Y)))
