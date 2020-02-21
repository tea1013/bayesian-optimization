import sys
sys.path.append('../')
import matplotlib.pyplot as plt

import numpy as np
from bo.PI import PI
from bo.EI import EI
from bo.GPUCB import GPUCB
from bo.MES import MES
from gp.kernel.RBF import RBF
from gp.gp import GPRegression
from functions.six_hump_camel import SixHumpCamel

f = SixHumpCamel()
X = []
Y = []
for _ in range(10):
    x = []
    x.append(np.random.uniform(f.bounds[0][0], f.bounds[0][1], 1)[0])
    x.append(np.random.uniform(f.bounds[1][0], f.bounds[1][1], 1)[0])
    y = -f.value(x)
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

for i in range(100):
    kernel = RBF(f.dim, lengthscale=0.3)
    gp_model = GPRegression(X.reshape(-1, f.dim), Y[:, None], kernel.kernel)
    gp_model.optimize()

    mes = MES(gp_model, f, np.max(Y))

    next_x = mes.next_input()
    next_y = -f.value(next_x)
    X = np.append(X, next_x)
    Y = np.append(Y, next_y)
    print('regret = {}'.format(f.y_opt - np.max(Y)))

