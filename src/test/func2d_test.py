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
from plot.regret import plot_regret_err
from save.experiment import save

f = SixHumpCamel()

regrets = []
n_init = 5
n_average = 3
n_iteration = 30
lengthscale = 1
for _ in range(n_average):
    X = []
    Y = []
    for _ in range(n_init):
        x = []
        x.append(np.random.uniform(f.bounds[0][0], f.bounds[0][1], 1)[0])
        x.append(np.random.uniform(f.bounds[1][0], f.bounds[1][1], 1)[0])
        y = -f.value(x)
        X.append(x)
        Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    regret = []
    for i in range(n_iteration):
        kernel = RBF(f.dim, lengthscale=lengthscale)
        gp_model = GPRegression(X.reshape(-1, f.dim), Y[:, None], kernel.kernel)
        if i % 5 == 0:
            gp_model.optimize_restarts()
            lengthscale = gp_model['.*rbf.lengthscale']

        mes = MES(gp_model, f, np.max(Y))

        next_x = mes.next_input()
        next_y = -f.value(next_x)
        X = np.append(X, next_x)
        Y = np.append(Y, next_y)
        print('regret = {}'.format(f.y_opt - np.max(Y)))
        regret.append(f.y_opt - np.max(Y))

    regrets.append(regret)

regrets = np.array(regrets)
regret_mean = np.mean(regrets, axis=0)
regret_std = np.std(regrets, axis=0)
regret_err = regret_std / np.sqrt(5 - 1)

info = {'dir_path': './csv', 'file_name': 'func2d_test.csv', 'n_iteration': n_iteration, 'regret': {'regret_mean': regret_mean, 'regret_err': regret_err}}
save(info)
plot_regret_err(regret_mean, regret_err, 'r', 'o', dir_path='./figure', file_name='func2d_test.pdf', title='func2d_test')
