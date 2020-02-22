import numpy as np
import matplotlib.pyplot as plt
import os

plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams["font.size"] = 15 # 全体のフォントサイズが変更されます。
plt.rcParams['xtick.labelsize'] = 14 # 軸だけ変更されます。
plt.rcParams['ytick.labelsize'] = 14 # 軸だけ変更されます
plt.rcParams['xtick.direction'] = 'in' # x axis in
plt.rcParams['ytick.direction'] = 'in' # y axis in
plt.rcParams['axes.linewidth'] = 1.0 # axis line width
plt.rcParams['axes.grid'] = True # make grid

def plot_regret(regret, color, marker, dir_path, file_name, title=''):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    _, ax = plt.subplots()
    ax.plot(range(len(regret)), regret, color=color, marker=marker)
    ax.set_title(title)
    plt.savefig(dir_path + '/' + file_name)

def plot_regret_err(regret_mean, regret_err, color, marker, dir_path, file_name, title=''):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    _, ax = plt.subplots()
    ax.plot(range(len(regret_mean)), regret_mean, color=color)
    ax.fill_between(range(len(regret_mean)), regret_mean + regret_err, regret_mean - regret_err, color=color, alpha=0.3)
    ax.set_title(title)
    plt.savefig(dir_path + '/' + file_name)

def plot_regrets_errs(settings, dir_path, file_name, title=''):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    _, ax = plt.subplots()
    for setting in settings:
        regret_mean = setting['regret_mean']
        regret_err = setting['regret_err']
        color = setting['color']
        ax.plot(range(len(regret_mean)), regret_mean, color=color)
        ax.fill_between(range(len(regret_mean)), regret_mean + regret_err, regret_mean - regret_err, color=color, alpha=0.3)
    plt.savefig(dir_path + '/' + file_name)
