import numpy as np
from bo.bo import BO
from scipy.stats import norm

class MES(BO):

    def __init__(self, gp_model, f, y_max, n_sampling=10, n_represent=10000):
        super().__init__(gp_model, f)
        self.f = f
        self.y_max = y_max
        self.n_sampling = n_sampling
        X_represent = []
        for _ in range(n_represent):
            x = []
            for d in range(f.dim):
                x.append(np.random.uniform(f.bounds[d][0], f.bounds[d][1], 1)[0])
            X_represent.append(x)

        self.represent_mean, self.represent_var = gp_model.predict_noiseless(np.array(X_represent))
        self.represent_std = np.sqrt(self.represent_var)
        self.max_samples = self.sampling_gumbel()

    def acquire(self, x):
        if x in self.gp_model.X:
            # print('x is already acquired')
            return 0

        x = x.reshape(-1, self.f.dim)
        pred_mean, pred_var =  self.gp_model.predict(x)
        pred_std = np.sqrt(pred_var)

        normalized_max = (self.max_samples - np.c_[pred_mean]) / np.c_[pred_std]
        pdf = norm.pdf(normalized_max)
        cdf = norm.cdf(normalized_max)
        res = (normalized_max * pdf) / (2 * cdf) - np.log2(cdf)
        res = np.mean(res, 1)
        return res

    def find_r(self, val, R, Y, threshould):
        current_r_pos = np.argmin(np.abs(val - R))
        if (np.abs(val - R[current_r_pos])) < threshould:
            return Y[current_r_pos]

        if R[current_r_pos] > val:
            left = Y[current_r_pos - 1]
            right = Y[current_r_pos]
        else:
            left = Y[current_r_pos]
            right = Y[current_r_pos + 1]

        for _ in range(10000):
            mid = (left + right) / 2.
            mid_r = self.approx_gumbel_by_cdf(mid)

            if (np.abs(val - mid_r)) < threshould:
                return mid

            if mid_r > val:
                right = mid
            else:
                left = mid

        return mid

    # ガンベル分布をcdfで近似する
    def approx_gumbel_by_cdf(self, y):
        return np.prod(norm.cdf((y - np.c_[self.represent_mean]) / np.c_[self.represent_std]), axis=0)

    def sampling_gumbel(self):
        left = self.y_max
        if self.approx_gumbel_by_cdf(left) < 0.25:
            right = np.max(self.represent_mean + 5 * self.represent_std)
            while (self.approx_gumbel_by_cdf(right) < 0.75):
                right = 2 * right - left

            Y = np.c_[np.linspace(left, right, 100)].T
            R = self.approx_gumbel_by_cdf(Y)
            Y = np.ravel(Y)

            # r = 0.25, 0.5, 0.75となるyを探索
            y1 = self.find_r(0.25, R, Y, 0.01)
            med = self.find_r(0.5, R, Y, 0.01)
            y2 = self.find_r(0.75, R, Y, 0.01)

            b = (y1 - y2) / (np.log(np.log(4/3)) - np.log(np.log(4)))
            a = med + b * np.log(np.log(2))

            max_samples = np.array(np.random.gumbel(a, b, self.n_sampling))

            # 観測最大点よりも小さいサンプル点は観測最大点に補正しておく
            max_samples[max_samples < left + 5 * np.sqrt(self.gp_model.noise_var)] = left + 5 * np.sqrt(self.gp_model.noise_var)

        else:
            # 最大値は観測最大の値より大きいため逆が成り立つ場合は最大値にノイズを乗せたものを返すようにする
            max_samples = (left + 5 * np.sqrt(self.gp_model.noise_var)) * np.ones(self.n_sampling)

        return max_samples
