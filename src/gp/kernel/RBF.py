from .kernel import Kernel
import GPy

class RBF(Kernel):
    def __init__(self, dim, lengthscale=1):
        self.dim = dim
        self.kernel = GPy.kern.RBF(dim, lengthscale=lengthscale)
