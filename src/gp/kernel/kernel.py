from abc import ABCMeta, abstractmethod
import numpy as np
import GPy

class Kernel(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def cov_matrix(self):
        pass
