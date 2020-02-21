from abc import ABCMeta, abstractmethod
import numpy as np
from scipydirect import minimize

class Function(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def value(self, x):
        pass
