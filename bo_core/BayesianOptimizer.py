import numpy as np
from scipy.optimize import minimize as spmin


class BO(object):
    """Bayesian optimization based on expected improvment"""

    def __init__(self, arg):
