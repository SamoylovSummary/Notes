import numpy as np

from toy_net.common import *

#---------------------------------------------------------------------------------------------------

"""
Проверяем, что модуль переимпортировался в IPython notebook.
"""
print('toy_net.optimizers reload')

#---------------------------------------------------------------------------------------------------


class Optimizer(object):
    """
    Optimizer - алгоритм вычисления дельты весов
    """

    def calculate_dw(self, speed, gw):
        """
        Вычисление дельты весов.
        Args:
            speed: скорость обучения.
            gw: dL/dw - градиент функции потерь по параметрам слоя (например, весам).
                Не может быть None (внешний код должен сам это проверять).
        Returns:
            dw: дельта весов - прибавляется к весам слоя.
        """
        raise NotImplementedError()

#---------------------------------------------------------------------------------------------------


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer
    """

    def calculate_dw(self, speed, gw):
        return -speed * gw

    def __str__(self):
        return 'SGD'

#---------------------------------------------------------------------------------------------------


class Adam(Optimizer):
    """
    Stochastic Gradient Descent optimizer
    """

    def __init__(self, beta1=0.9, beta2=0.999):
        self.beta1 = beta1
        self.beta2 = beta2
        self.m1 = None
        self.m2 = None
        self.step = 0

    def calculate_dw(self, speed, gw):
        self.step += 1
        if self.m1 is None:
            self.m1 = np.zeros_like(gw)
        self.m1 = self.beta1 * self.m1 + (1 - self.beta1) * gw
        m1_norm = self.m1 / (1 - self.beta1 ** self.step)
        if self.m2 is None:
            self.m2 = np.zeros_like(gw)
        self.m2 = self.beta2 * self.m2 + (1 - self.beta2) * np.square(gw)
        m2_norm = self.m2 / (1 - self.beta2 ** self.step)
        epsilon = 1e-5
        return -speed * m1_norm / np.sqrt(m2_norm + epsilon)

    def __str__(self):
        return '\n'.join([
            'Adam',
            'beta1:', indent(str(self.beta1)),
            'beta2:', indent(str(self.beta2)),
            'm1:', indent(str(self.m1)),
            'm2:', indent(str(self.m2)),
            'step:', indent(str(self.step))])

#---------------------------------------------------------------------------------------------------
