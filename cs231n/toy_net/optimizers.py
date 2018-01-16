import numpy as np
import math

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

    def __init__(self, beta1=0.9, beta2=0.9):
        self.beta1 = beta1
        self.beta2 = beta2
        self.m1 = None
        self.m2 = None
        self.step = 0

    def calculate_dw(self, speed, gw):
        self.step += 1
        m1 = np.zeros_like(gw)
        m1 = self.beta1 * m1 + (1 - self.beta1) * gw
        m1 /= 1 - self.beta1 ** self.step
        m2 = np.zeros_like(gw)
        m2 = self.beta2 * m2 + (1 - self.beta2) * np.square(gw)
        m2 /= 1 - self.beta2 ** self.step
        epsilon = 1e-5
        return -speed * m1 / np.sqrt(m2 + epsilon)

    def __str__(self):
        return '\n'.join([
            'Adam',
            'beta1:', indent(str(self.beta1)),
            'beta2:', indent(str(self.beta2)),
            'm1:', indent(str(self.m1)),
            'm2:', indent(str(self.m2)),
            'step:', indent(str(self.step))])

#---------------------------------------------------------------------------------------------------
