import numpy as np
import math

from toy_net.common import *

#---------------------------------------------------------------------------------------------------

"""
Проверяем, что модуль переимпортировался в IPython notebook.
"""
print('toy_net.functions reload')

#---------------------------------------------------------------------------------------------------


class Function(object):
    """
    Интерфейс слоя.
    """

    def reset_batch_data(self):
        """
        Очистить временные данные обработки минипакета.
        """
        pass

    def calculate_y(self, x):
        """
        Прямой проход.
        Args:
            x: сэмплы мини-пакета. Должен быть numpy-массивом, первое измерение (x.shape[0]) - 
                количество сэмплов, остальные измерения - признаки сэмплов.
        Returns:
            y: массив результатов применения функции к каждому сэмплу. y.shape[0] == x.shape[0].
        """
        raise NotImplementedError()
    
    def calculate_gx(self, x, y, gy):
        """
        Обратный проход.
        У одного и того же слоя backward может быть вызван несколько раз.
        Args:
            x: вход слоя в последнем calculate_y
            y: выход слоя в последнем calculate_y
            gy: dL/dy - градиент функции потерь по y, возвращаемых из calculate_y
        Returns:
            gx: dL/dx - градиент функции потерь по x, переданных в calculate_y
        """
        return None

    def calculate_gw(self, x, y, gy):
        """
        Обратный проход.
        У одного и того же слоя backward может быть вызван несколько раз.
        Args:
            x: вход слоя в последнем calculate_y
            y: выход слоя в последнем calculate_y
            gy: dL/dy - градиент функции потерь по y, возвращаемых из calculate_y
        Returns:
            gw: dL/dw - градиент функции потерь по параметрам слоя (например, весам). Этот градиент
                накапливается во внешней обёртке Layer, а потом передаётся в update.
                То есть параметры слоя изменяются не в backward, а в update.
                Может быть None.
        """
        return None
        
    def update(self, dw):
        """
        Отложенное обновление параметров (весов) слоя.
        Args:
            dw: дельта весов. Нужно прибавить к весам функции.
        """
        pass

#---------------------------------------------------------------------------------------------------


class FixedLinear(Function):
    
    def __init__(self, a, b):
        self.a = a
        self.b = b
        
    def calculate_y(self, x):
        return x * self.a + self.b
        
    def calculate_gx(self, x, y, gy):
        return gy / self.a

    def __str__(self):
        return '\n'.join([
            'a:', indent(str(self.a)),
            'b:', indent(str(self.b))])

#---------------------------------------------------------------------------------------------------


class Matrix(Function):
    
    def __init__(self, x_len, y_len, sigma=None, reg2=None, reg2_constrain=None, reg3=None):
        if sigma is None:
            sigma = 1 / math.sqrt(x_len)
        self.w = np.random.randn(x_len, y_len) * sigma
        self.reg2 = reg2
        self.reg2_constrain = reg2_constrain
        if reg2_constrain is not None:
            self.reg2_constrain *= reg2_constrain * sigma
        self.reg3 = reg3

    def calculate_y(self, x):
        return np.dot(x, self.w)

    def calculate_gx(self, x, y, gy):
        return np.dot(gy, self.w.transpose())

    def calculate_gw(self, x, y, gy):
        gw = np.dot(x.transpose(), gy)
        # Регуляризация
        # L2
        if self.reg2 is not None:
            # Ограничение области действия L2
            if self.reg2_constrain is not None:
                reg2_mask = np.std(self.w, axis=0, keepdims=True) > self.reg2_constrain
            else:
                reg2_mask = 1
            gw += self.w * reg2_mask * self.reg2 * 2
        # L3 для защиты от взрыва
        if self.reg3 is not None:
            gw += self.w * abs(self.w) * self.reg3 * 3
        return gw

    def update(self, dw):
        self.w = self.w + dw

    def trace_statistics(self, x):
        s = 'Matrix statistics:\n'
        s += indent(self._trace_single_statistics('column', self.w)) + '\n'
        s += indent(self._trace_single_statistics('y     ', np.dot(x, self.w))) + '\n'
        return s

    @staticmethod
    def _trace_single_statistics(name, x):
        mean = np.mean(x, axis=0)
        std = np.std(x, axis=0)
        std_log = np.log10(std)
        return '%s: mean: [%.4f, %.4f], std: [%.4f, %.4f], log10: [%.2f, %.2f]' \
             % (name, mean.min(), mean.max(), std.min(), std.max(), math.log10(std.min()), math.log10(std.max()))

    def __str__(self):
        return '\n'.join([
            'w:', indent(str(self.w)),
            'reg2:', indent(str(self.reg2)),
            'reg2_constrain:', indent(str(self.reg2_constrain)),
            'reg3:', indent(str(self.reg3))
        ])

#---------------------------------------------------------------------------------------------------


class Bias(Function):

    def __init__(self, x_len, reg2=None, reg3=None):
        self.b = np.zeros(x_len)
        self.reg2 = reg2
        self.reg3 = reg3
        
    def calculate_y(self, x):
        return x + self.b
        
    def calculate_gx(self, x, y, gy):
        return gy

    def calculate_gw(self, x, y, gy):
        gw = np.sum(gy, axis=0)
        # Регуляризация
        # L2
        if self.reg2 is not None:
            gw += self.b * self.reg2 * 2
        # L3 для защиты от взрыва
        if self.reg3 is not None:
            gw += self.b * abs(self.b) * self.reg3 * 3
        return gw

    def update(self, dw):
        self.b = self.b + dw

#---------------------------------------------------------------------------------------------------


class Reshape(Function):

    def __init__(self, shape):
        self.shape = shape

    def calculate_y(self, x):
        return x.reshape(-1, *self.shape)
        
    def calculate_gx(self, x, y, gy):
        return gy.reshape(-1, *x.shape[1:])

#---------------------------------------------------------------------------------------------------


class Relu(Function):

    def calculate_y(self, x):
        return np.maximum(x, 0)
        
    def calculate_gx(self, x, y, gy):
        return gy * (x > 0).astype(float)

#---------------------------------------------------------------------------------------------------


class Sigmoid(Function):

    def calculate_y(self, x):
        return 1 / (1 + np.exp(-x))
        
    def calculate_gx(self, x, y, gy):
        return gy * y * (1 - y)

# ---------------------------------------------------------------------------------------------------


class Tanh(Function):
    def calculate_y(self, x):
        x_norm = np.minimum(10, np.maximum(-10, x))
        x_e = np.exp(-2 * x_norm)
        return (1 - x_e) / (1 + x_e)

    def calculate_gx(self, x, y, gy):
        return gy * (1 - y * y)

# ---------------------------------------------------------------------------------------------------


class TanhRelu(Function):
    def calculate_y(self, x):
        x_norm = np.minimum(10, np.maximum(-10, x))
        x_e = np.exp(-2 * x_norm)
        return (1 - x_e) / (1 + x_e)

    def calculate_gx(self, x, y, gy):
        return gy * (x > 0).astype(float)

# ---------------------------------------------------------------------------------------------------


class TanhLin(Function):
    def calculate_y(self, x):
        x_norm = np.minimum(10, np.maximum(-10, x))
        x_e = np.exp(-2 * x_norm)
        return (1 - x_e) / (1 + x_e)

    def calculate_gx(self, x, y, gy):
        return gy * (x > 0).astype(float)

#---------------------------------------------------------------------------------------------------


class MaxPool(Function):

    def calculate_y(self, x):
        return x * indicator_of_max(x)
        
    def calculate_gx(self, x, y, gy):
        return gy * indicator_of_max(x)

def indicator_of_max(x):
    last_axis = len(x.shape) - 1
    return (x == x.max(axis=last_axis, keepdims=True)).astype(float)

#---------------------------------------------------------------------------------------------------


class Dropout(Function):

    def __init__(self, p):
        self.p = p
        self.mask = None

    def reset_batch_data(self):
        self.mask = None

    def calculate_y(self, x):
        if self.mask is None:
            self.mask = np.array(np.random.random(x.shape[1:]) > self.p, dtype='float')
        else:
            assert self.mask.shape == x.shape
        return x * self.mask / (1 - self.p)
        
    def calculate_gx(self, x, y, gy):
        assert self.mask is not None
        return gy * self.mask / (1 - self.p)

# ---------------------------------------------------------------------------------------------------


class BatchNorm(Function):
    def __init__(self):
        self.sigma = None

    def reset_batch_data(self):
        self.sigma = None

    def calculate_y(self, x):
        y = x - np.mean(x, axis=0, keepdims=True)
        self.sigma = np.maximum(0.001, np.std(y, axis=0, keepdims=True))
        y = y / self.sigma
        return y

    def calculate_gx(self, x, y, gy):
        return gy / self.sigma

    def __str__(self):
        return '\n'.join([
            'sigma:', indent(str(self.sigma))])

# ---------------------------------------------------------------------------------------------------


class MeanNorm(Function):

    def calculate_y(self, x):
        return x - np.mean(x, axis=0, keepdims=True)

    def calculate_gx(self, x, y, gy):
        # На самом деле формула должна быть сложней. Ну да ладно.
        return gy

#---------------------------------------------------------------------------------------------------


class GradNorm(Function):

    def __init__(self, grad_sum):
        self.grad_sum = grad_sum

    def calculate_y(self, x):
        return x
        
    def calculate_gx(self, x, y, gy):
        last_axis = len(gy.shape) - 1
        s = np.sum(np.abs(gy), axis=last_axis, keepdims=True) + self.grad_sum / 10
        return gy * self.grad_sum / s

#---------------------------------------------------------------------------------------------------


class SoftMax(Function):

    def calculate_y(self, x):
        last_axis = len(x.shape) - 1
        x_norm = x - np.max(x, axis=last_axis, keepdims=True)
        x_exp = np.exp(x_norm)
        return x_exp / np.sum(x_exp, axis=last_axis, keepdims=True)
        
    def calculate_gx(self, x, y, gy):
        last_axis = len(gy.shape) - 1
        s = np.sum(gy * y, axis=last_axis, keepdims=True)
        return y * (gy - s)

#---------------------------------------------------------------------------------------------------


class Loss(Function):
    """
    Общий предок функций потерь.
    """

    def __init__(self):
        self.ground = None

    def set_ground(self, ground):
        """
        Нужно вызывать перед обработкой каждого мини-пакета. То есть перед calculate_*
        Args:
            ground: one-hot-encoded groundtruth
        """
        self.ground = ground

#---------------------------------------------------------------------------------------------------


class SvmLoss(Loss):

    def calculate_y(self, x):
        """
        Args:
            x: one-hot-encoded predictions
        Returns:
            y: Вклад в loss каждого x. Финальный loss равен np.sum(y)
        """
        assert self.ground is not None

        # Операции векторные, обрабатывают сразу весь мини-пакет,
        # но для простоты в комментариях описываются действия для одного сэмпла.
        
        # Вычисляем оценку истинного варианта
        y = np.sum(x * self.ground, axis=1, keepdims=True)
        # Уменьшаем все оценки на оценку истинного варианта
        y = x - y
        # Увеличиваем на 1 оценки ложных вариантов.
        # Оценка истинного варианта остаётся нулевой.
        y = y + 1 - self.ground
        # Обрезаем оценки нулём
        y = np.maximum(y, 0)
        # Вычисляем суммарные потери
        y = np.sum(y, axis=1)
        # Сейчас в y хранится loss для каждого сэмпла. А нам нужно вклад каждого сэмпла 
        # в финальный loss.
        y = y / x.shape[0]
        return y
        
    def calculate_gx(self, x, y, gy):
        """
        Цепочка backward начинается со слоя потерь, поэтому gy здесь не используется.
        Returns:
            gx: dL/dx - градиент функции потерь по компонентам x.
        """
        assert self.ground is not None
        
        # Операции векторные, обрабатывают сразу весь мини-пакет,
        # но для простоты в комментариях описываются действия для одного сэмпла.
        
        # Вычисляем оценку истинного варианта
        gx = np.sum(x * self.ground, axis=1, keepdims=True)
        # Уменьшаем все оценки на оценку истинного варианта
        gx = x - gx
        # Увеличиваем на 1 оценки ложных вариантов.
        # Оценка истинного варианта остаётся нулевой.
        gx = gx + 1 - self.ground
        # Если оценка меньше нуля, градиент - 0
        gx = np.maximum(gx, 0)
        # Если оценка больше нуля, градиент - 1
        gx[gx > 0.0000001] = 1
        # Градиент истиннго варианта делаем равным минус суммы градиентов ложных вариантов
        gx[np.arange(gx.shape[0]), np.argmax(self.ground, axis=1)] = -np.sum(gx, axis=1)
        # Сейчас в gx хранится градиент loss отдельного сэмпла. А нам нужен градиент финального loss.
        gx = gx / x.shape[0]
        return gx

#---------------------------------------------------------------------------------------------------


class EntropyLoss(Loss):

    def calculate_y(self, x):
        """
        Args:
            x: one-hot-encoded predictions
        Returns:
            y: Вклад в loss каждого x. Финальный loss равен np.sum(y)
        """
        assert self.ground is not None
        x_norm = np.maximum(x, 0.000001)
        y = -np.sum(np.log(x_norm) * self.ground, axis=1)
        y = y / x.shape[0]
        return y
        
    def calculate_gx(self, x, y, gy):
        """
        Цепочка backward начинается со слоя потерь, поэтому gy здесь не используется.
        Returns:
            gx: dL/dx - градиент функции потерь по компонентам x.
        """
        assert self.ground is not None
        x_norm = np.maximum(x, 0.000001)
        gx = -self.ground / x_norm
        gx = gx / x.shape[0]
        return gx

#---------------------------------------------------------------------------------------------------
