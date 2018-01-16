import numpy as np
import math

from toy_net.common import *
from toy_net.optimizers import *

#---------------------------------------------------------------------------------------------------

"""
Проверяем, что модуль переимпортировался в IPython notebook.
"""
print('toy_net.net reload')

#---------------------------------------------------------------------------------------------------


class Node(object):
    """
    Интерфейс узла нейросети. Используется для сборки слоёв в цепочки.
    """

    def reset_batch_data(self):
        """
        Подготовка перед обработкой минипакета.
        Очищает временные данные.
        """
        raise NotImplementedError()

    def forward(self):
        """
        Прямой проход.
        Изначально вызывается у последнего (loss) слоя. Далее по цепочке от конца к началу 
        вызываются остальные слои.
        """
        raise NotImplementedError()

    def backward(self, gy):
        """
        Обратный проход.
        Изначально вызывается у последнего (loss) слоя. Далее по цепочке от конца к началу 
        вызываются остальные слои.
        """
        raise NotImplementedError()

    def update(self, speed):
        """
        Отложенное обновление параметров (весов) слоя.
        Вызывается снаружи по цепочке от конца к началу.
        Args:
            speed: скорость обучения.
        """
        raise NotImplementedError()

#---------------------------------------------------------------------------------------------------


class FunctionBatchData(object):
    """
    Временные данные обработки одного минипакета одним слоем.
    Внутренний класс для Layer.
    """

    def __init__(self):
        self.x = None
        self.y = None
        self.gx = None
        self.gy = None
        self.gw = None

    def accumulate_gradients(self, gx, gy, gw):
        def accumulate(x, y):
            return x + y if x is not None else y
        self.gx = accumulate(self.gx, gx)
        self.gy = accumulate(self.gy, gy)
        self.gw = accumulate(self.gw, gw)

    def __str__(self):
        return '\n'.join([
            'x:', indent(str(self.x)),
            'y:', indent(str(self.y)),
            'gx:', indent(str(self.gx)),
            'gy:', indent(str(self.gy)),
            'gw:', indent(str(self.gw))])

#---------------------------------------------------------------------------------------------------


class Layer(Node):
    """
    Узел нейросети, содержащий один слой. Обёртка над Function для сборки слоёв в цепочки.
    """

    def __init__(self, func, prev=None, optimizer=SGD()):
        self.func = func
        self.prev = prev
        self.optimizer = optimizer
        self.data = FunctionBatchData()
        self.are_weights_updated = False

    def reset_batch_data(self):
        """
        Подготовка перед обработкой минипакета.
        Очищает временные данные.
        """
        self.func.reset_batch_data()
        self.data = FunctionBatchData()
        self.are_weights_updated = False
        if self.prev is not None:
            self.prev.reset_batch_data()

    def forward(self):
        """
        Прямой проход.
        Изначально вызывается у последнего (loss) слоя. Далее по цепочке от конца к началу 
        вызываются остальные слои.
        """
        if self.data.y is not None:
            return self.data.y
        if self.prev is not None:
            self.data.x = self.prev.forward()
        assert self.data.x is not None
        self.data.y = self.func.calculate_y(self.data.x)
        return self.data.y

    def backward(self, gy):
        """
        Обратный проход.
        Изначально вызывается у последнего (loss) слоя. Далее по цепочке от конца к началу 
        вызываются остальные слои.
        """
        gx = self.func.calculate_gx(self.data.x, self.data.y, gy)
        gw = self.func.calculate_gw(self.data.x, self.data.y, gy)
        self.data.accumulate_gradients(gx, gy, gw)
        if self.prev is not None:
            self.prev.backward(gx)

    def update(self, speed):
        """
        Отложенное обновление параметров (весов) слоя.
        Вызывается снаружи по цепочке от конца к началу.
        """
        if self.are_weights_updated:
            return
        self.are_weights_updated = True
        if self.data.gw is not None:
            assert self.optimizer is not None
            dw = self.optimizer.calculate_dw(speed, self.data.gw)
            self.func.update(dw)
        if self.prev is not None:
            self.prev.update(speed)

    def __str__(self):
        return '\n'.join([
            'data:', indent(str(self.data)),
            'func:', indent(str(self.func)),
            'optimizer:', indent(str(self.optimizer))])

#---------------------------------------------------------------------------------------------------


class ConcatenateNode(Node):
    """
    Узел, объединяющий несколько входных слоёв на манер numpy.concatenate.
    Размерности входных слоёв должны быть согласованы следующим образом:
        (BatchSize, XA, X1, X2, ...)
        (BatchSize, XB, X1, X2, ...)
        (BatchSize, XC, X1, X2, ...)
    В приведённом примере размерность выхода будет такая:
        (BatchSize, XA+XB+XC, X1, X2, ...)
    """

    def __init__(self, prevs=None):
        self.prevs = prevs if prevs is not None else list()
        self.splits = None

    def reset_batch_data(self):
        self.splits = None
        for prev in self.prevs:
            prev.reset_batch_data()

    def forward(self):
        xs = [prev.forward() for prev in self.prevs]
        self.splits = integrate([x.shape[1] for x in xs])[:-1]
        return np.concatenate(xs, axis=1)

    def backward(self, gy):
        gy_parts = np.split(gy, self.splits, axis=1)
        assert len(gy_parts) == len(self.prevs)
        for i in len(gy_parts):
            self.prevs[i].backward(gy_parts[i])

    def update(self, speed):
        for prev in self.prevs:
            prev.update(speed)


def integrate(a):
    b = a[:]
    for i in range(1, len(b)):
        b[i] = b[i - 1] + b[i]
    return b

#---------------------------------------------------------------------------------------------------


class StackNode(Node):
    """
    Узел, объединяющий несколько входных слоёв на манер numpy.stack.
    Размерности входных слоёв должны быть одинаковыми:
        (BatchSize, X1, X2, ...)
        (BatchSize, X1, X2, ...)
        (BatchSize, X1, X2, ...)
    В приведённом примере размерность выхода будет такая:
        (BatchSize, 3, X1, X2, ...)
    """

    def __init__(self, prevs=None):
        self.prevs = prevs if prevs is not None else list()

    def reset_batch_data(self):
        for prev in self.prevs:
            prev.reset_batch_data()

    def forward(self):
        xs = [prev.forward() for prev in self.prevs]
        return np.stack(xs, axis=1)

    def backward(self, gy):
        gy_parts = np.split(gy, 1, axis=1)
        assert len(gy_parts) == len(self.prevs)
        for i in len(gy_parts):
            self.prevs[i].backward(gy_parts[i])

    def update(self, speed):
        for prev in self.prevs:
            prev.update(speed)

#---------------------------------------------------------------------------------------------------


class Net(object):

    def __init__(self, input_layers, loss_layer):
        self.input_layers = input_layers
        self.loss_layer = loss_layer

    def prepare_batch(self, xs):
        self.loss_layer.reset_batch_data()
        assert len(self.input_layers) == len(xs)
        for i, x in enumerate(xs):
            self.input_layers[i].data.x = x

    def forward(self, xs, ground):
        self.prepare_batch(xs)
        self.loss_layer.func.set_ground(ground)
        self.loss_layer.forward()
        return self.loss_layer.data.x, self.loss_layer.data.y

    def train(self, xs, ground, speed):
        y, loss = self.forward(xs, ground)
        self.loss_layer.backward(None)
        self.loss_layer.update(speed)
        return np.sum(loss)

    def calculate_accuracy(self, xs, ground):
        y, loss = self.forward(xs, ground)
        return np.mean(np.argmax(y, axis=1) == np.argmax(ground, axis=1))

    def calculate_loss(self, xs, ground):
        y, loss = self.forward(xs, ground)
        return np.sum(loss)

    def predict(self, xs):
        self.prepare_batch(xs)
        return self.loss_layer.prev.forward()

#---------------------------------------------------------------------------------------------------


def build_sequence_net(functions, optimizer=SGD):
    layers = list()
    for func in functions:
        last_layer = layers[-1] if layers else None
        layers.append(Layer(func, last_layer, optimizer()))
    net = Net([layers[0]], layers[-1])
    return net, layers

#---------------------------------------------------------------------------------------------------
