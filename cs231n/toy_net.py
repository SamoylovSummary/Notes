import numpy as np
import math

#---------------------------------------------------------------------------------------------------

"""
Проверка того, что модуль переимпортировался в IPython notebook.
По умолчанию изменения в модуле не приводят к изменениям в уже работающем IPython notebook.
Чтобы правки в модуле оперативно отражались в IPython notebook, приходится вставлять эту магию:

%reload_ext autoreload
%aimport toy_net
%autoreload 1

А чтобы убедиться, что модуль переимпортировался, используется print ниже.

Имейте в виду, что расширение autoreload очень глючное. Особенно оно не любит, когда импорт модуля 
завершается ошибкой. После исправления этой ошибки autoreload начинает падать при переимпорте.
Но при этом сам переимпорт вроде бы происходит.
"""
print('toy_net reload')

#---------------------------------------------------------------------------------------------------
# Вспомогательные функции.


def indent(text, prefix='  '):
    """Добавляет в начало каждой строки текста text сдвиг prefix"""
    lines = text.split('\n')
    lines = [prefix + line for line in lines]
    # Если последняя строка была пустая, не сдвигаем её.
    if lines[-1] == prefix:
        lines[-1] = ''
    return '\n'.join(lines)

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
            'func:', indent(str(self.func))])

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
        w_std = np.log10(np.std(self.w, axis=0))
        s += '  column std log10 min: %.2f, max: %.2f\n' % (w_std.min(), w_std.max())
        y_std = np.log10(np.std(np.dot(x, self.w), axis=0))
        s += '  y std log10 min: %.2f, max: %.2f' % (y_std.min(), y_std.max())
        return s

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


def build_sequence_net(functions):
    layers = list()
    for func in functions:
        last_layer = layers[-1] if layers else None
        layers.append(Layer(func, last_layer))
    net = Net([layers[0]], layers[-1])
    return net, layers

#---------------------------------------------------------------------------------------------------
