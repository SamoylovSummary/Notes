"""
Пакет для построения экспериментальной нейросети на numpy-массивах.
"""

#---------------------------------------------------------------------------------------------------

"""
Проверка того, что модуль переимпортировался в IPython notebook.
По умолчанию изменения в модуле не приводят к изменениям в уже работающем IPython notebook.
Чтобы правки в модуле оперативно отражались в IPython notebook, приходится вставлять эту магию:

%reload_ext autoreload
%autoreload 2

Расширение autoreload довольно глючное, поэтому чтобы убедиться, что модуль переимпортировался, 
используется этот print.
"""
print('toy_net reload')

#---------------------------------------------------------------------------------------------------

from toy_net.common import *
from toy_net.functions import *
from toy_net.net import *
from toy_net.optimizers import *
